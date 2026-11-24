"""Offline regressions for Pulse audio timing, resampling, and WebSocket delivery."""
import asyncio
import base64
import json
import math
import struct
import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pulse_pipecat_server as server
from fastapi.testclient import TestClient
from pipecat.frames.frames import EndFrame, InterruptionFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketCallbacks,
    FastAPIWebsocketClient,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner
from starlette.websockets import WebSocket


def tone(sample_rate=8000, seconds=0.4):
    samples = [
        round(10000 * math.sin(2 * math.pi * 440 * n / sample_rate))
        for n in range(round(sample_rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


async def serializer(wire_rate=8000, pipeline_rate=8000):
    result = server.PulseFrameSerializer("offline-test", sample_rate=wire_rate)
    await result.setup(SimpleNamespace(audio_in_sample_rate=pipeline_rate))
    return result


def audio_frame(pcm, sample_rate=8000, channels=1):
    return OutputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=channels)


def decode(message):
    return base64.b64decode(json.loads(message)["media"]["payload"], validate=True)


def websocket_scope():
    return {
        "type": "websocket", "asgi": {"version": "3.0"},
        "scheme": "ws", "path": "/pulse-stream", "raw_path": b"/pulse-stream",
        "query_string": b"", "headers": [], "subprotocols": [],
        "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8080),
    }


class PulseSerializerTests(unittest.IsolatedAsyncioTestCase):
    async def test_packet_timestamps_cover_the_actual_audio_duration(self):
        clock = SimpleNamespace(monotonic=lambda: 100.0)
        with patch.object(server, "time", clock):
            codec = await serializer()
            pcm = tone(seconds=0.04)
            messages = [
                json.loads(await codec.serialize(audio_frame(pcm)))
                for _ in range(3)
            ]
        self.assertEqual([int(m["media"]["timestamp"]) for m in messages], [0, 40, 80])
        self.assertEqual([len(base64.b64decode(m["media"]["payload"])) for m in messages], [640] * 3)
        self.assertTrue(any(pcm))
        self.assertFalse(base64.b64decode(messages[0]["media"]["payload"]).startswith(b"RIFF"))

    async def test_real_pauses_and_interruptions_do_not_reset_the_clock(self):
        clock = SimpleNamespace(value=100.0)
        with patch.object(server, "time", SimpleNamespace(monotonic=lambda: clock.value)):
            codec = await serializer()
            await codec.serialize(audio_frame(tone(seconds=0.04)))
            clock.value += 2.0
            resumed = json.loads(await codec.serialize(audio_frame(tone(seconds=0.04))))
            clear = json.loads(await codec.serialize(InterruptionFrame()))
            after_clear = json.loads(await codec.serialize(audio_frame(tone(seconds=0.04))))
        self.assertEqual(int(resumed["media"]["timestamp"]), 2000)
        self.assertEqual(clear["event"], "clear")
        self.assertEqual(int(after_clear["media"]["timestamp"]), 2040)

    async def test_24khz_tts_retains_pitch_and_level_after_resampling(self):
        codec = await serializer()
        pcm = decode(await codec.serialize(audio_frame(tone(24000, 1.0), 24000)))
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        self.assertGreater(len(samples), 6000)
        self.assertLessEqual(len(samples), 8100)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        # The quiet startup transient can add zero crossings; measure the dominant tone.
        waveform = np.asarray(samples, dtype=float)
        spectrum = np.abs(np.fft.rfft(waveform * np.hanning(len(waveform))))
        frequency = np.fft.rfftfreq(len(waveform), 1 / 8000)[int(np.argmax(spectrum))]
        self.assertGreater(rms, 4000)
        self.assertAlmostEqual(frequency, 440, delta=5)

    async def test_advertised_16khz_input_is_resampled_for_the_pipeline(self):
        codec = await serializer(wire_rate=16000, pipeline_rate=8000)
        frame = await codec.deserialize(json.dumps({
            "event": "media", "media": {
                "payload": base64.b64encode(tone(16000, 1.0)).decode("ascii")
            },
        }))
        self.assertEqual(frame.sample_rate, 8000)
        self.assertEqual(frame.num_channels, 1)
        self.assertGreater(len(frame.audio), 12000)
        self.assertLessEqual(len(frame.audio), 16200)

    async def test_malformed_audio_is_dropped_instead_of_entering_the_pipeline(self):
        codec = await serializer()
        for payload in ["not base64!", base64.b64encode(b"x").decode("ascii")]:
            with self.subTest(payload=payload):
                self.assertIsNone(await codec.deserialize(json.dumps({
                    "event": "media", "media": {"payload": payload}
                })))

    async def test_stereo_output_is_rejected_before_encoding(self):
        codec = await serializer()
        with self.assertRaisesRegex(ValueError, "mono"):
            await codec.serialize(audio_frame(tone(seconds=0.04), channels=2))


class PulseFormatTests(unittest.TestCase):
    def test_advertised_pcm_rate_is_used(self):
        self.assertEqual(server.get_pulse_sample_rate({"media_format": {
            "encoding": "audio/x-l16", "sample_rate": "16000", "channels": 1
        }}), 16000)
        self.assertEqual(server.get_pulse_sample_rate({}), 8000)

    def test_incompatible_codec_channels_and_rate_are_reported(self):
        for media_format in [
            {"encoding": "audio/x-mulaw"},
            {"channels": 2},
            {"sample_rate": 48000},
        ]:
            with self.subTest(media_format=media_format):
                with self.assertRaises(ValueError):
                    server.get_pulse_sample_rate({"media_format": media_format})


class PulseWebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonzero_audio_crosses_the_real_pipecat_transport(self):
        sent = []

        async def asgi_send(message):
            sent.append(message)

        websocket = WebSocket(websocket_scope(), AsyncMock(
            return_value={"type": "websocket.connect"}), asgi_send)
        await websocket.accept()
        tracked = server.PulseAudioWebSocket(websocket, 8000)
        codec = server.PulseFrameSerializer("transport-test")
        transport = FastAPIWebsocketTransport(
            websocket=tracked,
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                audio_out_10ms_chunks=20,
                audio_out_auto_silence=False,
                audio_out_end_silence_secs=0,
                add_wav_header=False,
                serializer=codec,
            ),
        )
        worker = PipelineWorker(
            Pipeline([transport.output()]),
            params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
            processor_unusable_policy=ProcessorUnusablePolicy.END,
        )
        runner = WorkerRunner()
        await runner.add_workers(worker)
        pcm = tone(seconds=0.4)
        await worker.queue_frames([audio_frame(pcm), EndFrame()])
        await asyncio.wait_for(runner.run(), timeout=10)

        messages = [
            json.loads(event["text"]) for event in sent
            if event["type"] == "websocket.send" and event.get("text")
        ]
        media = [message for message in messages if message.get("event") == "media"]
        delivered = b"".join(base64.b64decode(message["media"]["payload"]) for message in media)
        self.assertTrue(media)
        self.assertEqual([len(base64.b64decode(m["media"]["payload"])) for m in media], [3200, 3200])
        self.assertEqual(delivered, pcm)
        self.assertEqual(tracked.sent_packets, len(media))
        self.assertEqual(tracked.sent_bytes, len(pcm))
        self.assertEqual(tracked.failed_sends, 0)
        for previous, current in zip(media, media[1:]):
            duration = len(base64.b64decode(previous["media"]["payload"])) * 1000 // 16000
            self.assertGreaterEqual(
                int(current["media"]["timestamp"]) - int(previous["media"]["timestamp"]),
                duration,
            )

    async def test_failed_send_is_counted_even_when_pipecat_swallows_the_error(self):
        async def asgi_send(message):
            if message["type"] == "websocket.send":
                raise OSError("offline send failure")

        websocket = WebSocket(websocket_scope(), AsyncMock(
            return_value={"type": "websocket.connect"}), asgi_send)
        await websocket.accept()
        tracked = server.PulseAudioWebSocket(websocket, 8000)
        client = FastAPIWebsocketClient(
            tracked,
            FastAPIWebsocketCallbacks(
                on_client_connected=AsyncMock(),
                on_client_disconnected=AsyncMock(),
                on_session_timeout=AsyncMock(),
            ),
        )
        codec = await serializer()
        await client.send(await codec.serialize(audio_frame(tone(seconds=0.04))))
        self.assertEqual(tracked.sent_packets, 0)
        self.assertEqual(tracked.sent_bytes, 0)
        self.assertEqual(tracked.failed_sends, 1)


class PulseHandshakeTests(unittest.TestCase):
    def test_nested_stream_id_and_audio_format_reach_the_pipeline(self):
        received = []

        async def capture(websocket, stream_sid, customer_info, *, wire_sample_rate):
            received.append((stream_sid, wire_sample_rate))
            await websocket.send_json({"event": "offline-handshake-ok"})

        with patch.object(server, "run_pipecat_pipeline", capture):
            with TestClient(server.app) as client:
                with client.websocket_connect("/pulse-stream") as websocket:
                    connected = {"event": "connected", "protocol": "Pulse"}
                    websocket.send_json(connected)
                    self.assertEqual(websocket.receive_json(), connected)
                    start = {"event": "start", "start": {
                        "stream_sid": "nested-id",
                        "media_format": {"encoding": "audio/x-l16", "sample_rate": 16000},
                    }}
                    websocket.send_json(start)
                    self.assertEqual(websocket.receive_json(), start)
                    self.assertEqual(websocket.receive_json()["event"], "offline-handshake-ok")
        self.assertEqual(received, [("nested-id", 16000)])


class PulsePlaybackContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_outgoing_audio_matches_the_pulse_documented_json_fields(self):
        codec = await serializer()
        message = json.loads(await codec.serialize(audio_frame(tone(seconds=0.2))))
        self.assertEqual(set(message), {"event", "stream_sid", "sequence_number", "media"})
        self.assertEqual(set(message["media"]), {"chunk", "timestamp", "payload"})
        self.assertEqual(message["event"], "media")
        self.assertEqual(message["stream_sid"], "offline-test")
        self.assertIsInstance(message["sequence_number"], int)
        self.assertIsInstance(message["media"]["chunk"], int)
        self.assertIsInstance(message["media"]["timestamp"], str)
        self.assertEqual(decode(json.dumps(message)), tone(seconds=0.2))


class PulseEchoEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    def incoming(self, *, index=0, sample_rate=8000):
        return {
            "event": "media", "sequence_number": 45 + index,
            "stream_sid": "working-echo",
            "media": {"chunk": 43 + index, "timestamp": str(981 + index * 20),
                      "payload": base64.b64encode(tone(sample_rate, 0.02)).decode("ascii")},
        }

    def socket(self, messages):
        return SimpleNamespace(
            receive=AsyncMock(side_effect=[
                {"type": "websocket.receive", "text": json.dumps(message)}
                for message in messages
            ]),
            send_text=AsyncMock(), close=AsyncMock(),
        )

    async def test_speech_uses_the_live_echo_envelope_and_stt_keeps_caller_audio(self):
        incoming = self.incoming()
        socket = self.socket([incoming])
        tracked = server.PulseAudioWebSocket(socket, 8000, use_echo_envelope=True)
        pcm = tone(seconds=0.02)[::-1]  # Different bytes from the caller's audio.
        codec = await serializer()
        await tracked.send_text(await codec.serialize(audio_frame(pcm)))
        socket.send_text.assert_not_awaited()
        self.assertEqual(tracked.sent_generated_bytes, 0)
        received = await tracked.receive()
        outgoing = json.loads(socket.send_text.call_args.args[0])
        self.assertEqual(json.loads(received["text"]), incoming)
        self.assertEqual(decode(json.dumps(outgoing)), pcm)
        outgoing["media"]["payload"] = incoming["media"]["payload"]
        self.assertEqual(outgoing, incoming)
        self.assertEqual(tracked.sent_generated_bytes, 320)
        self.assertEqual(tracked.queued_audio_bytes, 0)

    async def test_idle_playback_is_silent_and_partial_speech_is_padded(self):
        socket = self.socket([self.incoming(index=i) for i in range(3)])
        tracked = server.PulseAudioWebSocket(socket, 8000, use_echo_envelope=True)
        pcm = tone(seconds=0.03)
        codec = await serializer()
        await tracked.send_text(await codec.serialize(audio_frame(pcm)))
        for _ in range(3):
            await tracked.receive()
        sent = [decode(call.args[0]) for call in socket.send_text.call_args_list]
        self.assertEqual(sent, [pcm[:320], pcm[320:] + bytes(160), bytes(320)])
        self.assertEqual(tracked.sent_generated_bytes, len(pcm))

    async def test_interruption_discards_buffered_speech_without_resetting_pulse_counters(self):
        socket = self.socket([self.incoming(index=10)])
        tracked = server.PulseAudioWebSocket(socket, 8000, use_echo_envelope=True)
        codec = await serializer()
        await tracked.send_text(await codec.serialize(audio_frame(tone(seconds=0.04))))
        await tracked.send_text(await codec.serialize(InterruptionFrame()))
        self.assertEqual(tracked.queued_audio_bytes, 0)
        self.assertEqual(tracked.sent_clears, 1)
        await tracked.receive()
        outgoing = json.loads(socket.send_text.call_args.args[0])
        self.assertEqual(outgoing["sequence_number"], 55)
        self.assertEqual(outgoing["media"]["timestamp"], "1181")
        self.assertEqual(decode(json.dumps(outgoing)), bytes(320))

    async def test_playback_failures_are_counted_at_the_actual_send(self):
        socket = self.socket([self.incoming()])
        socket.send_text.side_effect = OSError("offline playback failure")
        tracked = server.PulseAudioWebSocket(socket, 8000, use_echo_envelope=True)
        codec = await serializer()
        await tracked.send_text(await codec.serialize(audio_frame(tone(seconds=0.02))))
        with self.assertRaises(OSError):
            await tracked.receive()
        self.assertEqual(tracked.failed_sends, 1)
        self.assertEqual(tracked.sent_generated_bytes, 0)

    async def test_beep_uses_the_echo_envelope_and_negotiated_pcm_rate(self):
        incoming = [self.incoming(index=i, sample_rate=16000) for i in range(150)]
        socket = self.socket(incoming)
        await server.run_pulse_tone_test(socket, "working-echo", wire_sample_rate=16000)
        sent = [json.loads(call.args[0]) for call in socket.send_text.call_args_list]
        self.assertEqual(len(sent), 150)
        for original, outgoing in zip(incoming, sent):
            self.assertEqual(len(decode(json.dumps(outgoing))), 640)
            outgoing["media"]["payload"] = original["media"]["payload"]
            self.assertEqual(outgoing, original)
        pcm = b"".join(decode(call.args[0]) for call in socket.send_text.call_args_list)
        samples = np.frombuffer(pcm, dtype="<i2").astype(float)
        peak_bin = int(np.argmax(np.abs(np.fft.rfft(samples))))
        frequency = np.fft.rfftfreq(len(samples), 1 / 16000)[peak_bin]
        self.assertAlmostEqual(frequency, 440, delta=1)
        self.assertGreater(np.sqrt(np.mean(samples * samples)), 10000)
        socket.close.assert_awaited_once()


class PulseEchoTests(unittest.TestCase):
    def test_phone_audio_test_returns_pulse_messages_without_running_ai(self):
        ai = AsyncMock()
        with patch.object(server, "run_pipecat_pipeline", ai):
            with TestClient(server.app) as client:
                with client.websocket_connect("/pulse-stream?mode=echo") as ws:
                    connected = '{"event":"connected"}'
                    ws.send_text(connected)
                    self.assertEqual(ws.receive_text(), connected)
                    start = json.dumps({"event": "start", "sequence_number": 1,
                        "stream_sid": "echo-test", "start": {"stream_sid": "echo-test",
                        "media_format": {"encoding": "audio/x-l16", "sample_rate": 8000}}})
                    ws.send_text(start)
                    self.assertEqual(ws.receive_text(), start)
                    media = json.dumps({"event": "media", "stream_sid": "echo-test",
                        "sequence_number": 3, "media": {"chunk": 2, "timestamp": "10",
                        "payload": base64.b64encode(tone(seconds=0.02)).decode("ascii")}})
                    ws.send_text(media)
                    self.assertEqual(ws.receive_text(), media)
                    ws.send_json({"event": "stop"})
        ai.assert_not_awaited()

    def test_audio_test_call_uses_the_echo_stream_url(self):
        response = {"success": True, "unique_call_identifier": "offline-echo-call"}
        with patch.object(server.pulse_service, "initiate_click_to_call", return_value=response) as dial:
            with TestClient(server.app) as client:
                result = client.post("/api/call", json={
                    "name": "Audio Test", "phone_number": "9876543210", "audio_test": True})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(dial.call_args.kwargs["ws_url"].endswith("/pulse-stream?mode=echo"))



if __name__ == "__main__":
    unittest.main(verbosity=2)
