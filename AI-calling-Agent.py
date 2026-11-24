import os
import json
import uuid
import asyncio
import base64
import time
import wave
import requests
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
load_dotenv(override=True)

# ------------------------------------------------------------------------------
# Helper to check if an API key is actually set (not empty and not a placeholder)
# ------------------------------------------------------------------------------
def is_valid_key(key: Optional[str]) -> bool:
    return bool(key and not key.startswith("your_") and not key.startswith("..."))

# ------------------------------------------------------------------------------
# 1. Environment Configurations
# ------------------------------------------------------------------------------
PULSE_AUTH_URL = os.getenv("PULSE_AUTH_URL", "https://connecthub.pulsework360.com/auth/token")
PULSE_C2C_URL = os.getenv("PULSE_C2C_URL", "https://jxoclvto1l.execute-api.ap-south-1.amazonaws.com/v1/")
PULSE_ACCOUNT_CODE = os.getenv("PULSE_ACCOUNT_CODE", "")
PULSE_MEMBER_NAME = os.getenv("PULSE_MEMBER_NAME", "")
PULSE_MEMBER_PASSWORD = os.getenv("PULSE_MEMBER_PASSWORD", "")
PULSE_VIRTUAL_NUMBER = os.getenv("PULSE_VIRTUAL_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
PORT = int(os.getenv("PORT", 8080))

# AI Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# The measured 3.1 Flash-Lite response is faster for this phone conversation.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "128"))
USE_LOCAL_STT = os.getenv("USE_LOCAL_STT", "false").lower() == "true"

# ------------------------------------------------------------------------------
# 2. Pulse Telephony Service
# ------------------------------------------------------------------------------
class PulseService:
    def __init__(self):
        self.token = None

    def get_auth_token(self) -> str:
        payload = {
            "accountcode": PULSE_ACCOUNT_CODE,
            "membername": PULSE_MEMBER_NAME,
            "memberpassword": PULSE_MEMBER_PASSWORD,
        }
        res = requests.post(PULSE_AUTH_URL, json=payload, timeout=10)
        if res.status_code == 200:
            self.token = res.json().get("token")
            logger.info("Pulse token authenticated successfully.")
            return self.token
        else:
            logger.error(f"Pulse Auth failed: {res.text}")
            raise Exception(f"Failed to authenticate with Pulse: {res.text}")

    def initiate_click_to_call(self, customer_phone: str, ws_url: str, custom_params: dict = None) -> dict:
        if not self.token:
            self.get_auth_token()

        unique_id = str(uuid.uuid4())
        cleaned_phone = customer_phone.replace("+", "").replace(" ", "").replace("-", "")
        if cleaned_phone.startswith("91") and len(cleaned_phone) == 12:
            cleaned_phone = cleaned_phone[2:]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        body = {
            "ws_url": ws_url,
            "agent_phone_number": PULSE_VIRTUAL_NUMBER,
            "user_phone_number": cleaned_phone,
            "unique_call_identifier": unique_id,
            "custom_parameters": custom_params or {},
        }

        logger.info(f"Dialing {cleaned_phone} via Pulse Virtual Number ({PULSE_VIRTUAL_NUMBER})...")
        res = requests.post(PULSE_C2C_URL, json=body, headers=headers, timeout=15)

        if res.status_code in [401, 403]:
            logger.warning("Pulse token expired, refreshing...")
            self.get_auth_token()
            headers["Authorization"] = f"Bearer {self.token}"
            res = requests.post(PULSE_C2C_URL, json=body, headers=headers, timeout=15)

        if res.status_code in [200, 201]:
            logger.info(f"Pulse call placed successfully: {res.text}")
            return {"success": True, "unique_call_identifier": unique_id, "data": res.json()}
        else:
            logger.error(f"Pulse call failed ({res.status_code}): {res.text}")
            return {"success": False, "error": res.text, "status_code": res.status_code}

pulse_service = PulseService()

# ------------------------------------------------------------------------------
# 3. Pulse Audio Stream Serializer (audio/x-l16, 8000Hz, 16-bit Linear PCM)
# ------------------------------------------------------------------------------
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    InterruptionFrame,
    TTSSpeakFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)

def get_pulse_sample_rate(start: dict) -> int:
    """Validate the advertised wire format before treating its payload as PCM16."""
    media_format = start.get("media_format") or start.get("mediaFormat") or {}
    if not isinstance(media_format, dict):
        raise ValueError("Pulse start.media_format must be an object.")
    encoding = str(media_format.get("encoding", "audio/x-l16")).lower().split(";")[0].strip()
    if encoding not in {"audio/x-l16", "pcm_s16le", "linear16", "slin"}:
        raise ValueError(
            f"Pulse advertised unsupported audio encoding {encoding!r}; "
            "this server expects little-endian PCM16. Match the Pulse codec before retrying."
        )
    sample_rate = int(media_format.get("sample_rate", media_format.get("sampleRate", 8000)))
    channels = int(media_format.get("channels", 1))
    if sample_rate not in {8000, 16000, 24000}:
        raise ValueError(f"Unsupported Pulse PCM sample rate: {sample_rate} Hz.")
    if channels != 1:
        raise ValueError(f"Pulse must provide mono audio; advertised channels={channels}.")
    return sample_rate


class PulseFrameSerializer(FrameSerializer):
    """Convert Pipecat PCM16 audio to the Pulse JSON media protocol."""

    def __init__(self, stream_sid: str, sample_rate: int = 8000):
        super().__init__(
            params=FrameSerializer.InputParams(
                audio_in_sample_rate=sample_rate,
                audio_out_sample_rate=sample_rate,
            )
        )
        self._stream_sid = stream_sid
        self._seq = 0
        self._stream_started_at = time.monotonic()
        self._next_audio_sample = 0

    @property
    def outbound_packets(self) -> int:
        return self._seq

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            return json.dumps({"event": "clear", "stream_sid": self._stream_sid})
        elif isinstance(frame, AudioRawFrame):
            if frame.num_channels != 1:
                raise ValueError("Pulse output must be mono PCM16 audio.")
            pcm_data = await self._resample_audio_out(frame)
            if not pcm_data:
                return None
            if len(pcm_data) % 2:
                raise ValueError("Pulse PCM16 output contains an incomplete sample.")

            sample_rate = self.audio_out_sample_rate
            # Timestamps describe the audio timeline, including pauses. The
            # transport can emit packets of different lengths after resampling.
            elapsed_samples = int((time.monotonic() - self._stream_started_at) * sample_rate)
            first_sample = max(elapsed_samples, self._next_audio_sample)
            timestamp_ms = first_sample * 1000 // sample_rate
            self._next_audio_sample = first_sample + len(pcm_data) // 2
            self._seq += 1
            message = {
                "event": "media",
                "stream_sid": self._stream_sid,
                "sequence_number": self._seq,
                "media": {
                    "chunk": self._seq,
                    "timestamp": str(timestamp_ms),
                    "payload": base64.b64encode(pcm_data).decode("ascii"),
                },
            }
            if self._seq == 1 or self._seq % 50 == 0:
                logger.info(
                    f"Prepared Pulse audio: packet #{self._seq}, "
                    f"{len(pcm_data)} bytes at {sample_rate} Hz, timestamp={timestamp_ms}ms."
                )
            return json.dumps(message)
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
        except (ValueError, TypeError):
            logger.warning("Ignoring a Pulse message that is not valid JSON.")
            return None
        if not isinstance(message, dict):
            return None

        event = message.get("event")
        if event == "media":
            media = message.get("media") or {}
            if not isinstance(media, dict):
                logger.warning("Ignoring a Pulse media message with an invalid body.")
                return None
            payload = media.get("payload")
            if payload:
                try:
                    raw_pcm = base64.b64decode(payload, validate=True)
                except (ValueError, TypeError):
                    logger.warning("Ignoring invalid base64 in a Pulse audio packet.")
                    return None
                if len(raw_pcm) % 2:
                    logger.warning("Ignoring Pulse PCM16 audio with an incomplete sample.")
                    return None
                resampled = await self._resample_audio_in(raw_pcm)
                if not resampled:
                    return None
                return InputAudioRawFrame(
                    audio=resampled,
                    num_channels=1,
                    sample_rate=self._sample_rate,
                )
        elif event in ("stop", "ended"):
            return EndFrame()
        elif event == "error":
            details = message.get("error", message.get("message", message.get("code", "unspecified")))
            logger.error(f"Pulse reported a stream error: {details}")

        return None


# ------------------------------------------------------------------------------
# Call Audio Recording Service
# ------------------------------------------------------------------------------
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
RECORDINGS_META_FILE = os.path.join(RECORDINGS_DIR, "recordings.json")

def load_recordings_metadata() -> list:
    if os.path.exists(RECORDINGS_META_FILE):
        try:
            with open(RECORDINGS_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read recordings metadata: {e}")
    return []

def save_recordings_metadata(records: list):
    try:
        with open(RECORDINGS_META_FILE, "w", encoding="utf-8") as f:
            json.dump(records[:200], f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save recordings metadata: {e}")

class CallRecorder:
    """Records telephone call audio (caller + agent speech mixed) into standard WAV format."""
    def __init__(self, stream_sid: str, customer_name: str = "Customer", phone_number: str = "Unknown", sample_rate: int = 8000):
        self.stream_sid = stream_sid
        self.customer_name = customer_name
        self.phone_number = phone_number
        self.sample_rate = sample_rate
        self.filename = f"{stream_sid}.wav"
        self.filepath = os.path.join(RECORDINGS_DIR, self.filename)
        self.started_at = time.time()
        self.ended_at = None
        self.duration_seconds = 0
        self.is_closed = False
        try:
            self._wav = wave.open(self.filepath, "wb")
            self._wav.setnchannels(1)      # Mono
            self._wav.setsampwidth(2)      # 16-bit PCM (2 bytes)
            self._wav.setframerate(sample_rate)
            logger.info(f"Started audio recording for call {stream_sid} -> {self.filename}")
        except Exception as e:
            logger.error(f"Failed to initialize WAV call recorder: {e}")
            self._wav = None
            self.is_closed = True

    def write_pcm(self, pcm_bytes: bytes):
        if not self.is_closed and self._wav and pcm_bytes:
            try:
                self._wav.writeframes(pcm_bytes)
            except Exception as e:
                logger.error(f"Error writing recording frame: {e}")

    def close(self):
        if not self.is_closed:
            self.is_closed = True
            self.ended_at = time.time()
            self.duration_seconds = max(1, int(self.ended_at - self.started_at))
            if self._wav:
                try:
                    self._wav.close()
                    logger.info(f"Call recording saved: {self.filename} ({self.duration_seconds}s)")
                except Exception as e:
                    logger.error(f"Error closing WAV file: {e}")
            self.save_metadata()

    def save_metadata(self):
        try:
            records = load_recordings_metadata()
            file_size_kb = round(os.path.getsize(self.filepath) / 1024, 1) if os.path.exists(self.filepath) else 0
            entry = {
                "id": self.stream_sid,
                "filename": self.filename,
                "url": f"/recordings/{self.filename}",
                "customer_name": self.customer_name,
                "phone_number": self.phone_number,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
                "duration": self.duration_seconds,
                "file_size_kb": file_size_kb,
            }
            records.insert(0, entry)
            save_recordings_metadata(records)
        except Exception as e:
            logger.error(f"Error saving recording metadata: {e}")


class PulseAudioWebSocket:
    """Track actual sends, play generated PCM in Pulse's working echo envelope, and record audio."""

    def __init__(self, websocket: WebSocket, sample_rate: int, *, use_echo_envelope: bool = False, recorder: Optional[CallRecorder] = None):
        self._websocket = websocket
        self.sample_rate = sample_rate
        self._use_echo_envelope = use_echo_envelope
        self.recorder = recorder
        self._pending_audio = bytearray()
        self._logged_envelope = False
        self.sent_packets = 0
        self.sent_bytes = 0
        self.sent_generated_bytes = 0
        self.sent_clears = 0
        self.failed_sends = 0

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    @property
    def queued_audio_bytes(self) -> int:
        return len(self._pending_audio)

    @staticmethod
    def _media_packet(data):
        try:
            message = json.loads(data)
            if not isinstance(message, dict) or message.get("event") != "media":
                return None
            media = message.get("media")
            if not isinstance(media, dict):
                return None
            pcm = base64.b64decode(media.get("payload", ""), validate=True)
            if not pcm or len(pcm) % 2:
                return None
            return message, pcm
        except (ValueError, TypeError):
            return None

    async def send_text(self, data: str):
        if self._use_echo_envelope:
            packet = self._media_packet(data)
            if packet:
                # Pipecat paces this PCM. Pulse's receive loop supplies the
                # actual playback packet size, counters, and timestamp origin.
                self._pending_audio.extend(packet[1])
                return
            try:
                message = json.loads(data)
            except (ValueError, TypeError):
                message = None
            if isinstance(message, dict) and message.get("event") == "clear":
                self._pending_audio.clear()
        await self._send_text(data)

    async def receive(self):
        incoming = await self._websocket.receive()
        if self._use_echo_envelope and incoming.get("type") == "websocket.receive":
            packet = self._media_packet(incoming.get("text", incoming.get("bytes")))
            if packet:
                message, caller_pcm = packet
                packet_bytes = len(caller_pcm)
                generated_bytes = min(packet_bytes, len(self._pending_audio))
                playback = bytes(self._pending_audio[:generated_bytes])
                del self._pending_audio[:generated_bytes]
                # Keep the call's echo protocol active between replies without
                # sending the caller's own voice back to them.
                playback += bytes(packet_bytes - generated_bytes)
                if not self._logged_envelope:
                    self._logged_envelope = True
                    logger.info(
                        f"Using Pulse echo envelope for playback: "
                        f"sequence={message.get('sequence_number')}, "
                        f"chunk={message['media'].get('chunk')}, "
                        f"timestamp={message['media'].get('timestamp')}, "
                        f"packet_bytes={packet_bytes}."
                    )
                # The audible echo test proves this envelope works. Preserve
                # every field and its type; change only the raw PCM payload.
                message["media"]["payload"] = base64.b64encode(playback).decode("ascii")
                await self._send_text(
                    json.dumps(message, separators=(",", ":")),
                    generated_bytes=generated_bytes,
                )
                # Record conversation audio (mix caller PCM and agent voice)
                if self.recorder and not self.recorder.is_closed:
                    if generated_bytes > 0:
                        try:
                            c = np.frombuffer(caller_pcm, dtype=np.int16)
                            a = np.frombuffer(playback[:generated_bytes], dtype=np.int16)
                            mixed = np.clip(c.astype(np.int32) + a.astype(np.int32), -32768, 32767).astype(np.int16).tobytes()
                            self.recorder.write_pcm(mixed)
                        except Exception:
                            self.recorder.write_pcm(caller_pcm)
                    else:
                        self.recorder.write_pcm(caller_pcm)
        # STT must still receive the original caller audio, never the playback.
        return incoming

    async def receive_text(self):
        message = await self.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000), message.get("reason", ""))
        return message["text"]

    async def _send_text(self, data: str, *, generated_bytes: int | None = None):
        try:
            await self._websocket.send_text(data)
        except Exception as exc:
            self.failed_sends += 1
            logger.error(f"Pulse WebSocket send failed: {type(exc).__name__}: {exc}")
            raise

        try:
            message = json.loads(data)
        except (ValueError, TypeError):
            return
        if not isinstance(message, dict):
            return
        if message.get("event") == "media":
            payload = message.get("media", {}).get("payload", "")
            try:
                payload_bytes = len(base64.b64decode(payload, validate=True))
            except (ValueError, TypeError):
                logger.error("An outgoing Pulse packet contained invalid base64.")
                return
            self.sent_packets += 1
            self.sent_bytes += payload_bytes
            speech_bytes = payload_bytes if generated_bytes is None else generated_bytes
            first_speech_packet = speech_bytes > 0 and self.sent_generated_bytes == 0
            self.sent_generated_bytes += speech_bytes
            if first_speech_packet or (speech_bytes and self.sent_packets % 50 == 1):
                logger.info(
                    f"Sent audio to Pulse: packet #{self.sent_packets}, "
                    f"{payload_bytes} bytes ({speech_bytes} generated), "
                    f"PCM16 mono at {self.sample_rate} Hz, "
                    f"Pulse timestamp={message.get('media', {}).get('timestamp')}."
                )
        elif message.get("event") == "clear":
            self.sent_clears += 1
            logger.debug("Sent Pulse clear: interrupted the queued agent audio.")


# ------------------------------------------------------------------------------
# 4. Pipecat Bot Factory
# ------------------------------------------------------------------------------
async def run_pipecat_pipeline(
    websocket: WebSocket, stream_sid: str, customer_info: dict, *, wire_sample_rate: int = 8000
):
    """Initializes and runs the full Pipecat pipeline for the active Pulse call."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )
    from pipecat.workers.runner import WorkerRunner
    from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    # 1. Use the same playback envelope as the confirmed audible echo test with call recording.
    recorder = CallRecorder(
        stream_sid=stream_sid,
        customer_name=customer_info.get("name", "Customer"),
        phone_number=customer_info.get("phone", "Unknown"),
        sample_rate=wire_sample_rate,
    )
    serializer = PulseFrameSerializer(stream_sid=stream_sid, sample_rate=wire_sample_rate)
    audio_websocket = PulseAudioWebSocket(websocket, wire_sample_rate, use_echo_envelope=True, recorder=recorder)

    transport = FastAPIWebsocketTransport(
        websocket=audio_websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # Queue 20 ms of speech at a time. Incoming Pulse packets clock
            # playback using their exact envelope and PCM packet size.
            audio_out_10ms_chunks=2,
            audio_out_auto_silence=False,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # 2. STT (Speech to Text) - Local Moonshine, Groq Whisper, or Deepgram
    if USE_LOCAL_STT:
        from pipecat.services.moonshine.stt import (
            MoonshineSTTService,
            Model as MoonshineModel,
        )
        from pipecat.transcriptions.language import Language

        logger.info("Using local Moonshine STT (Moonshine streaming ONNX)...")
        stt = MoonshineSTTService(
            ttfs_p99_latency=0.5,
            settings=MoonshineSTTService.Settings(
                model=MoonshineModel.SMALL_STREAMING,
                language=Language.EN,
            ),
        )
    elif is_valid_key(GROQ_API_KEY):
        from pipecat.services.groq.stt import GroqSTTService
        logger.info("Using Groq Whisper STT (whisper-large-v3-turbo)...")
        stt = GroqSTTService(
            api_key=GROQ_API_KEY,
            settings=GroqSTTService.Settings(model="whisper-large-v3-turbo"),
        )
    elif is_valid_key(DEEPGRAM_API_KEY):
        from pipecat.services.deepgram.stt import DeepgramSTTService
        logger.info("Using Deepgram STT...")
        stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    else:
        from pipecat.services.moonshine.stt import (
            MoonshineSTTService,
            Model as MoonshineModel,
        )
        from pipecat.transcriptions.language import Language

        logger.info("No cloud STT key found, falling back to local Moonshine STT...")
        stt = MoonshineSTTService(
            ttfs_p99_latency=0.5,
            settings=MoonshineSTTService.Settings(
                model=MoonshineModel.SMALL_STREAMING,
                language=Language.EN,
            ),
        )
    # 3. LLM (The Brain) - Supports OpenAI, Gemini, or Groq
    system_prompt = (
                "1. Reply in one or two short sentences, at most 20 words total. "
        "Make the first sentence 6 to 10 words and end it with a full stop. "
        "Start with a direct answer. "    )

    if is_valid_key(OPENAI_API_KEY):
        from pipecat.services.openai.llm import OpenAILLMService
        logger.info("Using OpenAI LLM (gpt-4o-mini)...")
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            settings=OpenAILLMService.Settings(
                model="gpt-4o-mini",
                system_instruction=system_prompt,
            ),
        )
    elif is_valid_key(GEMINI_API_KEY):
        from pipecat.services.google.llm import GoogleLLMService
        logger.info(
            f"Using Google Gemini LLM ({GEMINI_MODEL}, "
            f"max_output_tokens={GEMINI_MAX_OUTPUT_TOKENS}, lowest supported thinking)..."
        )
        llm = GoogleLLMService(
            api_key=GEMINI_API_KEY,
            settings=GoogleLLMService.Settings(
                model=GEMINI_MODEL,
                system_instruction=system_prompt,
                max_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                # Pipecat selects the model's lowest documented thinking
                # setting, including "minimal" for 3.1 Flash-Lite.
            ),
        )
    elif is_valid_key(GROQ_API_KEY):
        from pipecat.services.openai.llm import OpenAILLMService
        logger.info("Using Groq LLM (llama-3.3-70b-versatile)...")
        llm = OpenAILLMService(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            settings=OpenAILLMService.Settings(system_instruction=system_prompt),
        )
    else:
        raise ValueError("No valid LLM key configured! Please set OPENAI_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY in .env.")

    # 4. TTS (Text to Speech) - Local Kokoro (Free) or Cartesia / ElevenLabs
    USE_KOKORO = os.getenv("USE_KOKORO", "true").lower() == "true"

    if USE_KOKORO:
        from pathlib import Path
        from pipecat.services.kokoro.tts import KokoroTTSService
        from pipecat.transcriptions.language import Language

        model_dir = Path(__file__).resolve().parent / "models" / "kokoro"
        logger.info("Using local Kokoro-82M TTS (100% Free & Local ONNX)...")
        tts = KokoroTTSService(
            model_path=str(model_dir / "kokoro-v1.0.onnx"),
            voices_path=str(model_dir / "voices-v1.0.bin"),
            settings=KokoroTTSService.Settings(
                voice="af_heart",
                language=Language.EN,
                speed=1.0,
            ),
        )
    elif is_valid_key(CARTESIA_API_KEY):
        from pipecat.services.cartesia.tts import CartesiaTTSService
        logger.info("Using Cartesia TTS...")
        tts = CartesiaTTSService(
            api_key=CARTESIA_API_KEY,
            sample_rate=8000,
            settings=CartesiaTTSService.Settings(
                voice="86e30c1d-714b-4074-a1f2-1cb6b552fb49",  # Natural phone voice
            ),
        )
    elif is_valid_key(ELEVENLABS_API_KEY):
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        logger.info("Using ElevenLabs TTS...")
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            voice_id="21m00Tcm4TlvDq8ikWAM",
        )
    elif is_valid_key(OPENAI_API_KEY):
        from pipecat.services.openai.tts import OpenAITTSService
        logger.info("Using OpenAI TTS...")
        tts = OpenAITTSService(
            api_key=OPENAI_API_KEY,
            voice="alloy",
        )
    else:
        raise ValueError("No valid TTS key configured! Please set CARTESIA_API_KEY, ELEVENLABS_API_KEY, or OPENAI_API_KEY in .env.")

    # 5. Pipeline Assembly with tuned VAD for telephone lines
    from pipecat.audio.vad.vad_analyzer import VADParams
    vad = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.8,
            start_secs=0.35,  # Prevents phone line clicks from triggering false interruptions
            stop_secs=0.2,
            min_volume=0.6,
        )
    )
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            # Phone noise alone must not interrupt the agent; require a word
            # recognized by STT before starting the caller's turn.
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=1)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Caller connected on stream {stream_sid}. Speaking a protected opening greeting...")
        greeting = "Hello, this is Alex, your AI assistant. How can I help you today?"
        # Speak immediately instead of waiting for the LLM. The opening audio
        # must survive caller speech or connection noise.
        greeting_frame = TTSSpeakFrame(greeting)
        greeting_frame.interruptible = False
        await worker.queue_frames([greeting_frame])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Caller disconnected on stream {stream_sid}.")
        await runner.cancel()

    @user_aggregator.event_handler("on_user_mute_stopped")
    async def on_caller_input_enabled(aggregator):
        logger.info(f"Caller input enabled on stream {stream_sid}.")

    try:
        await runner.run()
    finally:
        if recorder:
            recorder.close()
        logger.info(
            f"Pulse audio summary for {stream_sid}: "
            f"prepared={serializer.outbound_packets}, sent={audio_websocket.sent_packets}, "
            f"bytes={audio_websocket.sent_bytes}, generated_bytes={audio_websocket.sent_generated_bytes}, "
            f"queued_bytes={audio_websocket.queued_audio_bytes}, clear={audio_websocket.sent_clears}, "
            f"send_failures={audio_websocket.failed_sends}, wire_rate={wire_sample_rate} Hz."
        )
        if serializer.outbound_packets and not audio_websocket.sent_generated_bytes:
            logger.error("Agent audio was prepared but no generated PCM completed a Pulse WebSocket send.")

# ------------------------------------------------------------------------------
# 4. FastAPI Application & Endpoints
# ------------------------------------------------------------------------------
app = FastAPI(title="Pulse + Pipecat AI Calling Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount recordings folder to serve audio files directly for HTML5 playback
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

class CallRequest(BaseModel):
    name: str
    phone_number: str
    service_inquiry: Optional[str] = "AI Calling Solution"
    audio_test: bool = False
    mode: Optional[str] = "ai"

# In-memory store for active calls metadata
active_calls = {}

@app.get("/api/recordings")
def get_recordings():
    return load_recordings_metadata()

@app.get("/api/stats")
def get_stats():
    records = load_recordings_metadata()
    total_calls = len(records)
    total_duration_sec = sum(r.get("duration", 0) for r in records)
    total_mins = round(total_duration_sec / 60, 1)
    return {
        "total_calls": total_calls,
        "total_duration_seconds": total_duration_sec,
        "total_duration_minutes": total_mins,
        "active_calls": len(active_calls),
    }

@app.get("/api/leads")
def get_leads():
    return load_recordings_metadata()

@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Voice Hub</title>
        <style>
            :root {{
                --primary: #2563eb;
                --primary-hover: #1d4ed8;
                --success: #059669;
                --gray-dark: #374151;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --border: #e2e8f0;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: #1e293b; line-height: 1.5; }}
            .container {{ max-width: 960px; margin: 0 auto; padding: 24px 16px; }}
            
            /* Navbar */
            .navbar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px 20px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .nav-brand {{ display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; color: #0f172a; }}
            .badge-pulse {{ background: #dbeafe; color: #1e40af; font-size: 12px; padding: 3px 8px; border-radius: 9999px; font-weight: 600; }}
            .nav-tabs {{ display: flex; gap: 8px; list-style: none; }}
            .nav-tab-btn {{
                background: transparent;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                font-size: 14px;
                color: #64748b;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .nav-tab-btn:hover {{ background: #f1f5f9; color: #0f172a; }}
            .nav-tab-btn.active {{ background: var(--primary); color: white; }}

            /* Content Sections */
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}

            .card {{
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            h2 {{ font-size: 20px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}

            /* Stats Grid */
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
            .stat-card {{
                background: white;
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 16px 20px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }}
            .stat-title {{ font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
            .stat-value {{ font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 4px; }}
            .stat-sub {{ font-size: 12px; color: var(--success); margin-top: 4px; font-weight: 500; }}

            /* Form & Inputs */
            .form-group {{ margin-bottom: 16px; }}
            label {{ display: block; font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px; }}
            input[type="text"] {{
                width: 100%;
                padding: 10px 14px;
                font-size: 15px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: #f8fafc;
                outline: none;
                transition: border 0.2s;
            }}
            input[type="text"]:focus {{ border-color: var(--primary); background: white; }}

            /* Buttons */
            .btn {{
                width: 100%;
                padding: 12px 16px;
                font-size: 15px;
                font-weight: 600;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                transition: filter 0.15s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin-bottom: 10px;
            }}
            .btn:hover {{ filter: brightness(0.92); }}
            .btn-primary {{ background: var(--primary); color: white; }}
            .btn-success {{ background: var(--success); color: white; }}
            .btn-secondary {{ background: #475569; color: white; }}
            .btn-sm {{ width: auto; padding: 6px 12px; font-size: 13px; margin: 0; text-decoration: none; border-radius: 6px; display: inline-block; }}
            .btn-refresh {{ background: #f1f5f9; color: #334155; border: 1px solid var(--border); font-size: 13px; padding: 6px 14px; border-radius: 6px; cursor: pointer; }}
            .btn-refresh:hover {{ background: #e2e8f0; }}

            /* Table */
            .table-container {{ overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ background: #f8fafc; padding: 10px 14px; font-weight: 600; color: #475569; border-bottom: 1px solid var(--border); }}
            td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
            tr:hover {{ background: #f8fafc; }}

            /* Audio Player */
            audio {{ height: 36px; outline: none; width: 220px; }}

            /* Empty state */
            .empty-state {{ text-align: center; padding: 40px 20px; color: #64748b; }}
            .empty-state-icon {{ font-size: 40px; margin-bottom: 12px; }}

            #status-box {{
                margin-top: 15px;
                padding: 12px 16px;
                border-radius: 8px;
                font-weight: 600;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Top Navbar -->
            <nav class="navbar">
                <div class="nav-brand">
                    <span>📞 AI Voice Hub</span>
                    <span class="badge-pulse">Pulse: {PULSE_VIRTUAL_NUMBER}</span>
                </div>
                <div class="nav-tabs">
                    <button class="nav-tab-btn active" onclick="switchTab('dashboard')">📊 Dashboard</button>
                    <button class="nav-tab-btn" onclick="switchTab('dialer')">📞 Call Option</button>
                    <button class="nav-tab-btn" onclick="switchTab('recordings')">🎙️ Recordings</button>
                </div>
            </nav>

            <!-- 1. OVERALL DASHBOARD TAB -->
            <div id="tab-dashboard" class="tab-content active">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-title">Total Calls</div>
                        <div class="stat-value" id="stat-total-calls">0</div>
                        <div class="stat-sub">Across all sessions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Total Talk Time</div>
                        <div class="stat-value" id="stat-total-time">0 min</div>
                        <div class="stat-sub">Recorded audio</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Active AI Models</div>
                        <div class="stat-value" style="font-size: 18px; margin-top: 10px;">Kokoro + Moonshine</div>
                        <div class="stat-sub">100% Free Local AI Stack</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Gateway Status</div>
                        <div class="stat-value" style="font-size: 18px; color: var(--success); margin-top: 10px;">🟢 Online</div>
                        <div class="stat-sub">Pulse Virtual {PULSE_VIRTUAL_NUMBER}</div>
                    </div>
                </div>

                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <h2>Recent Call History</h2>
                        <button class="btn btn-primary btn-sm" onclick="switchTab('dialer')">➕ Place Call</button>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Customer</th>
                                    <th>Phone</th>
                                    <th>Time</th>
                                    <th>Duration</th>
                                    <th>Recording</th>
                                </tr>
                            </thead>
                            <tbody id="recent-calls-body">
                                <tr><td colspan="5" style="text-align: center; color: #64748b;">Loading calls...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- 2. CALL OPTION (DIALER) TAB -->
            <div id="tab-dialer" class="tab-content">
                <div class="card" style="max-width: 600px; margin: 0 auto;">
                    <h2>📞 Place Outbound AI Call</h2>
                    <p style="color: #64748b; font-size: 14px; margin-bottom: 20px;">
                        Dial any mobile number directly through ConnectHub Pulse with AI assistant Alex.
                    </p>

                    <div class="form-group">
                        <label for="name">Customer Name:</label>
                        <input type="text" id="name" placeholder="John Doe" value="Test Customer">
                    </div>

                    <div class="form-group">
                        <label for="phone">Customer Phone Number (10 digits):</label>
                        <input type="text" id="phone" placeholder="9876543210" value="">
                    </div>

                    <button class="btn btn-primary" onclick="triggerCall('ai')">
                        🚀 Call Customer Now (AI Agent Alex)
                    </button>
                    <button class="btn btn-success" onclick="triggerCall('tone')">
                        🔔 Test Beep Tone (Server Sound Test)
                    </button>
                    <button class="btn btn-secondary" onclick="triggerCall('echo')">
                        🔁 Test Phone Audio (Echo Test)
                    </button>

                    <div id="status-box"></div>

                    <div style="margin-top: 20px; font-size: 13px; color: #64748b; border-top: 1px solid var(--border); padding-top: 15px;">
                        • <b>AI Agent Alex:</b> Live bidirectional conversation. Both sides are recorded automatically.<br>
                        • <b>Beep Tone:</b> Plays a clear 3-second tone to confirm carrier speaker audio.<br>
                        • <b>Echo Test:</b> Echoes back whatever you speak into the phone.
                    </div>
                </div>
            </div>

            <!-- 3. RECORDINGS TAB -->
            <div id="tab-recordings" class="tab-content">
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <h2>🎙️ Call Audio Recordings</h2>
                        <button class="btn-refresh" onclick="fetchRecordings()">🔄 Refresh List</button>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Customer</th>
                                    <th>Phone</th>
                                    <th>Date & Time</th>
                                    <th>Duration</th>
                                    <th>Listen Recording</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody id="all-recordings-body">
                                <tr><td colspan="6" style="text-align: center; color: #64748b;">Loading recordings...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Tab Switcher
            function switchTab(tabId) {{
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.nav-tab-btn').forEach(btn => btn.classList.remove('active'));
                
                document.getElementById('tab-' + tabId).classList.add('active');
                event.target.classList.add('active');

                if (tabId === 'dashboard' || tabId === 'recordings') {{
                    fetchStats();
                    fetchRecordings();
                }}
            }}

            // Fetch Stats for Dashboard
            async function fetchStats() {{
                try {{
                    const res = await fetch('/api/stats');
                    const data = await res.json();
                    document.getElementById('stat-total-calls').innerText = data.total_calls || 0;
                    document.getElementById('stat-total-time').innerText = (data.total_duration_minutes || 0) + ' min';
                }} catch (e) {{
                    console.error('Failed to load stats', e);
                }}
            }}

            // Fetch Recordings for Tables
            async function fetchRecordings() {{
                try {{
                    const res = await fetch('/api/recordings');
                    const recordings = await res.json();

                    // 1. Populate Dashboard Recent Calls (up to 5)
                    const recentTbody = document.getElementById('recent-calls-body');
                    if (recordings.length === 0) {{
                        recentTbody.innerHTML = '<tr><td colspan="5" class="empty-state">No calls recorded yet.</td></tr>';
                    }} else {{
                        recentTbody.innerHTML = recordings.slice(0, 5).map(r => `
                            <tr>
                                <td><b>${{r.customer_name || 'Customer'}}</b></td>
                                <td>${{r.phone_number || '-'}}</td>
                                <td>${{r.timestamp}}</td>
                                <td>${{r.duration}}s</td>
                                <td>
                                    <audio controls preload="none" src="${{r.url}}"></audio>
                                </td>
                            </tr>
                        `).join('');
                    }}

                    // 2. Populate Full Recordings Tab
                    const allTbody = document.getElementById('all-recordings-body');
                    if (recordings.length === 0) {{
                        allTbody.innerHTML = '<tr><td colspan="6" class="empty-state"><div class="empty-state-icon">🎙️</div>No recordings found yet.<br>Place a call from the "Call Option" tab to record your first conversation!</td></tr>';
                    }} else {{
                        allTbody.innerHTML = recordings.map(r => `
                            <tr>
                                <td><b>${{r.customer_name || 'Customer'}}</b></td>
                                <td>${{r.phone_number || '-'}}</td>
                                <td>${{r.timestamp}}</td>
                                <td>${{r.duration}}s (${{r.file_size_kb}} KB)</td>
                                <td>
                                    <audio controls preload="none" src="${{r.url}}"></audio>
                                </td>
                                <td>
                                    <a href="${{r.url}}" download="${{r.filename}}" class="btn btn-secondary btn-sm">⬇ Download</a>
                                </td>
                            </tr>
                        `).join('');
                    }}
                }} catch (e) {{
                    console.error('Failed to load recordings', e);
                }}
            }}

            // Trigger Outbound Phone Call
            async function triggerCall(mode = 'ai') {{
                const name = document.getElementById('name').value.trim();
                const phone = document.getElementById('phone').value.trim();
                const statusBox = document.getElementById('status-box');

                if (!phone || phone.length < 10) {{
                    statusBox.style.display = 'block';
                    statusBox.style.background = '#fef2f2';
                    statusBox.style.color = '#dc2626';
                    statusBox.innerText = 'Please enter a valid 10-digit phone number.';
                    return;
                }}

                statusBox.style.display = 'block';
                statusBox.style.background = '#eff6ff';
                statusBox.style.color = '#2563eb';
                statusBox.innerText = 'Dialing ' + phone + ' via Pulse (' + mode.toUpperCase() + ' mode)...';

                try {{
                    const res = await fetch('/api/call', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ name: name, phone_number: phone, mode: mode }})
                    }});
                    const data = await res.json();
                    if (data.success) {{
                        statusBox.style.background = '#f0fdf4';
                        statusBox.style.color = '#16a34a';
                        if (mode === 'tone') {{
                            statusBox.innerText = 'Beep test ringing. Answer to hear the 3-second tone.';
                        }} else if (mode === 'echo') {{
                            statusBox.innerText = 'Echo test ringing. Speak after answering to hear echo.';
                        }} else {{
                            statusBox.innerText = 'Call Placed! Ringing phone for AI conversation (Recording active)...';
                        }}
                        // Refresh recordings after short delay
                        setTimeout(fetchRecordings, 8000);
                    }} else {{
                        statusBox.style.background = '#fef2f2';
                        statusBox.style.color = '#dc2626';
                        statusBox.innerText = 'Call failed: ' + (data.error || 'Check server logs');
                    }}
                }} catch (e) {{
                    statusBox.style.background = '#fef2f2';
                    statusBox.style.color = '#dc2626';
                    statusBox.innerText = 'Network error: ' + e.message;
                }}
            }}

            // Auto-load on page start
            window.addEventListener('DOMContentLoaded', () => {{
                fetchStats();
                fetchRecordings();
            }});
        </script>
    </body>
    </html>
    """

@app.post("/api/call")
def initiate_call(req: CallRequest):
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/pulse-stream"
    if req.mode in ("echo", "tone"):
        ws_url += f"?mode={req.mode}"
    elif req.audio_test:
        ws_url += "?mode=echo"
    
    res = pulse_service.initiate_click_to_call(
        customer_phone=req.phone_number,
        ws_url=ws_url,
        custom_params={
            "customer_name": req.name,
            "customer_phone": req.phone_number,
            "phone_number": req.phone_number,
            "service_inquiry": req.service_inquiry,
        },
    )

    if not res.get("success"):
        return JSONResponse(status_code=400, content={"success": False, "error": res.get("error")})

    call_id = res.get("unique_call_identifier")
    active_calls[call_id] = {
        "name": req.name,
        "phone_number": req.phone_number,
        "service_inquiry": req.service_inquiry,
    }

    return {"success": True, "call_id": call_id, "message": f"Dialing {req.name}..."}

async def run_pulse_tone_test(websocket: WebSocket, stream_sid: str, *, wire_sample_rate: int = 8000):
    """Send 3 seconds of a clear 440Hz beep tone to confirm Pulse telephone playback."""
    import math, struct
    logger.info(f"Starting Pulse phone tone test on stream {stream_sid}...")
    sample_rate = wire_sample_rate
    total_samples = sample_rate * 3
    samples = [int(16000 * math.sin(2 * math.pi * 440 * n / sample_rate)) for n in range(total_samples)]
    pcm_bytes = struct.pack(f"<{len(samples)}h", *samples)
    serializer = PulseFrameSerializer(stream_sid, sample_rate)
    audio_websocket = PulseAudioWebSocket(websocket, sample_rate, use_echo_envelope=True)
    try:
        await audio_websocket.send_text(await serializer.serialize(OutputAudioRawFrame(
            audio=pcm_bytes, sample_rate=sample_rate, num_channels=1,
        )))
        while audio_websocket.queued_audio_bytes:
            message = json.loads(await audio_websocket.receive_text())
            if message.get("event") in ("stop", "ended", "error"):
                break
        # Let the final telephone packet play before closing the test call.
        if not audio_websocket.queued_audio_bytes:
            await asyncio.sleep(0.1)
        logger.info(
            f"Pulse tone summary for {stream_sid}: sent={audio_websocket.sent_packets}, "
            f"generated_bytes={audio_websocket.sent_generated_bytes}, "
            f"queued_bytes={audio_websocket.queued_audio_bytes}."
        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"Error in Pulse tone test: {exc}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

async def run_pulse_echo_test(websocket: WebSocket, stream_sid: str, *, wire_sample_rate: int = 8000):
    """Return Pulse's media JSON unchanged to isolate telephone playback, with deep forensics."""
    sent_packets = 0
    sent_bytes = 0
    logger.info(f"Pulse phone echo test started on stream {stream_sid}.")
    collected_pcm = bytearray()
    try:
        while True:
            message_text = await websocket.receive_text()
            message = json.loads(message_text)
            if not isinstance(message, dict):
                continue
            event = message.get("event")
            if event == "media":
                media = message.get("media") or {}
                payload = media.get("payload") if isinstance(media, dict) else None
                if not payload:
                    continue
                try:
                    pcm = base64.b64decode(payload, validate=True)
                except (ValueError, TypeError):
                    logger.warning("Ignoring invalid audio in the Pulse echo test.")
                    continue
                
                collected_pcm.extend(pcm)

                if sent_packets == 0:
                    import struct
                    samples_le = struct.unpack(f"<{len(pcm)//2}h", pcm)
                    samples_be = struct.unpack(f">{len(pcm)//2}h", pcm)
                    logger.info(f"=== PULSE FORENSIC INSPECTION ===")
                    logger.info(f"RAW JSON FROM PULSE: {message_text}")
                    logger.info(f"MESSAGE TOP-LEVEL KEYS: {list(message.keys())}")
                    logger.info(f"MEDIA KEYS: {list(media.keys())}")
                    logger.info(f"TIMESTAMP TYPE: {type(media.get('timestamp')).__name__}, VALUE: {repr(media.get('timestamp'))}")
                    logger.info(f"CHUNK TYPE: {type(media.get('chunk')).__name__}, VALUE: {repr(media.get('chunk'))}")
                    logger.info(f"SEQUENCE TYPE: {type(message.get('sequence_number')).__name__}, VALUE: {repr(message.get('sequence_number'))}")
                    logger.info(f"PCM BYTES: {len(pcm)} bytes ({len(pcm)//2} samples)")
                    logger.info(f"FIRST 8 SAMPLES (Little-Endian <h): {samples_le[:8]}")
                    logger.info(f"FIRST 8 SAMPLES (Big-Endian >h): {samples_be[:8]}")
                    logger.info(f"=================================")

                # The integration guide uses a WebSocket echo endpoint. Keep
                # the complete original JSON and audio bytes for this test.
                await websocket.send_text(message_text)
                sent_packets += 1
                sent_bytes += len(pcm)
            elif event in ("stop", "ended"):
                break
            elif event == "error":
                logger.error(f"Pulse echo test error: {message.get('error', message.get('message'))}")
    except WebSocketDisconnect:
        pass
    finally:
        logger.info(
            f"Pulse echo summary for {stream_sid}: "
            f"sent={sent_packets}, bytes={sent_bytes}."
        )
        if collected_pcm:
            try:
                import wave
                with wave.open("echo_recorded.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(wire_sample_rate)
                    wf.writeframes(collected_pcm)
                logger.info(f"Saved {len(collected_pcm)} bytes of echo audio to echo_recorded.wav")
            except Exception as e:
                logger.error(f"Failed to save echo_recorded.wav: {e}")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.websocket("/pulse-stream")
async def pulse_audio_stream(websocket: WebSocket):
    """WebSocket connection that Pulse connects to during the call."""
    await websocket.accept()
    logger.info("Pulse WebSocket connected. Waiting for 'start' handshake...")

    stream_sid = None
    wire_sample_rate = 8000
    customer_info = {"name": "Customer"}
    call_mode = websocket.query_params.get("mode", "ai")

    try:
        # 1. Read initial handshake frames until we get 'start'
        while not stream_sid:
            msg_text = await websocket.receive_text()
            data = json.loads(msg_text)
            event = data.get("event")
            # The working echo returns these control frames as well as media.
            if call_mode == "echo" or event in ("connected", "start"):
                await websocket.send_text(msg_text)

            if event == "connected":
                logger.info("Pulse audio protocol connected.")
            elif event == "start":
                start = data.get("start") or {}
                stream_sid = data.get("stream_sid") or start.get("stream_sid")
                if not isinstance(stream_sid, str) or not stream_sid.strip():
                    raise ValueError("Pulse start handshake is missing stream_sid.")
                wire_sample_rate = get_pulse_sample_rate(start)
                custom_params = start.get("custom_parameters") or {}
                if custom_params.get("customer_name"):
                    customer_info["name"] = custom_params["customer_name"]
                if custom_params.get("customer_phone"):
                    customer_info["phone"] = custom_params["customer_phone"]
                elif custom_params.get("phone_number"):
                    customer_info["phone"] = custom_params["phone_number"]

                if customer_info.get("phone") in (None, "Unknown") and active_calls:
                    latest = list(active_calls.values())[-1]
                    customer_info["phone"] = latest.get("phone_number", "Unknown")
                    if customer_info.get("name") in (None, "Customer"):
                        customer_info["name"] = latest.get("name", "Customer")
                logger.info(
                    f"Handshake complete! Stream SID: {stream_sid}, "
                    f"PCM16 mono at {wire_sample_rate} Hz, Mode: {call_mode}; "
                    f"Media Format: {start.get('media_format', start.get('mediaFormat', {}))}"
                )
                if not (start.get("media_format") or start.get("mediaFormat")):
                    logger.warning("Pulse did not advertise a media format; using the existing 8000 Hz PCM16 default.")
                break

        # 2. Hand over the socket to the requested call mode.
        if call_mode == "echo":
            await run_pulse_echo_test(websocket, stream_sid, wire_sample_rate=wire_sample_rate)
        elif call_mode == "tone":
            await run_pulse_tone_test(websocket, stream_sid, wire_sample_rate=wire_sample_rate)
        else:
            await run_pipecat_pipeline(
                websocket, stream_sid, customer_info, wire_sample_rate=wire_sample_rate
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket closed for stream: {stream_sid}")
    except Exception as e:
        logger.error(f"Error handling Pulse stream: {e}")

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    module_name = Path(__file__).stem
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=PORT, reload=True)
