# 📞 AI Phone Calling Agent

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Pipecat AI](https://img.shields.io/badge/Pipecat-1.16+-orange.svg)](https://github.com/pipecat-ai/pipecat)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17+-blueviolet.svg)](https://onnxruntime.ai/)
[![Telephony](https://img.shields.io/badge/Telephony-ConnectHub%20Pulse-green.svg)](https://connecthub.pulsework360.com/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](LICENSE)

An enterprise-ready, low-latency, full-duplex conversational AI voice calling system. Built on **Pipecat** and integrated directly with **ConnectHub Pulse** carrier telephony, the system supports both high-throughput cloud AI providers and a **100% free / open-source local neural stack** running locally with sub-second response latency.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [AI Engine Provider Matrix](#-ai-engine-provider-matrix)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Quickstart](#-installation--quickstart)
- [Local Model Setup](#-local-model-setup)
- [Configuration Reference](#-configuration-reference)
- [Web Operations Console](#-web-operations-console)
- [API & WebSocket Specification](#-api--websocket-specification)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Production Deployment & Security](#-production-deployment--security)
- [License](#-license)

---

## 🏗️ System Architecture

The agent connects telecommunication carrier infrastructure with real-time neural inference pipelines over high-speed bidirectional WebSockets.

```
                              ┌──────────────────────────────────────────────┐
                              │     ConnectHub Pulse Carrier Gateway        │
                              └──────────────────────┬───────────────────────┘
                                                     │ 
                                                     ▼ Bidirectional WebSocket (PCM16, 8kHz Mono)
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Application Core (Port 8080)                                                                   │
│                                                                                                        │
│  ┌─────────────────────────────┐        ┌───────────────────────────────────────────────────────────┐  │
│  │    PulseFrameSerializer     │        │                   Call Recording Pipeline                 │  │
│  │  (Timestamp Clock & Jitter) │◄───────┤ High-fidelity WAV Capture + Metadata Persistence          │  │
│  └──────────────┬──────────────┘        └───────────────────────────────────────────────────────────┘  │
│                 │ Audio Frames                                                                         │
│                 ▼                                                                                      │
│  ┌─────────────────────────────┐        ┌───────────────────────────────────────────────────────────┐  │
│  │   Silero VAD / Interruption │───────►│ Barge-In Engine: Sends ClearFrame on Caller Speech        │  │
│  └──────────────┬──────────────┘        └───────────────────────────────────────────────────────────┘  │
│                 │ Speech Segments                                                                      │
│                 ▼                                                                                      │
│  ┌─────────────────────────────┐        ┌───────────────────────────────────────────────────────────┐  │
│  │  Speech-to-Text (Ear)       │        │ • Groq Whisper-large-v3-turbo (~180ms TTFB)               │  │
│  │                             │───────►│ • Useful Sensors Moonshine (Local ONNX, $0 cost)         │  │
│  │                             │        │ • Deepgram Nova-2 Streaming                               │  │
│  └──────────────┬──────────────┘        └───────────────────────────────────────────────────────────┘  │
│                 │ Transcribed Tokens                                                                   │
│                 ▼                                                                                      │
│  ┌─────────────────────────────┐        ┌───────────────────────────────────────────────────────────┐  │
│  │  LLM Conversational Brain   │        │ • Google Gemini 3.1 Flash-Lite (Low Latency Thinking)    │  │
│  │                             │───────►│ • OpenAI GPT-4o-mini                                      │  │
│  │                             │        │ • Groq LLaMA 3.3 70B Versatile                            │  │
│  └──────────────┬──────────────┘        └───────────────────────────────────────────────────────────┘  │
│                 │ Synthesized Response Stream                                                          │
│                 ▼                                                                                      │
│  ┌─────────────────────────────┐        ┌───────────────────────────────────────────────────────────┐  │
│  │  Text-to-Speech (Voice)     │        │ • Kokoro-82M ONNX (Local Neural Voice, <100ms, $0 cost)   │  │
│  │                             │───────►│ • Cartesia Sonic Telephony (~100ms Cloud TTS)             │  │
│  │                             │        │ • ElevenLabs Conversational Voice                         │  │
│  └──────────────┬──────────────┘        └───────────────────────────────────────────────────────────┘  │
│                 │ Audio Resampler (24kHz -> 8kHz PCM16 Linear)                                         │
│                 ▼                                                                                      │
│  ┌─────────────────────────────┐                                                                       │
│  │  PulseAudioWebSocket Client │────────► Outbound Audio Stream to Telephony Gateway                   │
│  └─────────────────────────────┘                                                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

* **⚡ Sub-Second Telephony Turnaround**: Bidirectional 8000 Hz, 16-bit Linear PCM audio transmission engineered for minimal jitter and carrier synchronization.
* **🗣️ Full Barge-In (Interruption Handling)**: Immediate cancellation of queued voice synthesis when caller speech is detected by Silero VAD, providing a natural conversational experience.
* **💰 Flexible Multi-Tier AI Stack**:
  * **Free / Local Tier**: Moonshine ONNX STT + Gemini 3.1 Flash-Lite + Kokoro-82M ONNX TTS ($0 infrastructure voice cost).
  * **Enterprise Cloud Tier**: Deepgram Streaming STT + OpenAI GPT-4o + Cartesia Sonic TTS.
* **🎙️ Automated Call Recording**: Real-time continuous recording of every conversation stored as standard 16-bit PCM `.wav` with full metadata persistence (`recordings.json`).
* **🌐 Web Operations Console**: 3-tab responsive web interface:
  * **Analytics Dashboard**: Call volume, talk duration, system telemetry, and engine status.
  * **Outbound Call Option**: Direct browser-based click-to-call dialer with built-in loopback echo and tone diagnostic tools.
  * **Recordings Archive**: Searchable call table with inline HTML5 audio playback and one-click WAV download.
* **🛡️ Security First**: Zero-leak configuration with strict separation of credentials in ignored environment variables.

---

## 🧠 AI Engine Provider Matrix

| Pipeline Component | Local / Free Option | Cloud Enterprise Option | Latency Target |
| :--- | :--- | :--- | :--- |
| **STT (Ear)** | Useful Sensors Moonshine ONNX | Groq Whisper / Deepgram Nova-2 | ~150ms – 250ms |
| **LLM (Brain)** | Google Gemini 3.1 Flash-Lite | OpenAI GPT-4o-mini / Groq LLaMA 3.3 70B | ~200ms – 400ms |
| **TTS (Voice)** | Kokoro-82M ONNX (`af_heart`) | Cartesia Sonic / ElevenLabs | ~80ms – 150ms |
| **VAD** | Silero VAD v5 | Silero VAD v5 | <30ms |

---

## 📁 Project Structure

```text
.
├── AI-calling-Agent.py       # Primary server application & pipeline orchestration
├── ai_calling_agent.py       # Python standard naming alias
├── pulse_pipecat_server.py   # Test-compatible server module
├── test_pulse_audio.py       # Comprehensive offline unit & regression test suite
├── requirements.txt          # Production dependency specifications
├── .env.example              # Sanitized configuration template (safe for Git)
├── .gitignore                # Enterprise-grade exclusion rules (secrets, audio, models)
├── README.md                 # System documentation & technical guide
├── models/
│   └── kokoro/
│       ├── .gitkeep          # Preserves directory structure in Git
│       ├── kokoro-v1.0.onnx  # [Ignored] Kokoro-82M neural model weights (325 MB)
│       └── voices-v1.0.bin   # [Ignored] Kokoro multi-voice binary embeddings (28 MB)
└── recordings/               # [Ignored] Auto-generated call recordings (.wav) & metadata
```

---

## ⚙️ Prerequisites

1. **Python**: `3.10` or `3.11` (64-bit recommended).
2. **Telephony**: A valid **ConnectHub Pulse** account (Account Code, Member Name, Password, and Virtual Number).
3. **Public Gateway**: A public URL pointing to port `8080` (e.g. [ngrok](https://ngrok.com/), Cloudflare Tunnel, or a public static IP / domain with TLS).

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/sibi45/AI-calling-agent.git
cd AI-calling-agent
```

### 2. Configure Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Local Model Setup

If you intend to use the local neural voice stack (**Kokoro-82M ONNX**), download the model weights into `models/kokoro/`:

```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" -OutFile "models/kokoro/kokoro-v1.0.onnx"
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "models/kokoro/voices-v1.0.bin"
```

```bash
# Linux / macOS
wget -O models/kokoro/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -O models/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

*(Note: These binary files are excluded by `.gitignore` and will never be pushed to your repository).*

---

## ⚙️ Configuration Reference

Create your local `.env` configuration file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your active credentials:

```dotenv
# ==============================================================================
# CONNECTHUB PULSE TELEPHONY CONFIGURATION
# ==============================================================================
PULSE_AUTH_URL=https://connecthub.pulsework360.com/auth/token
PULSE_C2C_URL=https://jxoclvto1l.execute-api.ap-south-1.amazonaws.com/v1/
PULSE_ACCOUNT_CODE=your_pulse_account_code_here
PULSE_MEMBER_NAME=your_pulse_member_name_here
PULSE_MEMBER_PASSWORD=your_pulse_member_password_here
PULSE_VIRTUAL_NUMBER=your_pulse_virtual_number_here

# Public Tunnel URL for Pulse WebSocket callbacks
PUBLIC_BASE_URL=https://your-domain-or-ngrok.ngrok-free.dev
PORT=8080

# ==============================================================================
# AI ENGINE CONFIGURATIONS
# ==============================================================================

# --- Speech-to-Text (STT) ---
GROQ_API_KEY=your_groq_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
USE_LOCAL_STT=false

# --- Large Language Model (LLM) ---
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_MAX_OUTPUT_TOKENS=128
OPENAI_API_KEY=your_openai_api_key_here

# --- Text-to-Speech (TTS) ---
USE_KOKORO=true
CARTESIA_API_KEY=your_cartesia_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### Configuration Parameters

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PULSE_AUTH_URL` | String | ConnectHub URL | Telephony token authentication endpoint |
| `PULSE_C2C_URL` | String | AWS API Gateway | Click-to-call outbound gateway endpoint |
| `PULSE_ACCOUNT_CODE` | String | - | Pulse Account identifier |
| `PULSE_MEMBER_NAME` | String | - | Pulse Agent member identifier |
| `PULSE_MEMBER_PASSWORD` | String | - | Pulse Agent member secret |
| `PULSE_VIRTUAL_NUMBER` | String | - | Telephony virtual caller ID number |
| `PUBLIC_BASE_URL` | String | - | Public URL (HTTPS) accessible by carrier |
| `PORT` | Integer | `8080` | Local server listening port |
| `USE_LOCAL_STT` | Boolean | `false` | When `true`, forces offline Moonshine STT |
| `GROQ_API_KEY` | String | - | Groq Whisper / LLaMA API credential |
| `DEEPGRAM_API_KEY` | String | - | Deepgram streaming STT credential |
| `GEMINI_API_KEY` | String | - | Google Gemini LLM API credential |
| `GEMINI_MODEL` | String | `gemini-3.1-flash-lite` | Gemini model variant |
| `USE_KOKORO` | Boolean | `true` | When `true`, activates offline Kokoro TTS |
| `CARTESIA_API_KEY` | String | - | Cartesia Sonic cloud TTS credential |

---

## 💻 Running the Server

### Development Mode
```bash
python AI-calling-Agent.py
```

### Production Mode (Uvicorn Direct)
```bash
uvicorn AI-calling-Agent:app --host 0.0.0.0 --port 8080 --workers 1 --loop uvloop
```

Once started, access the Web Operations Console at:
```text
http://localhost:8080
```

---

## 🖥️ Web Operations Console

The server includes a built-in enterprise single-page application:

1. **📊 Overview Dashboard**:
   - Total Calls, Completed Calls, and Talk Time duration metrics.
   - Real-time engine health indicators (STT, LLM, TTS status).
   - Recent calls summary table with status badges.
2. **📞 Call Option (Outbound Dialer)**:
   - Enter destination phone number with customer name and custom inquiry notes.
   - Choose operation mode:
     - **🤖 AI Agent Mode**: Connects live conversational AI agent Alex.
     - **🔊 Beep Tone Test**: Plays carrier audio calibration tone.
     - **🔁 Audio Echo Test**: Full-duplex echo loopback to test network latency and round-trip quality.
3. **🎙️ Recordings Manager**:
   - Historical call catalog with timestamp, duration, caller ID, and call unique ID.
   - Inline HTML5 wave player for instant review.
   - Direct `.wav` download button.

---

## 📡 API & WebSocket Specification

### 1. Initiate Outbound Call
`POST /api/call`

Triggers ConnectHub Pulse Click-to-Call gateway to dial a customer and connect audio to the WebSocket stream.

**Request Body:**
```json
{
  "name": "Jane Doe",
  "phone_number": "+1234567890",
  "service_inquiry": "Product Demo",
  "mode": "ai",
  "audio_test": false
}
```

**Response (Success):**
```json
{
  "success": true,
  "unique_call_identifier": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "data": { "message": "Call initiated successfully" }
}
```

### 2. Retrieve Call Recordings
`GET /api/recordings`

Returns metadata for all persisted call recordings.

**Response:**
```json
[
  {
    "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "filename": "3fa85f64-5717-4562-b3fc-2c963f66afa6.wav",
    "url": "/recordings/3fa85f64-5717-4562-b3fc-2c963f66afa6.wav",
    "timestamp": "2026-11-24 17:30:00",
    "duration_seconds": 45
  }
]
```

### 3. System Metrics
`GET /api/stats`

Returns aggregated call statistics.

### 4. Telephony Audio Stream
`WebSocket /pulse-stream`

Full-duplex WebSocket stream adhering to ConnectHub Pulse protocol specifications:
- **Audio Format**: `audio/x-l16`, 8000 Hz sample rate, 16-bit Linear PCM, Mono.
- **Handshake Events**: `start`, `media`, `clear`, `stop`.

---

## 🧪 Testing & Quality Assurance

The repository includes a regression test suite covering serialization, timing jitter, clock monotonic consistency, audio resampling, and protocol envelopes:

```powershell
# Run all unit tests
.\venv\Scripts\python.exe -m unittest test_pulse_audio.py
```

```text
Ran 19 tests in 0.779s

OK
```

### Core Regressions Tested
* Timestamp linearity during speech pauses and interruptions.
* 24 kHz to 8 kHz audio pitch retention and RMS power maintenance.
* Complete envelope discard upon barge-in `ClearFrame`.
* Protocol packet integrity against incomplete PCM chunks and invalid base64 payloads.

---

## 🔒 Production Deployment & Security

1. **Credential Isolation**:
   - Environment variables must never be checked into Git. The repository includes strict `.gitignore` rules covering `.env`, `.env.*`, and `*.env`.
2. **Audio Privacy (GDPR / HIPAA Compliance)**:
   - Call recordings contain customer PII. The `recordings/` directory is excluded from version control by default. In production environments, configure scheduled archival to encrypted object storage (e.g. AWS S3 with KMS).
3. **Network Security**:
   - Place the service behind an enterprise reverse proxy (e.g. NGINX, Caddy, or AWS ALB) with HTTPS and WSS (WebSocket Secure) termination.
   - Configure firewall rules to restrict WebSocket ingress to ConnectHub carrier gateway IP ranges.

---

## 📄 License

This project is licensed under the [BSD 2-Clause License](LICENSE).
