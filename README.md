# 📞 AI Phone Calling Agent

A production-ready, low-latency, bidirectional AI Phone Calling Agent built on **Pipecat** and integrated with **ConnectHub Pulse** telephony, featuring a **100% Free / Open-Source local AI stack**.

---

## 🌟 Key Features

* **⚡ Ultra-Low Latency Telephony:** Bidirectional real-time audio over WebSockets with ConnectHub Pulse carrier gateway (8000 Hz, 16-bit PCM mono).
* **🗣️ Full Barge-In (Interruption Handling):** Powered by Silero VAD + STT word-level trigger. If the customer interrupts the AI mid-sentence, the agent silences instantly.
* **🎙️ Automatic Call Recording:** Every conversation is recorded in high-fidelity WAV format and displayed in the web dashboard with an embedded HTML5 audio player.
* **🌐 Modern 3-Tab Web Dashboard:**
  * **📊 Overall Dashboard:** Real-time KPI metrics (total calls, talk time, engine health).
  * **📞 Call Option:** Web dialer for outbound calls with AI Agent Alex, beep tone test, and echo test.
  * **🎙️ Recordings:** Call history table with inline audio player and one-click WAV download.
* **💰 100% Free Voice Stack Supported:**
  * **STT (Ear):** Groq Whisper (`whisper-large-v3-turbo`, ~180ms) or local Moonshine ONNX.
  * **LLM (Brain):** Google Gemini 3.1 Flash-Lite (low latency thinking mode).
  * **TTS (Voice):** Kokoro-82M (local ONNX neural voice, <100ms, $0 cost).

---

## 🚀 Quick Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
* **Pulse Telephony:** `PULSE_MEMBER_NAME`, `PULSE_MEMBER_PASSWORD`, `PULSE_VIRTUAL_NUMBER`
* **Public Tunnel:** `PUBLIC_BASE_URL` (e.g. ngrok: `https://xxxx.ngrok-free.dev`)
* **AI API Keys:** `GROQ_API_KEY` or `GEMINI_API_KEY`

---

## 💻 Running the Server

Start the voice calling agent server:
```bash
python AI-calling-Agent.py
```
Open your browser and navigate to:
```text
http://localhost:8080
```

---

## 📁 Project Structure

```text
├── AI-calling-Agent.py       # Main server application & FastAPI endpoints
├── ai_calling_agent.py       # Python standard alias
├── .env.example              # Template for environment variables (safe for Git)
├── .gitignore                # Git ignore rules (prevents secret leaks)
├── models/
│   └── kokoro/               # Kokoro-82M ONNX local voice model
├── recordings/               # Auto-saved call audio recordings (.wav)
└── README.md
```

---

## 🔒 Security Note
* Sensitive credentials, API keys, passwords, and phone numbers are strictly kept inside `.env`.
* `.env` and `.wav` call recordings are ignored by `.gitignore` and are never committed to GitHub.

---

## 📄 License
Licensed under the [BSD 2-Clause License](LICENSE).
