import streamlit as st
import av
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

st.title("🎤 ASR Test — Google Speech-to-Text")


# ============================================================
# Google Client (Uses your working TOML secrets)
# ============================================================

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()


# ============================================================
# Audio Processor — Captures ALL audio frames
# ============================================================

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio_frame(self, frame: av.AudioFrame):
        self.frames.append(frame)
        return frame


# ============================================================
# WebRTC
# ============================================================

webrtc_ctx = webrtc_streamer(
    key="asr-test",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("🎧 **Recording… speak now, then press STOP above.**")
else:
    st.write("Press **START** to begin recording.")


# ============================================================
# Transcription
# ============================================================

def transcribe_bytes(data: bytes) -> str:
    audio = speech.RecognitionAudio(content=data)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="de-DE",
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        return ""


if st.button("Transcribe"):
    if not webrtc_ctx or not webrtc_ctx.audio_processor:
        st.error("❌ No audio processor available.")
        st.stop()

    frames = webrtc_ctx.audio_processor.frames
    if not frames:
        st.error("❌ No audio captured.")
        st.stop()

    # Convert frames → raw PCM16 bytes
    pcm_bytes = b"".join(
        f.to_ndarray().astype(np.int16).tobytes()
        for f in frames
    )

    st.write(f"Captured audio length: {len(pcm_bytes)} bytes")

    if len(pcm_bytes) < 5000:
        st.warning("⚠️ Very short audio — try speaking for 1–2 seconds.")
        st.stop()

    st.write("⏳ Transcribing…")
    text = transcribe_bytes(pcm_bytes)

    st.success(f"**Transcript:** {text}")
