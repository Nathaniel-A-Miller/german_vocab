import streamlit as st
import numpy as np
import wave
import tempfile
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

st.title("🎤 HTML Microphone ASR Test (No WebRTC)")


# ============================================================
# Google Speech Client
# ============================================================

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()


# ============================================================
# Record audio
# ============================================================

uploaded = st.audio_input("Press to record")

if uploaded is not None:
    st.success("Audio recorded!")

    # ------------------------------------------------------------
    # Save recording to temp file
    # ------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name   # <--- THIS DEFINES tmp_path
    st.write(f"Temp WAV saved to: {tmp_path}")

    # ------------------------------------------------------------
    # Read WAV metadata WITHOUT discarding header
    # ------------------------------------------------------------
    with wave.open(tmp_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        pcm_bytes = wf.readframes(num_frames)

    st.write(f"Channels: {num_channels}")
    st.write(f"Sample width: {sample_width} bytes")
    st.write(f"Sample rate: {sample_rate}")
    st.write(f"PCM data length: {len(pcm_bytes)}")

    # ------------------------------------------------------------
    # Read the FULL WAV file including header
    # ------------------------------------------------------------
    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    st.write(f"Total WAV bytes sent to Google: {len(wav_bytes)}")

    # ------------------------------------------------------------
    # Google ASR
    # ------------------------------------------------------------
    st.write("⏳ Transcribing…")

    audio = speech.RecognitionAudio(content=wav_bytes)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="de-DE",
        audio_channel_count=num_channels,
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        st.success(f"**Transcript:** {transcript}")
    else:
        st.error("No transcription returned.")
