import streamlit as st
import numpy as np
import wave
import tempfile
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

st.title("🎤 HTML5 Microphone Test (No WebRTC, No Extra Dependencies)")


# ========= Google ASR client =========

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()


# ========= Microphone upload =========

uploaded = st.audio_input("Press to record")

if uploaded is not None:
    st.success("Audio recorded!")

    # save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ========= Read WAV file using Python wave module =========
    with wave.open(tmp_path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        pcm_bytes = wf.readframes(num_frames)

    st.write(f"Channels: {num_channels}")
    st.write(f"Sample width (bytes): {sample_width}")
    st.write(f"Sample rate: {sample_rate}")
    st.write(f"PCM bytes length: {len(pcm_bytes)}")

    # Google expects LINEAR16 → if sample width != 2, we convert

    if sample_width != 2:
        st.error("This WAV file is not 16-bit PCM. Try recording again.")
        st.stop()

    # ========= Send to Google Speech-to-Text =========

    st.write("⏳ Transcribing…")

# Read the full WAV file again (including header)
with open(tmp_path, "rb") as f:
    wav_bytes = f.read()

    audio = speech.RecognitionAudio(content=wav_bytes)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="de-DE",
        audio_channel_count=1,
        enable_automatic_punctuation=False,
    )


    response = client.recognize(config=config, audio=audio)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        st.success(f"**Transcript:** {transcript}")
    else:
        st.error("No transcription returned.")
