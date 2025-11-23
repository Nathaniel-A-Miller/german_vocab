import streamlit as st
import numpy as np
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
import tempfile
import soundfile as sf

st.title("🎤 HTML5 Microphone Test (No WebRTC)")


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

    # Save to temp WAV for reading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Read audio using soundfile
    data, samplerate = sf.read(tmp_path)

    # Convert to 16-bit PCM
    pcm_data = (data * 32767).astype(np.int16).tobytes()

    st.write(f"Sample rate: {samplerate}")
    st.write(f"PCM length: {len(pcm_data)} bytes")

    # ========= Google ASR =========

    st.write("⏳ Transcribing…")

    audio = speech.RecognitionAudio(content=pcm_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=samplerate,
        language_code="de-DE",
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        st.success(f"**Transcript:** {transcript}")
    else:
        st.error("No transcription returned.")
