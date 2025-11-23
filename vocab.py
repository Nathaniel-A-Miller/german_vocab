import streamlit as st
import json
import random
import wave
import tempfile
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

st.set_page_config(page_title="German Vocab Trainer", page_icon="🎤")


# ============================================================
# Load Vocabulary
# ============================================================

@st.cache_data
def load_vocab(path="german_vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

vocab_all = load_vocab()


# ============================================================
# Organize Sets
# ============================================================

@st.cache_data
def get_sets(vocab):
    return sorted({entry["set"] for entry in vocab})

sets_available = get_sets(vocab_all)


# ============================================================
# Google Speech Client
# ============================================================

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]  # nested TOML table
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()


def transcribe_wav_file(path, sample_rate, channels):
    with open(path, "rb") as f:
        wav_bytes = f.read()

    audio = speech.RecognitionAudio(content=wav_bytes)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="de-DE",
        audio_channel_count=channels,
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        return response.results[0].alternatives[0].transcript.lower()
    return ""


# ============================================================
# Session State Initialization
# ============================================================

if "selected_set" not in st.session_state:
    st.session_state.selected_set = sets_available[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"

if "progress" not in st.session_state:
    st.session_state.progress = {}

def init_progress(set_name):
    if set_name not in st.session_state.progress:
        st.session_state.progress[set_name] = {
            "reviewed": set(),
            "correct": 0,
            "wrong": 0,
            "mistakes": []
        }

init_progress(st.session_state.selected_set)


# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.header("Settings")

set_choice = st.sidebar.selectbox("Select vocabulary set", sets_available)

if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    init_progress(set_choice)

mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

progress = st.session_state.progress[st.session_state.selected_set]

total_items = len([v for v in vocab_all if v["set"] == st.session_state.selected_set])

st.sidebar.markdown(f"""
### Progress
- Total: **{total_items}**
- Reviewed: **{len(progress["reviewed"]) }**
- Correct: **{progress["correct"]}**
- Wrong: **{progress["wrong"]}**
""")

if st.session_state.mode == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left: **{len(progress['mistakes'])}**")


# ============================================================
# Select Current Word
# ============================================================

filtered_vocab = [v for v in vocab_all if v["set"] == st.session_state.selected_set]

if st.session_state.mode == "Review Mistakes":
    filtered_vocab = progress["mistakes"][:]

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [v for v in filtered_vocab if v["word"] not in progress["reviewed"]]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        st.session_state.current = random.choice(filtered_vocab) if filtered_vocab else None

if "current" not in st.session_state:
    pick_new_word()

entry = st.session_state.current

st.title("🎤 German Pronunciation Trainer")

if entry is None:
    msg = "You have reviewed all items! 🎉" if st.session_state.mode == "Study" else "No more mistakes! 🎉"
    st.success(msg)
    st.stop()


# ============================================================
# Display English Meaning Only
# ============================================================

meaning = entry["meaning"]
g_singular = f"{entry['gender']} {entry['word']}".lower()
g_plural = entry["plural"].lower()

st.markdown(f"""
## Meaning:
**{meaning}**

Say aloud:
**{entry['gender']} {entry['word']} — {entry['plural']}**
""")


# ============================================================
# Recording (HTML5)
# ============================================================

audio_data = st.audio_input("Press to record your German pronunciation")

if audio_data is not None:

    # Save to temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data.read())
        tmp_path = tmp.name

    # Read WAV metadata
    with wave.open(tmp_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    # Auto-transcribe
    st.write("⏳ Transcribing…")
    transcript = transcribe_wav_file(tmp_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    # Check correctness
    correct = g_singular in transcript and g_plural in transcript

    progress["reviewed"].add(entry["word"])

    if correct:
        st.success("Correct! 🎉")
        progress["correct"] += 1
        if entry in progress["mistakes"]:
            progress["mistakes"].remove(entry)
    else:
        st.error("Not quite.")
        progress["wrong"] += 1
        if entry not in progress["mistakes"]:
            progress["mistakes"].append(entry)

    # Reveal correct forms and example sentence
    examples = entry["examples"] if entry["examples"] else "_None provided_"
    st.markdown(f"""
### Correct forms:
- **{entry['gender']} {entry['word']}**
- **{entry['plural']}**

### Example:
{examples}
""")

    if st.button("Next word"):
        pick_new_word()
        st.rerun()
