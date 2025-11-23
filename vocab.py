import streamlit as st
import random
import json
import io
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

st.set_page_config(page_title="German Vocab Trainer", page_icon="🎤")


# ============================================================
# Load vocab
# ============================================================

@st.cache_data
def load_vocab(path="german_vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

vocab_all = load_vocab()


# ============================================================
# Extract all sets
# ============================================================

@st.cache_data
def get_sets(vocab):
    return sorted({entry["set"] for entry in vocab})

sets_available = get_sets(vocab_all)


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
# Sidebar
# ============================================================

st.sidebar.header("Settings")

# --- Select set ---
set_choice = st.sidebar.selectbox("Select vocabulary set", sets_available)
if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    init_progress(set_choice)

# --- Study mode or Review Mistakes ---
mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

progress = st.session_state.progress[st.session_state.selected_set]

# --- Progress display ---
total_items = len([v for v in vocab_all if v["set"] == st.session_state.selected_set])
reviewed_count = len(progress["reviewed"])
correct_count = progress["correct"]
wrong_count = progress["wrong"]

st.sidebar.markdown(f"""
### Progress  
- Total items: **{total_items}**  
- Reviewed: **{reviewed_count}**  
- Correct: **{correct_count}**  
- Wrong: **{wrong_count}**  
""")

if st.session_state.mode == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left: **{len(progress['mistakes'])}**")


# ============================================================
# Filter vocab by set & mode
# ============================================================

filtered_vocab = [v for v in vocab_all if v["set"] == st.session_state.selected_set]

if st.session_state.mode == "Review Mistakes":
    filtered_vocab = progress["mistakes"][:]  # list copy


# ============================================================
# Select current word
# ============================================================

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [v for v in filtered_vocab if v["word"] not in progress["reviewed"]]
        if not remaining:
            st.session_state.current = None
        else:
            st.session_state.current = random.choice(remaining)
    else:  # Review Mistakes
        if not filtered_vocab:
            st.session_state.current = None
        else:
            st.session_state.current = random.choice(filtered_vocab)

if "current" not in st.session_state:
    pick_new_word()

entry = st.session_state.current

st.title("🎤 German Pronunciation Trainer")

if entry is None:
    if st.session_state.mode == "Study":
        st.success("You have reviewed all items in this set! 🎉")
    else:
        st.success("No more mistakes to review! 🎉")
    st.stop()


# ============================================================
# Prepare target forms
# ============================================================

singular_phrase = f"{entry['gender']} {entry['word']}".lower()
plural_form = entry['plural'].lower()

# Display meaning
st.markdown(f"""
### Meaning:
**{entry['meaning']}**

Pronounce:

- **{entry['gender']} {entry['word']}**
- **{entry['plural']}**
""")


# ============================================================
# Audio Recorder
# ============================================================

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio_frame(self, frame):
        self.frames.append(frame.to_ndarray())
        return frame

st.write("Press **Start** and speak.")

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
)


# ============================================================
# Google Speech Client
# ============================================================

@st.cache_resource
def get_speech_client():
    st.write("SECRETS RAW STRING:")
    st.write(repr(st.secrets["google"]["credentials"]))
    key_dict = json.loads(st.secrets["google"]["credentials"])
    credentials = service_account.Credentials.from_service_account_info(key_dict)
    return speech.SpeechClient(credentials=credentials)

client = get_speech_client()


def transcribe_google(raw_audio_bytes):
    audio = speech.RecognitionAudio(content=raw_audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="de-DE",
    )
    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript.lower()
    return ""


def check_match(asr_text, singular, plural):
    return singular in asr_text and plural in asr_text


# ============================================================
# Submit recording
# ============================================================

if st.button("Submit"):
    if not (webrtc_ctx and webrtc_ctx.audio_receiver):
        st.warning("No audio captured.")
        st.stop()

    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
    if not audio_frames:
        st.warning("No audio captured.")
        st.stop()

    audio_ndarray = audio_frames[0].to_ndarray().flatten()
    raw_bytes = audio_ndarray.astype(np.int16).tobytes()

    st.write("⏳ Transcribing...")
    text = transcribe_google(raw_bytes)

    st.markdown(f"### You said:\n**{text}**")

    correct = check_match(text, singular_phrase, plural_form)

    # Update progress
    progress["reviewed"].add(entry["word"])

    if correct:
        st.success("Correct! 🎉")
        progress["correct"] += 1

        # remove from mistakes if reviewing
        if entry in progress["mistakes"]:
            progress["mistakes"].remove(entry)

    else:
        st.error("Not quite.")
        progress["wrong"] += 1

        # add to mistakes if not already present
        if entry not in progress["mistakes"]:
            progress["mistakes"].append(entry)

    # Show correct forms & examples
    st.markdown(f"""
### Correct forms:
- **{entry['gender']} {entry['word']}**
- **{entry['plural']}**

### Examples:
{entry['examples'] if entry['examples'] else '_None provided_'}
""")

    if st.button("Next word"):
        pick_new_word()
        st.rerun()
