import streamlit as st
import random
import json
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

set_choice = st.sidebar.selectbox("Select vocabulary set", sets_available)
if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    init_progress(set_choice)

mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

progress = st.session_state.progress[st.session_state.selected_set]

# Sidebar stats
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
    filtered_vocab = progress["mistakes"][:]  # copy


# ============================================================
# Select current word
# ============================================================

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

st.markdown(f"""
### Meaning:
**{entry['meaning']}**

Pronounce:

- **{entry['gender']} {entry['word']}**
- **{entry['plural']}**
""")


# ============================================================
# Audio Recorder — FIXED VERSION
# ============================================================

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.count = 0

    def recv_audio_frame(self, frame):
        self.count += 1
        self.frames.append(frame)
        return frame



st.write("Press **Start**, then speak, then press **Stop**.")

webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_receiver_size=1024,
    media_stream_constraints={
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False
    },
)


# === DEBUG: Check if audio frames are coming in ===
if webrtc_ctx and webrtc_ctx.state.playing:
    st.write(f"🎧 Frames received (live): {webrtc_ctx.audio_processor.count}")
else:
    st.write("⚠️ WebRTC not playing yet")


# ============================================================
# Google Speech Client
# ============================================================

@st.cache_resource
def get_speech_client():
    key_dict = st.secrets["google"]["credentials"]  # already a dict via TOML tables
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
# Submit audio — FIXED VERSION
# ============================================================

if st.button("Submit"):

    # --- DEBUG: check processor and frames ---
    if not webrtc_ctx:
        st.error("❌ webrtc_ctx is None (WebRTC failed to initialize)")
        st.stop()

    if not hasattr(webrtc_ctx, "audio_processor") or webrtc_ctx.audio_processor is None:
        st.error("❌ audio_processor missing")
        st.stop()

    st.write(f"Frames in processor: {webrtc_ctx.audio_processor.count}")

    audio_frames = webrtc_ctx.audio_processor.frames

    if not audio_frames:
        st.error("❌ No frames recorded. Did you press Stop?")
        st.stop()

    # --- Assemble bytes from all frames ---
    raw_bytes = b"".join(
        frame.to_ndarray().astype(np.int16).tobytes()
        for frame in audio_frames
    )

    st.write(f"Raw audio bytes length: {len(raw_bytes)}")

    if len(raw_bytes) < 3000:
        st.error("❌ Audio too short — try speaking for 1–2 seconds.")
        st.stop()

    st.write("⏳ Transcribing...")
    text = transcribe_google(raw_bytes)

    st.markdown(f"### You said:\n**{text}**")

    correct = check_match(text, singular_phrase, plural_form)

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
