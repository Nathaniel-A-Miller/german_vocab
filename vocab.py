import streamlit as st
import os
import json
import random
import wave
import tempfile
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

st.set_page_config(page_title="German Vocab Trainer", page_icon="🎤")

# ============================================================
# Load vocabulary
# ============================================================

@st.cache_data
def load_vocab(folder="german_vocab"):
    vocab = []
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    for file in files:
        path = os.path.join(folder, file)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                item["source_file"] = file
            vocab.extend(items)

    return vocab, files

vocab_all, vocab_files = load_vocab()

# ============================================================
# Google Speech Client
# ============================================================

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()

def transcribe(path, rate, channels):
    with open(path, "rb") as f:
        audio_bytes = f.read()

    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="de-DE",
        sample_rate_hertz=rate,
        audio_channel_count=channels,
    )

    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript.lower()
    return ""

# ============================================================
# Answer checking logic
# ============================================================

def check(entry, transcript):
    t = transcript.lower().strip()
    tokens = t.split()

    pos = entry["pos"]
    word = entry["word"].lower()
    gender = entry["gender"].lower().strip()
    plural = entry["plural"].lower().strip()

    # Verb and reflexive verb: match all components
    if pos in ["verb", "reflexive verb"]:
        parts = word.split()
        return all(p in tokens for p in parts)

    # Adjectives/adverbs
    if "adjective" in pos or "adverb" in pos:
        return word in tokens

    # Non-noun fallback
    if not pos.startswith("noun"):
        return word in tokens

    # Noun strict matching
    singular_tokens = f"{gender} {word}".split()
    plural_tokens = plural.split() if plural else []

    # singular match
    singular_ok = any(
        tokens[i:i+len(singular_tokens)] == singular_tokens
        for i in range(len(tokens) - len(singular_tokens) + 1)
    )

    if plural in ["", "—"]:
        return singular_ok

    # plural match
    plural_ok = any(
        tokens[i:i+len(plural_tokens)] == plural_tokens
        for i in range(len(tokens) - len(plural_tokens) + 1)
    )

    return singular_ok and plural_ok

# ============================================================
# Session state initialization
# ============================================================

def init_progress(filename):
    if "progress" not in st.session_state:
        st.session_state.progress = {}

    if filename not in st.session_state.progress:
        st.session_state.progress[filename] = {
            "reviewed": set(),
            "correct": 0,
            "wrong": 0,
            "mistakes": []
        }

if "selected_file" not in st.session_state:
    st.session_state.selected_file = vocab_files[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"

if "current" not in st.session_state:
    st.session_state.current = None

if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

init_progress(st.session_state.selected_file)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Vocabulary Source File")

file_choice = st.sidebar.selectbox(
    "Choose vocab JSON file",
    vocab_files,
    index=vocab_files.index(st.session_state.selected_file)
)

if file_choice != st.session_state.selected_file:
    st.session_state.selected_file = file_choice
    init_progress(file_choice)
    st.session_state.current = None
    st.session_state.review_queue = []

progress = st.session_state.progress[st.session_state.selected_file]
filtered_vocab = [
    v for v in vocab_all if v["source_file"] == st.session_state.selected_file
]

mode_choice = st.sidebar.selectbox(
    "Mode",
    ["Study", "Review Mistakes"],
    index=["Study", "Review Mistakes"].index(st.session_state.mode)
)

if mode_choice != st.session_state.mode:
    st.session_state.mode = mode_choice
    st.session_state.current = None

    if mode_choice == "Review Mistakes":
        mistakes = progress["mistakes"]
        st.session_state.review_queue = list(dict.fromkeys(mistakes))

# Sidebar progress display
st.sidebar.markdown(f"""
### Progress
- Total: **{len(filtered_vocab)}**
- Reviewed (Study): **{len(progress['reviewed'])}**
- Correct: **{progress['correct']}**
- Wrong: **{progress['wrong']}**
""")

if st.session_state.mode == "Review Mistakes":
    st.sidebar.markdown(
        f"### Mistakes left this round: **{len(st.session_state.review_queue)}**"
    )

# ============================================================
# Word selection
# ============================================================

def pick_word():
    if st.session_state.mode == "Study":
        remaining = [
            v for v in filtered_vocab
            if v["word"] not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        q = st.session_state.review_queue
        if not q:
            st.session_state.current = None
        else:
            w = q[0]
            st.session_state.current = next(v for v in filtered_vocab if v["word"] == w)

if st.session_state.current is None:
    pick_word()

entry = st.session_state.current

# ============================================================
# Main UI
# ============================================================

st.title("🎤 German Vocab Trainer")

if entry is None:
    if st.session_state.mode == "Study":
        st.success("You finished all items! 🎉")
    else:
        st.success("No mistakes left! 🎉")
    st.stop()

# Prompt
st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German form:
""")

# ============================================================
# Audio input — immediate evaluation
# ============================================================

audio_input = st.audio_input(
    "Record your pronunciation",
    key=f"audio_{entry['word']}"
)

if audio_input:
    # Save temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    with wave.open(audio_path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()

    st.write("Transcribing…")

    transcript = transcribe(audio_path, rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    correct = check(entry, transcript)
    first_time = entry["word"] not in progress["reviewed"]

    # ========== Study mode ==========
    if st.session_state.mode == "Study":
        if correct:
            st.success("Correct! 🎉")
            if first_time:
                progress["correct"] += 1
            if entry["word"] in progress["mistakes"]:
                progress["mistakes"].remove(entry["word"])
        else:
            st.error("Not quite.")
            if first_time:
                progress["wrong"] += 1
            if entry["word"] not in progress["mistakes"]:
                progress["mistakes"].append(entry["word"])

    # ========== Review Mistakes mode ==========
    else:
        if correct:
            st.success("Correct! 🎉")
            if entry["word"] in progress["mistakes"]:
                progress["mistakes"].remove(entry["word"])
            st.session_state.review_queue.pop(0)
        else:
            st.error("Not quite.")
            w = st.session_state.review_queue.pop(0)
            st.session_state.review_queue.append(w)

    progress["reviewed"].add(entry["word"])

    # Reveal answer
    st.markdown(f"""
### Correct German:
- **{entry['word']}**
- POS: **{entry['pos']}**
- Gender: **{entry['gender'] or "—"}**
- Plural: **{entry['plural'] or "—"}**

### Example:
{entry['examples'][0] if entry['examples'] else "_None provided_"}
""")

    if st.button("Next"):
        pick_word()
        st.rerun()
