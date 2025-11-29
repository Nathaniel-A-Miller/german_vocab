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
# HELPERS
# ============================================================

def make_key(entry):
    return f"{entry['source_file']}::{entry['word']}"

# ============================================================
# LOAD ALL VOCAB FILES
# ============================================================

@st.cache_data
def load_all_vocab(folder="german_vocab"):
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

vocab_all, vocab_files = load_all_vocab()

# ============================================================
# GOOGLE SPEECH CLIENT
# ============================================================

@st.cache_resource
def get_client():
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()

def transcribe_wav_file(path, sample_rate, channels):
    with open(path, "rb") as f:
        audio_bytes = f.read()

    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="de-DE",
        sample_rate_hertz=sample_rate,
        audio_channel_count=channels,
    )

    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript.lower()
    return ""

# ============================================================
# CHECK ANSWER
# ============================================================

def check_answer(entry, transcript):
    t = transcript.lower().strip()

    pos = entry["pos"].lower()
    word = entry["word"].lower().strip()
    gender = entry["gender"].lower().strip()
    plural = entry["plural"].lower().strip()

    # VERBS — exact infinitive only
    if pos in ["verb", "reflexive verb"]:
        return t == word

    # ADJECTIVES / ADVERBS — lemma only
    if "adjective" in pos or "adverb" in pos:
        return t == word

    # NON-NOUNS fallback
    if not pos.startswith("noun"):
        return t == word

    # NOUNS — exact singular or exact plural only
    singular_form = f"{gender} {word}".strip()
    plural_form = plural

    return t == singular_form or (plural_form and t == plural_form)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "selected_file" not in st.session_state:
    st.session_state.selected_file = vocab_files[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"

if "progress" not in st.session_state:
    st.session_state.progress = {}

if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

if "current" not in st.session_state:
    st.session_state.current = None

# ============================================================
# SIDEBAR — FILE SELECTION
# ============================================================

st.sidebar.header("Vocabulary Source File")
file_choice = st.sidebar.selectbox(
    "Choose vocab JSON file",
    vocab_files,
    key="file_selector"
)

# ============================================================
# FILE SWITCH HANDLING
# ============================================================

if file_choice != st.session_state.selected_file:

    st.session_state.selected_file = file_choice

    st.session_state.progress[file_choice] = {
        "reviewed": set(),
        "correct": 0,
        "wrong": 0,
        "mistakes": []
    }

    st.session_state.review_queue = []
    st.session_state.current = None

    st.rerun()

# ============================================================
# PROGRESS BINDING
# ============================================================

progress = st.session_state.progress.setdefault(
    st.session_state.selected_file,
    {"reviewed": set(), "correct": 0, "wrong": 0, "mistakes": []}
)

# ============================================================
# FILTER VOCAB
# ============================================================

filtered_vocab = [
    v for v in vocab_all
    if v["source_file"] == st.session_state.selected_file
]

# ============================================================
# SIDEBAR MODE SELECTION
# ============================================================

mode_choice = st.sidebar.selectbox(
    "Mode",
    ["Study", "Review Mistakes", "Easy Mode (Show German)"],
    key="mode_selector"
)

if mode_choice != st.session_state.mode:
    st.session_state.mode = mode_choice

    if mode_choice == "Review Mistakes":
        if not progress["mistakes"]:
            st.success("No mistakes to review! 🎉")
            st.stop()
        st.session_state.review_queue = list(dict.fromkeys(progress["mistakes"]))

    st.session_state.current = None
    st.rerun()

# ============================================================
# DISPLAY PROGRESS
# ============================================================

st.sidebar.markdown(f"""
### Progress
- Total: **{len(filtered_vocab)}**
- Reviewed: **{len(progress['reviewed'])}**
- Correct: **{progress['correct']}**
- Wrong: **{progress['wrong']}**
""")

if st.session_state.mode == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left: **{len(st.session_state.review_queue)}**")

# ============================================================
# PICK NEXT WORD
# ============================================================

def pick_new_word():
    if st.session_state.mode in ["Study", "Easy Mode (Show German)"]:
        remaining = [
            v for v in filtered_vocab
            if make_key(v) not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        if not st.session_state.review_queue:
            st.session_state.current = None
        else:
            key = st.session_state.review_queue[0]
            st.session_state.current = next(
                v for v in filtered_vocab if make_key(v) == key
            )

if st.session_state.current is None:
    pick_new_word()

entry = st.session_state.current

if entry is None:
    if st.session_state.mode == "Study":
        st.success("You finished all items in this set! 🎉")
    else:
        st.success("No mistakes left! 🎉")
    st.stop()

# ============================================================
# PROMPT
# ============================================================

if st.session_state.mode == "Study":
    st.markdown(f"""
### Say the correct German for:

## *{entry['meaning']}*
""")

elif st.session_state.mode == "Easy Mode (Show German)":
    st.markdown(f"""
### Say this German aloud:

**{entry['gender']} {entry['word']}**

**Plural:** {entry['plural'] or "—"}

**Meaning:** *{entry['meaning']}*
""")

st.markdown("""
- **Noun:** article + singular → plural  
- **Verb:** infinitive  
- **Reflexive verb:** both parts  
""")

# ============================================================
# RECORD AUDIO
# ============================================================

st.session_state.pop("audio_input", None)

audio_input = st.audio_input(
    "Press to record your pronunciation",
    key=f"audio_{make_key(entry)}"
)

if audio_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing...")
    transcript = transcribe_wav_file(audio_path, sample_rate, channels)
    st.markdown(f"### You said:\n**{transcript}**")

    correct = check_answer(entry, transcript)
    key = make_key(entry)
    first_time = key not in progress["reviewed"]

    if correct:
        st.success("Correct! 🎉")
        if first_time:
            progress["correct"] += 1
        if key in progress["mistakes"]:
            progress["mistakes"].remove(key)
        if st.session_state.mode == "Review Mistakes":
            if key in st.session_state.review_queue:
                st.session_state.review_queue.remove(key)
    else:
        st.error("Not quite.")
        if first_time:
            progress["wrong"] += 1
        if key not in progress["mistakes"]:
            progress["mistakes"].append(key)
        if st.session_state.mode == "Review Mistakes":
            if key in st.session_state.review_queue:
                st.session_state.review_queue.append(
                    st.session_state.review_queue.pop(0)
                )

    progress["reviewed"].add(key)

    example = entry["examples"][0] if entry["examples"] else "_None provided_"

    st.markdown(f"""
### Correct German:
- **{entry['word']}**
- POS: **{entry['pos']}**
- Gender: **{entry['gender']}**
- Plural: **{entry['plural']}**

### Example:
{example}
""")

    if st.button("Next"):
        st.session_state.current = None
        st.rerun()
