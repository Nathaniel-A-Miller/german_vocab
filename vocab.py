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
# Load ALL vocab files from /vocab folder
# ============================================================

@st.cache_data
def load_all_vocab(folder="vocab"):
    vocab = []
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    for file in files:
        path = os.path.join(folder, file)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                item["source_file"] = file   # Tag each entry with file
            vocab.extend(items)

    return vocab, files


vocab_all, vocab_files = load_all_vocab()


# ============================================================
# Google Speech Client
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
# POS / Whole-word / Reflexive matching logic
# ============================================================

def check_answer(entry, transcript):
    t = transcript.lower().strip()
    tokens = t.split()

    pos = entry["pos"]
    word = entry["word"].lower().strip()
    gender = entry["gender"].lower().strip()
    plural = entry["plural"].lower().strip()

    # -----------------------------
    # VERBS (including reflexive)
    # -----------------------------
    if pos in ["verb", "reflexive verb"]:

        # If exact phrase appears
        if word in t:
            return True

        # Split reflexive verbs
        parts = word.split()
        if len(parts) > 1:
            return all(p in tokens for p in parts)

        return word in tokens

    # -----------------------------
    # ADJECTIVES / ADVERBS
    # -----------------------------
    if "adjective" in pos or "adverb" in pos:
        return word in tokens

    # -----------------------------
    # NON-NOUN fallback
    # -----------------------------
    if not pos.startswith("noun"):
        return word in tokens

    # -----------------------------
    # NOUNS — whole word matching
    # -----------------------------
    singular_form = f"{gender} {word}".strip()

    singular_ok = (singular_form in t) or (word in tokens)

    # Uncountable nouns
    if plural == "":
        return singular_ok

    # Require plural explicitly
    plural_ok = plural in tokens or plural in t.split()

    return singular_ok and plural_ok


# ============================================================
# Session State
# ============================================================

if "selected_file" not in st.session_state:
    st.session_state.selected_file = vocab_files[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"

if "progress" not in st.session_state:
    st.session_state.progress = {}

# Ensure progress entry exists
if st.session_state.selected_file not in st.session_state.progress:
    st.session_state.progress[st.session_state.selected_file] = {
        "reviewed": set(),
        "correct": 0,
        "wrong": 0,
        "mistakes": []
    }

progress = st.session_state.progress[st.session_state.selected_file]


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Vocabulary Source File")

# ----- FILE SELECTOR -----
file_choice = st.sidebar.selectbox("Choose vocab JSON file", vocab_files)

if file_choice != st.session_state.selected_file:
    st.session_state.selected_file = file_choice
    # Reset progress for this file if needed
    if file_choice not in st.session_state.progress:
        st.session_state.progress[file_choice] = {
            "reviewed": set(),
            "correct": 0,
            "wrong": 0,
            "mistakes": []
        }
    progress = st.session_state.progress[file_choice]

filtered_vocab = [v for v in vocab_all if v["source_file"] == st.session_state.selected_file]


# ----- STUDY MODE / REVIEW MODE -----
mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

# Progress display
total_items = len(filtered_vocab)
reviewed_count = len(progress["reviewed"])

st.sidebar.markdown(f"""
### Progress
- Total: **{total_items}**
- Reviewed: **{reviewed_count}**
- Correct: **{progress['correct']}**
- Wrong: **{progress['wrong']}**
""")

if mode_choice == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left: **{len(progress['mistakes'])}**")


# ============================================================
# Select a word
# ============================================================

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [v for v in filtered_vocab if v["word"] not in progress["reviewed"]]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        st.session_state.current = (
            random.choice(progress["mistakes"]) if progress["mistakes"] else None
        )


if "current" not in st.session_state:
    pick_new_word()

entry = st.session_state.current

st.title("🎤 German Vocab Trainer")

if entry is None:
    if mode_choice == "Study":
        st.success("You finished all items in this set! 🎉")
    else:
        st.success("No mistakes left! 🎉")
    st.stop()


# ============================================================
# PROMPT (English only)
# ============================================================

st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German form:
- **Noun:** say article + singular, then plural  
- **Verb:** say infinitive  
- **Reflexive verb:** say both parts  
- **Adjective/Adverb:** say lemma  
""")


# ============================================================
# Audio Input
# ============================================================

audio_input = st.audio_input(
    "Press to record your pronunciation",
    key=f"audio_{entry['word']}"
)

if audio_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    # Extract audio metadata
    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing...")

    transcript = transcribe_wav_file(audio_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    correct = check_answer(entry, transcript)
    first_time = entry["word"] not in progress["reviewed"]

    if correct:
        st.success("Correct! 🎉")
        if first_time:
            progress["correct"] += 1
            if entry in progress["mistakes"]:
                progress["mistakes"].remove(entry)
    else:
        st.error("Not quite.")
        if first_time:
            progress["wrong"] += 1
            if entry not in progress["mistakes"]:
                progress["mistakes"].append(entry)

    progress["reviewed"].add(entry["word"])

    # Reveal correct answer
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
        pick_new_word()
        st.rerun()
