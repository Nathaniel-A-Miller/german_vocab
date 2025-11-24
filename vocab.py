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
# Load ALL vocab files from /german_vocab folder
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
    countability = entry.get("countability", "countable")

    # -----------------------------
    # VERBS (including reflexive)
    # -----------------------------
    if pos in ["verb", "reflexive verb"]:
        parts = word.split()
        return all(p in tokens for p in parts)

    # -----------------------------
    # ADJECTIVES / ADVERBS
    # -----------------------------
    if "adjective" in pos or "adverb" in pos:
        return word in tokens

    # -----------------------------
    # NON-noun fallback
    # -----------------------------
    if not pos.startswith("noun"):
        return word in tokens

    # -----------------------------
    # STRICT NOUN MATCHING
    # -----------------------------
    singular_form = f"{gender} {word}".strip()
    
    # Require exact singular with article
    singular_ok = singular_form in t
    
    # Uncountable nouns (plural empty or marked)
    if plural == "" or plural == "—":
        return singular_ok
    
    # Require plural explicitly
    plural_ok = plural in tokens
    
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

if st.session_state.selected_file not in st.session_state.progress:
    st.session_state.progress[st.session_state.selected_file] = {
        "reviewed": set(),
        "correct": 0,
        "wrong": 0,
        "mistakes": []
    }

progress = st.session_state.progress[st.session_state.selected_file]

# Queue for Review Mistakes (finite cycle)
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Vocabulary Source File")

# File selector
file_choice = st.sidebar.selectbox("Choose vocab JSON file", vocab_files)

if file_choice != st.session_state.selected_file:
    st.session_state.selected_file = file_choice
    if file_choice not in st.session_state.progress:
        st.session_state.progress[file_choice] = {
            "reviewed": set(),
            "correct": 0,
            "wrong": 0,
            "mistakes": []
        }
    progress = st.session_state.progress[file_choice]

filtered_vocab = [v for v in vocab_all if v["source_file"] == st.session_state.selected_file]

# Mode selector
mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])

# Handle mode change
if mode_choice != st.session_state.get("mode"):
    st.session_state.mode = mode_choice

    if mode_choice == "Review Mistakes":
        if not progress["mistakes"]:
            st.success("No mistakes to review! 🎉")
            st.stop()

        # Build new finite review queue
        st.session_state.review_queue = progress["mistakes"].copy()

    st.session_state._pending_mode_switch = True

else:
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
    st.sidebar.markdown(f"### Mistakes left: **{len(st.session_state.review_queue)}**")

# ============================================================
# Select a word
# ============================================================

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [v for v in filtered_vocab if v["word"] not in progress["reviewed"]]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        # Review Mistakes: pop first item
        if not st.session_state.review_queue:
            st.session_state.current = None
        else:
            st.session_state.current = st.session_state.review_queue[0]

# Handle pending mode switch
if st.session_state.get("_pending_mode_switch"):
    del st.session_state["_pending_mode_switch"]
    pick_new_word()
    st.rerun()

# Initial pick
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
# Prompt (English only)
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

    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing...")

    transcript = transcribe_wav_file(audio_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    correct = check_answer(entry, transcript)
    first_time = entry["word"] not in progress["reviewed"]

    # ===== Update progress =====

    if correct:
        st.success("Correct! 🎉")

        # Study mode: standard update
        if first_time:
            progress["correct"] += 1

        # Review mode: remove from queue
        if entry in st.session_state.review_queue:
            st.session_state.review_queue.remove(entry)

        # If in mistakes, remove
        if entry in progress["mistakes"]:
            progress["mistakes"].remove(entry)

    else:
        st.error("Not quite.")

        if first_time:
            progress["wrong"] += 1

        # Add to mistakes list if not already
        if entry not in progress["mistakes"]:
            progress["mistakes"].append(entry)

        # Review mode: incorrect → move to back of queue
        if st.session_state.mode == "Review Mistakes":
            if entry in st.session_state.review_queue:
                st.session_state.review_queue.append(
                    st.session_state.review_queue.pop(0)
                )

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
