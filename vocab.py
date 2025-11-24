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
# Load ALL vocab
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
# POS / lemma checking
# ============================================================

def check_answer(entry, transcript):
    t = transcript.lower().strip()
    tokens = t.split()

    pos = entry["pos"]
    word = entry["word"].lower().strip()
    gender = entry["gender"].lower().strip()
    plural = entry["plural"].lower().strip()

    # VERBS
    if pos in ["verb", "reflexive verb"]:
        parts = word.split()
        return all(p in tokens for p in parts)

    # ADJECTIVES / ADVERBS
    if "adjective" in pos or "adverb" in pos or pos == "adjective":
        return word in tokens

    # OTHER NON-NOUN
    if not pos.startswith("noun"):
        return word in tokens

    # NOUNS
    singular_tokens = f"{gender} {word}".split()
    plural_tokens = plural.split() if plural else []

    sing_ok = any(
        tokens[i:i + len(singular_tokens)] == singular_tokens
        for i in range(len(tokens) - len(singular_tokens) + 1)
    )

    if plural in ["", "—"]:
        return sing_ok

    plur_ok = any(
        tokens[i:i + len(plural_tokens)] == plural_tokens
        for i in range(len(tokens) - len(plural_tokens) + 1)
    )

    return sing_ok and plur_ok


# ============================================================
# Session state initialization
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

# Mistakes review queue
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

# Prevent double-evaluation of the same audio
if "last_word_eval" not in st.session_state:
    st.session_state.last_word_eval = None

# Current word
if "current" not in st.session_state:
    st.session_state.current = None


# ============================================================
# Sidebar UI
# ============================================================

st.sidebar.header("Vocabulary Source File")

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
    st.session_state.review_queue = []
    st.session_state.current = None

filtered_vocab = [v for v in vocab_all if v["source_file"] == st.session_state.selected_file]

mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

# Build review queue fresh when entering Review Mode
if mode_choice == "Review Mistakes" and not st.session_state.review_queue:
    unique = list(dict.fromkeys(progress["mistakes"]))
    st.session_state.review_queue = unique.copy()


# Sidebar progress
st.sidebar.markdown(f"""
### Progress
- Total: **{len(filtered_vocab)}**
- Reviewed (Study): **{len(progress["reviewed"])}**
- Correct: **{progress["correct"]}**
- Wrong: **{progress["wrong"]}**
""")

if mode_choice == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left this round: **{len(st.session_state.review_queue)}**")


# ============================================================
# Pick next word
# ============================================================

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [
            v for v in filtered_vocab
            if v["word"] not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None

    else:  # Review Mistakes
        queue = st.session_state.review_queue
        if not queue:
            st.session_state.current = None
        else:
            st.session_state.current = next(
                v for v in filtered_vocab if v["word"] == queue[0]
            )


# If no current word selected yet
if st.session_state.current is None:
    pick_new_word()

entry = st.session_state.current

if entry is None:
    if mode_choice == "Study":
        st.success("🎉 You finished all items in this set!")
    else:
        st.success("🎉 All mistakes corrected!")
    st.stop()


# ============================================================
# Prompt
# ============================================================

st.title("🎤 German Vocab Trainer")

st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German form:
- **Nouns**: article + singular, then plural  
- **Verbs**: infinitive  
- **Adjectives / Adverbs**: lemma  
""")


# ============================================================
# Audio input — stable key prevents double-eval
# ============================================================

audio_input = st.audio_input("Record your pronunciation", key="stable_audio_key")


# ============================================================
# Evaluation
# ============================================================

if audio_input:

    # Avoid evaluating same audio twice
    if st.session_state.last_word_eval == entry["word"]:
        st.stop()
    st.session_state.last_word_eval = entry["word"]

    # Save tmp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    # Extract WAV info
    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    transcript = transcribe_wav_file(audio_path, sample_rate, channels)
    st.markdown(f"### You said:\n**{transcript}**")

    correct = check_answer(entry, transcript)
    first_time = entry["word"] not in progress["reviewed"]

    # ============================================================
    # Correct
    # ============================================================
    if correct:
        st.success("Correct! 🎉")

        if first_time:
            progress["correct"] += 1

        # Remove from global mistakes list
        if entry["word"] in progress["mistakes"]:
            progress["mistakes"].remove(entry["word"])

        # Remove from review queue safely
        if st.session_state.mode == "Review Mistakes":
            queue = st.session_state.review_queue
            if entry["word"] in queue:
                queue.remove(entry["word"])

    # ============================================================
    # Wrong
    # ============================================================
    else:
        st.error("Not quite.")

        if first_time:
            progress["wrong"] += 1

        if entry["word"] not in progress["mistakes"]:
            progress["mistakes"].append(entry["word"])

        # Safe rotation
        if st.session_state.mode == "Review Mistakes":
            queue = st.session_state.review_queue

            if entry["word"] in queue:
                # If only ONE item left: repeat immediately (your rule)
                if len(queue) == 1:
                    pass  # no rotation

                # More than one item: rotate to the back
                elif len(queue) > 1:
                    i = queue.index(entry["word"])
                    w = queue.pop(i)
                    queue.append(w)

    # Mark reviewed (Study mode only)
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
        pick_new_word()
        st.rerun()
