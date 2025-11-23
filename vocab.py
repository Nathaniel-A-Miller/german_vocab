import streamlit as st
import json
import random
import wave
import tempfile
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

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
# Get Sets
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
    creds = st.secrets["google"]["credentials"]
    credentials = service_account.Credentials.from_service_account_info(creds)
    return speech.SpeechClient(credentials=credentials)

client = get_client()


def transcribe_wav_file(path, sample_rate, channels):
    with open(path, "rb") as f:
        wav_bytes = f.read()

    audio = speech.RecognitionAudio(content=wav_bytes)

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
# POS-Aware, Whole-Word Matching
# ============================================================

def check_answer(entry, transcript):
    """Return True if ASR transcript contains the expected forms."""

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

    # singular must appear as whole token or full phrase
    singular_ok = (singular_form in t) or (word in tokens)

    # uncountable nouns have no plural requirement
    if "uncountable" in pos or plural == "":
        return singular_ok

    # plural must appear EXACTLY
    plural_ok = plural in tokens or plural in t.split()

    return singular_ok and plural_ok


# ============================================================
# Session State
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

progress = st.session_state.progress[st.session_state.selected_set]


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Settings")

set_choice = st.sidebar.selectbox("Select vocabulary set", sets_available)
if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    init_progress(set_choice)
    progress = st.session_state.progress[set_choice]

mode_choice = st.sidebar.selectbox("Mode", ["Study", "Review Mistakes"])
st.session_state.mode = mode_choice

total_items = len([v for v in vocab_all if v["set"] == st.session_state.selected_set])

st.sidebar.markdown(f"""
### Progress
- Total: **{total_items}**
- Reviewed: **{len(progress['reviewed'])}**
- Correct: **{progress['correct']}**
- Wrong: **{progress['wrong']}**
""")

if mode_choice == "Review Mistakes":
    st.sidebar.markdown(f"### Mistakes left: **{len(progress['mistakes'])}**")


# ============================================================
# Pick Word
# ============================================================

filtered = [v for v in vocab_all if v["set"] == st.session_state.selected_set]

if mode_choice == "Review Mistakes":
    filtered = progress["mistakes"][:]


def pick_new_word():
    if mode_choice == "Study":
        remaining = [
            v for v in filtered
            if v["word"] not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        st.session_state.current = random.choice(filtered) if filtered else None


if "current" not in st.session_state:
    pick_new_word()

entry = st.session_state.current

st.title("🎤 German Vocab Trainer")

if entry is None:
    msg = "You're done with this set! 🎉" if mode_choice == "Study" else "No mistakes left! 🎉"
    st.success(msg)
    st.stop()


# ============================================================
# Prompt
# ============================================================

st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German word:

- **Noun** → say article + singular, then plural  
- **Verb** → say infinitive  
- **Adjective/Adverb** → say lemma  
""")


# ============================================================
# Audio Input (reset per word)
# ============================================================

audio_file = st.audio_input(
    "Press to record your pronunciation",
    key=f"audio_{entry['word']}"
)

if audio_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing…")
    transcript = transcribe_wav_file(audio_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    # Determine correctness
    correct = check_answer(entry, transcript)

    # Count only FIRST time we see this word
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

    # Mark word reviewed
    progress["reviewed"].add(entry["word"])

    # Reveal
    st.markdown(f"""
### Correct German:
- **Word:** {entry['word']}
- **POS:** {entry['pos']}
- **Gender:** {entry['gender'] or '—'}
- **Plural:** {entry['plural'] or '—'}

### Example:
{entry['examples'][0] if entry['examples'] else '_None provided_'}
""")

    if st.button("Next"):
        pick_new_word()
        st.rerun()
