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
# Get available sets
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
    creds = st.secrets["google"]["credentials"]  # nested dict
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
# POS-Aware Matching Logic
# ============================================================

def check_answer(entry, transcript):
    """Return True if the ASR transcript contains the expected German word(s)."""

    t = transcript.lower().strip()
    pos = entry["pos"]
    word = entry["word"].lower().strip()
    gender = entry["gender"].lower().strip()
    plural = entry["plural"].lower().strip()

    # ---------------------------------------
    # VERBS / REFLEXIVE VERBS
    # ---------------------------------------
    if pos in ["verb", "reflexive verb"]:
        return word in t

    # ---------------------------------------
    # ADJECTIVES / ADVERBS / PARTICIPLES
    # ---------------------------------------
    if pos.startswith("adjective") or pos == "adverb":
        return word in t

    # ---------------------------------------
    # NON-NOUNS fallback
    # ---------------------------------------
    if pos not in ["noun", "noun (compound)", "noun (uncountable)"]:
        return word in t

    # ---------------------------------------
    # NOUNS – must check singular + plural
    # ---------------------------------------
    singular_form = f"{gender} {word}".strip()

    singular_ok = (singular_form in t) or (word in t)

    # plural may be optional (e.g. uncountable nouns)
    if pos == "noun (uncountable)" or plural == "":
        return singular_ok

    plural_variants = {plural}
    if plural.endswith("e"):
        plural_variants.add(plural[:-1])
    if plural.endswith("en"):
        plural_variants.add(plural[:-2])

    plural_ok = any(pv in t for pv in plural_variants)

    return singular_ok and plural_ok


# ============================================================
# Initialize Session State
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

# Select set
set_choice = st.sidebar.selectbox("Select vocabulary set", sets_available)
if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    init_progress(set_choice)

# Study or Review Mistakes
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
# Filter and Pick Word
# ============================================================

filtered = [v for v in vocab_all if v["set"] == st.session_state.selected_set]

if st.session_state.mode == "Review Mistakes":
    filtered = progress["mistakes"][:]

def pick_new_word():
    if st.session_state.mode == "Study":
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
    msg = (
        "You're done with this set! 🎉"
        if st.session_state.mode == "Study"
        else "No mistakes left! 🎉"
    )
    st.success(msg)
    st.stop()


# ============================================================
# Show ONLY the English
# ============================================================

st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German word:

- If it's a noun → say **article + singular**, then plural  
- If it's a verb → say the infinitive  
- If it's an adjective or adverb → say the lemma  
""")

# ============================================================
# Audio Recorder (HTML5 file input)
# ============================================================

audio_file = st.audio_input(
    "Press to record your pronunciation",
    key=f"audio_{entry['word']}"
)

if audio_file is not None:

    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # Read metadata
    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing…")
    transcript = transcribe_wav_file(audio_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    # Evaluate correctness
    correct = check_answer(entry, transcript)

    # Update progress
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

    # Reveal correct answers
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
