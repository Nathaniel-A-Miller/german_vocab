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
# POS / Whole-word / Reflexive / Noun matching logic
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
        parts = word.split()
        return all(p in tokens for p in parts)

    # -----------------------------
    # ADJECTIVES / ADVERBS
    # -----------------------------
    if "adjective" in pos or "adverb" in pos or pos == "adjective":
        return word in tokens

    # -----------------------------
    # NON-NOUN fallback
    # -----------------------------
    if not pos.startswith("noun"):
        return word in tokens

    # ============================================================
    # STRICT NOUN MATCHING (TOKEN-BASED)
    # ============================================================

    # singular must be EXACT: [article, noun]
    singular_form = f"{gender} {word}".strip()
    singular_tokens = singular_form.split()    # e.g. ["die", "arbeit"]

    # plural tokens
    plural_tokens = plural.split() if plural else []

    singular_ok = False
    for i in range(len(tokens) - len(singular_tokens) + 1):
        if tokens[i:i + len(singular_tokens)] == singular_tokens:
            singular_ok = True
            break

    # ---- Uncountable nouns (Option B) ----
    if plural == "" or plural == "—":
        return singular_ok

    # ---- Check plural: must match exact token sequence ----
    plural_ok = False
    for i in range(len(tokens) - len(plural_tokens) + 1):
        if tokens[i:i + len(plural_tokens)] == plural_tokens:
            plural_ok = True
            break

    return singular_ok and plural_ok


# ============================================================
# Session State Initialization Helpers
# ============================================================

def init_progress_for_file(filename: str):
    """Ensure we have a progress structure for the given file."""
    if "progress" not in st.session_state:
        st.session_state.progress = {}

    if filename not in st.session_state.progress:
        st.session_state.progress[filename] = {
            "reviewed": set(),   # all words ever seen in Study
            "correct": 0,
            "wrong": 0,
            "mistakes": [],      # list of word strings
        }


if "selected_file" not in st.session_state:
    st.session_state.selected_file = vocab_files[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"   # "Study" or "Review Mistakes"

if "review_queue" not in st.session_state:
    st.session_state.review_queue = []   # active queue of mistake words for this round

if "current" not in st.session_state:
    st.session_state.current = None

# Make sure we have progress data for the initially selected file
init_progress_for_file(st.session_state.selected_file)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Vocabulary Source File")

file_choice = st.sidebar.selectbox(
    "Choose vocab JSON file",
    vocab_files,
    index=vocab_files.index(st.session_state.selected_file),
)

# If user switches files, reset current word and queue
if file_choice != st.session_state.selected_file:
    st.session_state.selected_file = file_choice
    init_progress_for_file(file_choice)
    st.session_state.current = None
    st.session_state.review_queue = []

progress = st.session_state.progress[st.session_state.selected_file]
filtered_vocab = [
    v for v in vocab_all if v["source_file"] == st.session_state.selected_file
]

mode_choice = st.sidebar.selectbox(
    "Mode",
    ["Study", "Review Mistakes"],
    index=["Study", "Review Mistakes"].index(st.session_state.mode),
)

# Handle mode switch
if mode_choice != st.session_state.mode:
    st.session_state.mode = mode_choice
    st.session_state.current = None  # force re-pick

    if mode_choice == "Review Mistakes":
        mistakes = progress["mistakes"]
        if not mistakes:
            # no queue; we'll show "no mistakes" later
            st.session_state.review_queue = []
        else:
            # unique mistakes while preserving order
            unique_mistakes = list(dict.fromkeys(mistakes))
            st.session_state.review_queue = unique_mistakes.copy()

total_items = len(filtered_vocab)
reviewed_count = len(progress["reviewed"])

st.sidebar.markdown(f"""
### Progress
- Total: **{total_items}**
- Reviewed (Study): **{reviewed_count}**
- Correct (first time): **{progress['correct']}**
- Wrong (first time): **{progress['wrong']}**
""")

if st.session_state.mode == "Review Mistakes":
    st.sidebar.markdown(
        f"### Mistakes left this round: **{len(st.session_state.review_queue)}**"
    )


# ============================================================
# Word selection
# ============================================================

def pick_new_word():
    """Pick the next word based on the current mode."""
    if st.session_state.mode == "Study":
        remaining = [
            v for v in filtered_vocab
            if v["word"] not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None
    else:
        # Review Mistakes mode: take first from review_queue
        if not st.session_state.review_queue:
            st.session_state.current = None
        else:
            word_to_review = st.session_state.review_queue[0]
            st.session_state.current = next(
                v for v in filtered_vocab if v["word"] == word_to_review
            )


# If we don't have a current word yet, pick one
if st.session_state.current is None:
    pick_new_word()

entry = st.session_state.current

# ============================================================
# Main UI
# ============================================================

st.title("🎤 German Vocab Trainer")

if entry is None:
    if st.session_state.mode == "Study":
        st.success("You finished all items in this set! 🎉")
    else:
        st.success("No mistakes left to review! 🎉")
    st.stop()

# Prompt
st.markdown(f"""
## Meaning:
**{entry['meaning']}**

🎙️ Say the correct German form:
- **Noun:** say article + singular, then plural  
- **Verb:** say infinitive  
- **Reflexive verb:** say both parts  
- **Adjective/Adverb:** say lemma  
""")

# Audio input (stable key!)
audio_input = st.audio_input(
    "Press to record your pronunciation",
    key="audio_recorder",
)

# Buttons: explicit Check + Next
col1, col2 = st.columns(2)
with col1:
    check_pressed = st.button("Check Answer")
with col2:
    next_pressed = st.button("Next")

# ============================================================
# Check Answer Logic
# ============================================================

if check_pressed and audio_input is not None:
    # Save temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    # Get WAV metadata
    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()

    st.write("⏳ Transcribing...")

    transcript = transcribe_wav_file(audio_path, sample_rate, channels)

    st.markdown(f"### You said:\n**{transcript}**")

    correct = check_answer(entry, transcript)
    first_time = entry["word"] not in progress["reviewed"]

    # ---------- Study mode ----------
    if st.session_state.mode == "Study":
        if correct:
            st.success("Correct! 🎉")

            if first_time:
                progress["correct"] += 1

            # remove from mistakes if it was there
            if entry["word"] in progress["mistakes"]:
                progress["mistakes"].remove(entry["word"])
        else:
            st.error("Not quite.")

            if first_time:
                progress["wrong"] += 1

            if entry["word"] not in progress["mistakes"]:
                progress["mistakes"].append(entry["word"])

    # ---------- Review Mistakes mode ----------
    else:
        if correct:
            st.success("Correct! 🎉")

            # Remove from global mistakes list
            if entry["word"] in progress["mistakes"]:
                progress["mistakes"].remove(entry["word"])

            # And from this round's queue (front)
            if (
                st.session_state.review_queue
                and st.session_state.review_queue[0] == entry["word"]
            ):
                st.session_state.review_queue.pop(0)
        else:
            st.error("Not quite.")

            # Move current word to back of queue
            if (
                st.session_state.review_queue
                and st.session_state.review_queue[0] == entry["word"]
            ):
                w = st.session_state.review_queue.pop(0)
                st.session_state.review_queue.append(w)

    # Mark as "seen" at least once
    progress["reviewed"].add(entry["word"])

    # Reveal the correct solution
    st.markdown(f"""
### Correct German:
- **{entry['word']}**
- POS: **{entry['pos']}**
- Gender: **{entry['gender'] or "—"}**
- Plural: **{entry['plural'] or "—"}**

### Example:
{entry['examples'][0] if entry['examples'] else "_None provided_"}
""")


# ============================================================
# Next button behavior
# ============================================================

if next_pressed:
    pick_new_word()
    st.rerun()
