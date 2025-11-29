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
# POS / noun matching logic
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
    if "adjective" in pos or "adverb" in pos:
        return word in tokens

    # -----------------------------
    # NON-NOUN fallback
    # -----------------------------
    if not pos.startswith("noun"):
        return word in tokens

    # ============================================================
    # STRICT NOUN MATCHING
    # ============================================================

    singular_form = f"{gender} {word}".strip()
    singular_tokens = singular_form.split()
    plural_tokens = plural.split() if plural else []
    toks = tokens

    # Check singular
    singular_ok = False
    for i in range(len(toks) - len(singular_tokens) + 1):
        if toks[i:i+len(singular_tokens)] == singular_tokens:
            singular_ok = True
            break

    # Uncountable nouns
    if plural == "" or plural == "—":
        return singular_ok

    # Check plural
    plural_ok = False
    for i in range(len(toks) - len(plural_tokens) + 1):
        if toks[i:i+len(plural_tokens)] == plural_tokens:
            plural_ok = True
            break

    return singular_ok and plural_ok


# ============================================================
# Session state initialization
# ============================================================

if "selected_file" not in st.session_state:
    st.session_state.selected_file = vocab_files[0]

if "mode" not in st.session_state:
    st.session_state.mode = "Study"

if "easy_mode" not in st.session_state:
    st.session_state.easy_mode = False

if "selected_set" not in st.session_state:
    st.session_state.selected_set = "All sets"

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

if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

if "_force_refresh" not in st.session_state:
    st.session_state._force_refresh = False


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Vocabulary Source File")

file_choice = st.sidebar.selectbox(
    "Choose vocab JSON file",
    vocab_files,
    key="file_selector"
)

# Change vocab file
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
    
    # Reset current word and trigger rerun
    if "current" in st.session_state:
        del st.session_state["current"]
    st.rerun()

filtered_vocab = [
    v for v in vocab_all
    if v["source_file"] == st.session_state.selected_file
]

# Get unique sets from current file
available_sets = sorted(list(set(
    v.get("set", "No set") for v in filtered_vocab
)))

# Add "All sets" option
set_options = ["All sets"] + available_sets

if "selected_set" not in st.session_state:
    st.session_state.selected_set = "All sets"

set_choice = st.sidebar.selectbox(
    "Filter by set",
    set_options,
    index=set_options.index(st.session_state.selected_set) if st.session_state.selected_set in set_options else 0,
    key="set_selector"
)

# Handle set change
if set_choice != st.session_state.selected_set:
    st.session_state.selected_set = set_choice
    if "current" in st.session_state:
        del st.session_state["current"]
    st.rerun()

# Filter by set if not "All sets"
if st.session_state.selected_set != "All sets":
    filtered_vocab = [
        v for v in filtered_vocab
        if v.get("set", "No set") == st.session_state.selected_set
    ]

mode_choice = st.sidebar.selectbox(
    "Mode",
    ["Study", "Review Mistakes"],
    key="mode_selector"
)

# Easy mode toggle
easy_mode = st.sidebar.checkbox(
    "Easy mode (show German word)",
    value=st.session_state.easy_mode,
    key="easy_mode_checkbox"
)
st.session_state.easy_mode = easy_mode

# Mode change handling
if mode_choice != st.session_state.get("mode"):
    st.session_state.mode = mode_choice

    if mode_choice == "Review Mistakes":
        if not progress["mistakes"]:
            st.success("No mistakes to review! 🎉")
            st.stop()

        # Unique list of mistakes
        st.session_state.review_queue = list(dict.fromkeys(progress["mistakes"]))

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
    st.sidebar.markdown(
        f"### Mistakes left: **{len(st.session_state.review_queue)}**"
    )


# ============================================================
# Word selection
# ============================================================

def pick_new_word():
    if st.session_state.mode == "Study":
        remaining = [
            v for v in filtered_vocab
            if v["word"] not in progress["reviewed"]
        ]
        st.session_state.current = random.choice(remaining) if remaining else None

    else:  # Review mode
        if not st.session_state.review_queue:
            st.session_state.current = None
        else:
            word_to_review = st.session_state.review_queue[0]
            st.session_state.current = next(
                v for v in filtered_vocab
                if v["word"] == word_to_review
            )

# Handle mode switch
if st.session_state.get("_pending_mode_switch"):
    del st.session_state["_pending_mode_switch"]
    pick_new_word()
    st.rerun()

if "current" not in st.session_state:
    pick_new_word()

entry = st.session_state.current

if entry is None:
    if mode_choice == "Study":
        st.success("You finished all items in this set! 🎉")
    else:
        st.success("No mistakes left! 🎉")
    st.stop()


# ============================================================
# Prompt
# ============================================================

if st.session_state.easy_mode:
    # Easy mode: show German word
    st.markdown(f"""
### Say this German word correctly:  
## {entry['word']}
- **Gender:** {entry['gender'] or "—"}
- **Plural:** {entry['plural'] or "—"}
- **Meaning:** {entry['meaning']}
""")
else:
    # Normal mode: show English meaning
    st.markdown(f"""
### Say the correct German for:  
## {entry['meaning']}
""")

st.markdown("""
- **Noun:** article + singular, then plural  
- **Verb:** infinitive  
- **Reflexive verb:** both parts  
- **Adjective/Adverb:** lemma  
""")


# ============================================================
# Audio Input
# ============================================================

# Clear old audio to prevent widget reuse issues
st.session_state.pop("audio_input", None)

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
    
    # Store first_time status for later use
    st.session_state.was_first_time = first_time

    # ============================================================
    # UPDATE PROGRESS + MESSAGES
    # ============================================================

    if correct:
        st.success("Correct! 🎉")

        if first_time:
            progress["correct"] += 1

        if entry["word"] in progress["mistakes"]:
            progress["mistakes"].remove(entry["word"])

        if st.session_state.mode == "Review Mistakes":
            if entry["word"] in st.session_state.review_queue:
                st.session_state.review_queue.remove(entry["word"])

    else:
        # default wrong message
        message = "Not quite."

        if first_time:
            progress["wrong"] += 1

        if entry["word"] not in progress["mistakes"]:
            progress["mistakes"].append(entry["word"])

        # review-mode rotation
        if st.session_state.mode == "Review Mistakes":

            current_word = entry["word"]
            if current_word in st.session_state.review_queue:
                idx = st.session_state.review_queue.index(current_word)
                w = st.session_state.review_queue.pop(idx)
                st.session_state.review_queue.append(w)

            # ONE-ITEM-LEFT CASE → modify message
            if len(st.session_state.review_queue) == 1:
                message = "Not quite — try again."

        st.error(message)
        
        # Show expected vs heard
        expected_parts = []
        if entry["pos"].startswith("noun"):
            expected_parts.append(f"{entry['gender']} {entry['word']}")
            if entry['plural'] and entry['plural'] != "—":
                expected_parts.append(entry['plural'])
        elif entry["pos"] in ["verb", "reflexive verb"]:
            expected_parts.append(entry["word"])
        else:
            expected_parts.append(entry["word"])
        
        expected_str = ", ".join(expected_parts)
        st.markdown(f"**Expected:** {expected_str}")
        st.markdown(f"**You said:** {transcript}")

        # one-item-left reset (removes stale message)
        if st.session_state.mode == "Review Mistakes":
            if len(st.session_state.review_queue) == 1:
                st.rerun()

    progress["reviewed"].add(entry["word"])
    
    # Mark as correct button (only show when wrong)
    if not correct:
        if st.button("Mark as Correct (ASR error)", key=f"override_{entry['word']}"):
            # Undo the wrong marking using stored first_time status
            if st.session_state.get("was_first_time", False):
                progress["wrong"] -= 1
                progress["correct"] += 1
            
            # Remove from mistakes
            if entry["word"] in progress["mistakes"]:
                progress["mistakes"].remove(entry["word"])
            
            # Remove from review queue if in review mode
            if st.session_state.mode == "Review Mistakes":
                if entry["word"] in st.session_state.review_queue:
                    st.session_state.review_queue.remove(entry["word"])
            
            # Clean up
            if "was_first_time" in st.session_state:
                del st.session_state["was_first_time"]
            
            # Pick next word and advance
            pick_new_word()
            st.rerun()

    # ============================================================
    # Reveal correct answer
    # ============================================================

    example = entry['examples'][0] if entry['examples'] else "_None provided_"

    st.markdown(f"""
### Correct German:
- **{entry['word']}**
- POS: **{entry['pos']}**
- Gender: **{entry['gender'] or "—"}**
- Plural: **{entry['plural'] or "—"}**

### Example:
{example}
""")

    if st.button("Next"):
        pick_new_word()
        st.rerun()
