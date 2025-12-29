import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from streamlit_mic_recorder import speech_to_text
import base64

# --- 1. SETUP & CONFIGURATION ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Nexus Pro: AI Tutor", page_icon="🎓", layout="wide")

# --- 2. AUDIO HELPER ---
def speak_text(text: str) -> str:
    """Converts text to speech using OpenAI TTS and returns a base64 HTML audio tag."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        audio_data = response.content
        b64 = base64.b64encode(audio_data).decode()
        return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}"></audio>'
    except Exception:
        return ""

# --- 3. OTHER HELPERS ---
def get_pdf_text(pdf_file) -> str:
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def convert_chat_to_text(messages) -> str:
    chat_str = "--- NEXUS AI STUDY SESSION ---\n\n"
    for msg in messages:
        role = "STUDENT" if msg.get("role") == "user" else "AI TUTOR"
        chat_str += f"[{role}]:\n{msg.get('content')}\n\n"
    return chat_str

# --- 4. STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = ""

# --- 5. SIDEBAR & LOGIC ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    subject = st.selectbox(
        "Select Subject:", ["General", "Math 🧮", "Science 🧬", "History 🏛️", "English 📚"]
    )
    uploaded_file = st.file_uploader("Upload Notes (PDF)", type=["pdf"])

    if uploaded_file and not st.session_state.file_context:
        st.session_state.file_context = get_pdf_text(uploaded_file)
        st.success("File Processed!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_quiz = ""
        st.session_state.file_context = ""
        st.experimental_rerun()

# Personality Prompt
base_prompt = f"You are Nexus, an AI Tutor specialized in {subject}."
if st.session_state.file_context:
    system_prompt = f"{base_prompt}\n\nContext from student notes: {st.session_state.file_context[:3000]}"
else:
    system_prompt = base_prompt

# --- 6. MAIN INTERFACE ---
st.title(f"🎓 Nexus Pro: {subject} Mode")

# Voice Input (STT)
st.write("### 🎙️ Voice Command")
voice_text = speech_to_text(
    language="en",
    start_prompt="Click to Speak 🎤",
    stop_prompt="Stop Recording 🛑",
    just_once=True,
    key="STT",
)

tab1, tab2 = st.tabs(["💬 Chat Tutor", "📝 Quiz Zone"])

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message.get("role", "user")):
            st.markdown(message.get("content", ""))

    chat_input = st.chat_input("Ask a question...")
    final_query = voice_text if voice_text else chat_input

    if final_query:
        st.session_state.messages.append({"role": "user", "content": final_query})
        with st.chat_message("user"):
            st.markdown(final_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the text response
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages[-5:],
                stream=True,
            )
            for chunk in stream:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    full_response += delta.content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Generate and play Audio (TTS)
            with st.spinner("Nexus is speaking..."):
                audio_html = speak_text(full_response)
                if audio_html:
                    st.components.v1.html(audio_html, height=60)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    st.header("Exam Prep")
    if st.session_state.current_quiz:
        st.markdown(st.session_state.current_quiz)
    else:
        st.markdown("No quiz generated yet. Use the Chat Tutor to request practice questions.")

