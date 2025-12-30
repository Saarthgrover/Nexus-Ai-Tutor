import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from streamlit_mic_recorder import speech_to_text
import base64
import sys
import importlib
from PIL import Image
import io
try:
    _mod = importlib.import_module("streamlit.runtime.scriptrunner.script_run_context")
    get_script_run_ctx = getattr(_mod, "get_script_run_ctx", lambda: None)
except Exception:
    def get_script_run_ctx():
        return None
import time
import shutil
import re


def main() -> None:
    # --- 1. SETUP & CONFIGURATION ---
    # Prefer Streamlit secrets, then environment variable
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    client = None
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
        except Exception:
            client = None

    st.set_page_config(page_title="Nexus Pro: AI Tutor", page_icon="🎓", layout="wide")

    if client is None:
        st.warning("OpenAI API key not found. Features requiring OpenAI will be disabled.\nSet the OPENAI_API_KEY environment variable or add OPENAI_API_KEY to .streamlit/secrets.toml to enable them.")

    # Runtime configuration: model and cost controls
    model_name = None
    try:
        model_name = st.secrets.get("MODEL_NAME")
    except Exception:
        model_name = None
    if not model_name:
        model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    max_history = 3
    try:
        max_history = int(st.secrets.get("MAX_HISTORY", 3))
    except Exception:
        try:
            max_history = int(os.getenv("MAX_HISTORY", 3))
        except Exception:
            max_history = 3

    max_tokens = 256
    try:
        max_tokens = int(st.secrets.get("MAX_TOKENS", 256))
    except Exception:
        try:
            max_tokens = int(os.getenv("MAX_TOKENS", 256))
        except Exception:
            max_tokens = 256

    # --- 2. AUDIO HELPER ---
    def speak_text(text: str) -> str:
        """Converts text to speech using OpenAI TTS and returns a base64 HTML audio tag."""
        if client is None:
            return ""
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

    # --- Image helpers ---
    def try_ocr_image(image_file) -> tuple[str | None, str | None]:
        """Attempt OCR using pytesseract. Returns (text, error_message)."""
        try:
            import pytesseract
        except Exception:
            return None, "pytesseract not installed or Tesseract engine missing"
        try:
            img = Image.open(image_file)
            text = pytesseract.image_to_string(img)
            return text, None
        except Exception as e:
            return None, str(e)

    def summarize_text_via_openai(text: str) -> str:
        if client is None:
            return "OpenAI client not configured. Cannot summarize automatically."
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes text extracted from images into concise summaries and action items."},
                {"role": "user", "content": f"Summarize and interpret the following extracted text from an image. Provide a concise summary, key points, and action items:\n\n{text[:3000]}"},
            ]
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=min(512, max_tokens),
                temperature=0.2,
            )
            # Parse response conservatively
            content = getattr(resp.choices[0], "message", None)
            if content is None:
                content = getattr(resp.choices[0], "text", "")
            if isinstance(content, dict):
                return content.get("content", "").strip()
            return str(content)
        except Exception as e:
            return f"OpenAI error during summarization: {e}"

    # --- 4. STATE MANAGEMENT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_context" not in st.session_state:
        st.session_state.file_context = ""
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = ""
    if "image_context" not in st.session_state:
        st.session_state.image_context = ""
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "image_path" not in st.session_state:
        st.session_state.image_path = None
    if "file_summary" not in st.session_state:
        st.session_state.file_summary = ""

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

        # --- Summarize uploaded notes ---
        if st.session_state.file_context:
            st.markdown("---")
            st.markdown("### Summarize Uploaded Notes")
            summary_mode = st.selectbox("Summary style:", ["Short (3-5 bullets)", "Detailed (paragraph)", "Bulleted list"], index=0)
            if st.button("Summarize Notes"):
                notes_text = st.session_state.file_context
                if not notes_text or notes_text.strip() == "":
                    st.warning("No notes found to summarize")
                else:
                    if client is None:
                        # simple local fallback: extract sentences
                        sents = re.split(r'(?<=[.!?])\s+', notes_text)
                        if summary_mode.startswith("Short"):
                            picked = sents[:5]
                            summary = "\n\n".join(picked)
                        elif summary_mode.startswith("Detailed"):
                            summary = notes_text[:2000]
                            if len(notes_text) > 2000:
                                summary += "..."
                        else:
                            picked = sents[:10]
                            summary = "\n".join([f"- {s.strip()}" for s in picked if s.strip()])
                        st.session_state.file_summary = summary
                        st.info("Summary generated locally (limited). For better summaries, set OPENAI_API_KEY.")
                    else:
                        # Use OpenAI to summarize
                        try:
                            user_prompt = (
                                f"Please produce a {summary_mode} summary of the following student notes. Return the result in markdown.\n\n{notes_text}"
                            )
                            messages = [
                                {"role": "system", "content": "You are a helpful assistant that summarizes study notes into clear, concise summaries."},
                                {"role": "user", "content": user_prompt},
                            ]
                            resp = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                max_tokens=800,
                                temperature=0.2,
                            )
                            # extract content (non-stream)
                            content = getattr(resp.choices[0], "message", None)
                            if content is None:
                                # older shape
                                content = getattr(resp.choices[0], "text", "")
                            summary_text = content.get("content") if isinstance(content, dict) else content
                            if not summary_text:
                                # fallback: try attribute
                                summary_text = str(content)
                            st.session_state.file_summary = summary_text
                            st.success("Notes summarized using OpenAI")
                        except Exception as e:
                            st.error("Failed to summarize via OpenAI")
                            st.exception(e)

            if st.session_state.file_summary:
                with st.expander("View last summary", expanded=False):
                    st.markdown(st.session_state.file_summary)
                    try:
                        st.download_button("Download summary", data=st.session_state.file_summary.encode("utf-8"), file_name="notes_summary.md")
                    except Exception:
                        pass

        uploaded_image = st.file_uploader(
            "Upload Image / Screenshot",
            type=["png", "jpg", "jpeg", "bmp", "gif", "webp"],
        )

        if uploaded_image and not st.session_state.image_bytes:
            try:
                image_bytes = uploaded_image.read()
                st.session_state.image_bytes = image_bytes
                image = Image.open(io.BytesIO(image_bytes))
                st.session_state.image_context = f"{uploaded_image.name}"
                # save a persistent copy to ./uploads with a timestamped filename
                try:
                    uploads_dir = os.path.join(os.getcwd(), "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    safe_name = f"{int(time.time())}_{uploaded_image.name}"
                    save_path = os.path.join(uploads_dir, safe_name)
                    with open(save_path, "wb") as f:
                        f.write(image_bytes)
                    st.session_state.image_path = save_path
                except Exception:
                    st.session_state.image_path = None

                st.image(image, caption="Uploaded image preview", use_column_width=True)
                if st.session_state.image_path:
                    st.success(f"Image uploaded and saved to: {st.session_state.image_path}")
                else:
                    st.success("Image uploaded (could not save to disk).")
                # Image analysis tools
                with st.expander("Image Tools / Analyze", expanded=False):
                    ocr_btn = st.button("Extract text (OCR)")
                    if ocr_btn:
                        text, err = try_ocr_image(io.BytesIO(image_bytes))
                        if err:
                            st.error(f"OCR failed: {err}")
                        else:
                            if not text or text.strip() == "":
                                st.info("No text detected in image by OCR.")
                            else:
                                st.success("Extracted text from image (OCR)")
                                st.text_area("Extracted text", value=text, height=200)
                                # store into image_context for later summarization
                                st.session_state.image_context = text

                    summarize_btn = st.button("Summarize image (OpenAI)")
                    if summarize_btn:
                        # prefer OCR text if available
                        ocr_text = st.session_state.get("image_context") or None
                        if not ocr_text:
                            # try OCR on the fly
                            ocr_text, err = try_ocr_image(io.BytesIO(image_bytes))
                            if err:
                                st.warning("OCR not available; attempting OpenAI visual analysis (may fail depending on API).")
                        if ocr_text:
                            summary = summarize_text_via_openai(ocr_text)
                            st.markdown("**Image summary (from extracted text):**")
                            st.write(summary)
                        else:
                            # fallback: attempt OpenAI vision/description if supported
                            if client is None:
                                st.error("OpenAI client not configured — cannot perform image summarization.")
                            else:
                                try:
                                    # Try to send the image bytes embedded as base64 to the model via a text prompt.
                                    b64 = base64.b64encode(image_bytes).decode()
                                    user_msg = (
                                        "You will be provided an image encoded as base64. "
                                        "Provide a concise description and summary of the image, list key elements and any actionable items. "
                                        "If the base64 cannot be decoded, explain that you couldn't analyze the image.\n\n"
                                        f"Image (base64): data:image/png;base64,{b64[:200]}..."
                                    )
                                    resp = client.chat.completions.create(
                                        model=model_name,
                                        messages=[
                                            {"role": "system", "content": "You are an assistant that can interpret images provided as base64 strings and summarise their content."},
                                            {"role": "user", "content": user_msg},
                                        ],
                                        max_tokens=min(512, max_tokens),
                                        temperature=0.2,
                                    )
                                    content = getattr(resp.choices[0], "message", None)
                                    if content is None:
                                        content = getattr(resp.choices[0], "text", "")
                                    if isinstance(content, dict):
                                        analysis = content.get("content", "").strip()
                                    else:
                                        analysis = str(content)
                                    st.markdown("**Image summary (OpenAI analysis):**")
                                    st.write(analysis)
                                except Exception as e:
                                    st.error("OpenAI visual analysis failed")
                                    st.exception(e)
            except Exception as e:
                st.error("Failed to process uploaded image.")
                st.exception(e)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_quiz = ""
            st.session_state.file_context = ""
            st.session_state.image_context = ""
            st.session_state.image_bytes = None
            st.session_state.image_path = None
            st.experimental_rerun()

        # --- Requirements manager ---
        st.markdown("---")
        st.markdown("### Manage requirements.txt")
        req_path = os.path.join(os.getcwd(), "requirements.txt")

        def read_requirements():
            if not os.path.exists(req_path):
                return []
            with open(req_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            # filter out blanks but keep comments if you want; here we keep non-empty lines
            return [ln for ln in lines if ln != ""]

        def write_requirements(lines: list[str]):
            # backup existing file and return backup path (or None)
            bak_path = None
            try:
                if os.path.exists(req_path):
                    bak_path = f"{req_path}.bak_{int(time.time())}"
                    shutil.copy(req_path, bak_path)
            except Exception:
                bak_path = None
            # write new file
            with open(req_path, "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(f"{ln}\n")
            return bak_path

        try:
            current_reqs = read_requirements()
        except Exception:
            current_reqs = []

        if current_reqs:
            st.write("Current `requirements.txt`:")
            st.write("\n".join(current_reqs))
        else:
            st.info("No entries found in requirements.txt")

        def list_backups():
            base = f"{req_path}.bak_"
            dirn = os.path.dirname(req_path) or "."
            files = []
            try:
                for fn in os.listdir(dirn):
                    if fn.startswith(os.path.basename(base)):
                        files.append(os.path.join(dirn, fn))
            except Exception:
                files = []
            return sorted(files, reverse=True)

        backups = list_backups()
        if backups:
            st.markdown("**Backups:**")
            # show most recent 5 backups with download buttons and human-readable timestamps
            for i, b in enumerate(backups[:5]):
                base = os.path.basename(b)
                ts_part = base.split(".bak_")[-1] if ".bak_" in base else ""
                try:
                    ts = int(ts_part)
                    readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                except Exception:
                    readable = base
                st.write(f"{readable} — {base}")
                try:
                    with open(b, "rb") as bf:
                        data = bf.read()
                    st.download_button(f"Download {base}", data=data, file_name=base, key=f"dl_{i}")
                except Exception:
                    st.write("(failed to read backup for download)")

            # select a backup to restore
            def backup_label(p: str) -> str:
                bn = os.path.basename(p)
                if ".bak_" in bn:
                    ts_str = bn.split(".bak_")[-1]
                    try:
                        return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(ts_str)))} — {bn}"
                    except Exception:
                        return bn
                return bn

            sel = st.selectbox("Select a backup to restore", options=backups, format_func=backup_label)
            if sel:
                if st.button("Restore selected backup"):
                    # backup current requirements first
                    try:
                        if os.path.exists(req_path):
                            restore_bak = f"{req_path}.bak_restore_{int(time.time())}"
                            shutil.copy(req_path, restore_bak)
                        else:
                            restore_bak = None
                    except Exception:
                        restore_bak = None

                    try:
                        shutil.copy(sel, req_path)
                        if restore_bak:
                            st.success(f"Restored {sel} → requirements.txt (previous file backed up: {restore_bak})")
                        else:
                            st.success(f"Restored {sel} → requirements.txt")
                    except Exception as e:
                        st.error("Failed to restore backup")
                        st.exception(e)
                    st.experimental_rerun()
        else:
            st.write("No backups found")

        # add a single package
        new_pkg = st.text_input("Add package (e.g. package==1.2.3)")
        if st.button("Add package"):
            np = new_pkg.strip()
            if np:
                if np in current_reqs:
                    st.warning("Package already present")
                else:
                    current_reqs.append(np)
                    bak = write_requirements(current_reqs)
                    if bak:
                        st.info(f"Backup created: {bak}")
                    st.success(f"Added: {np}")
                    st.experimental_rerun()
            else:
                st.warning("Enter a package string to add")

        # upload a requirements file to merge
        uploaded_reqs = st.file_uploader("Upload requirements file to merge", type=["txt", "pip", "requirements"], key="req_upload")
        if uploaded_reqs:
            try:
                uploaded_text = uploaded_reqs.read().decode(errors="ignore").splitlines()
                uploaded_clean = [ln.strip() for ln in uploaded_text if ln.strip() and not ln.strip().startswith("#")]
            except Exception:
                uploaded_clean = []

            if uploaded_clean:
                if st.button("Merge uploaded requirements"):
                    merged = current_reqs.copy()
                    for ln in uploaded_clean:
                        if ln not in merged:
                            merged.append(ln)
                    bak = write_requirements(merged)
                    if bak:
                        st.info(f"Backup created: {bak}")
                    st.success("Merged uploaded requirements into requirements.txt")
                    st.experimental_rerun()

                # two-step overwrite: set pending overwrite then confirm
                if st.button("Overwrite requirements.txt with uploaded file"):
                    st.session_state.pending_overwrite = uploaded_clean
                    st.session_state.pending_overwrite_name = getattr(uploaded_reqs, "name", "uploaded_requirements")

                if st.session_state.get("pending_overwrite"):
                    st.warning(f"You are about to overwrite `requirements.txt` with {st.session_state.get('pending_overwrite_name')}. This will create a backup.")
                    if st.button("Confirm overwrite requirements.txt"):
                        bak = write_requirements(st.session_state.pending_overwrite)
                        if bak:
                            st.success(f"Overwrote requirements.txt (backup: {bak})")
                        else:
                            st.success("Overwrote requirements.txt (no backup created)")
                        st.session_state.pending_overwrite = None
                        st.session_state.pending_overwrite_name = None
                        st.experimental_rerun()
            else:
                st.warning("Uploaded file contained no valid lines to merge")

        # delete selected packages
        to_remove = st.multiselect("Select entries to remove", options=current_reqs)
        if to_remove:
            if st.button("Delete selected entries"):
                new_list = [ln for ln in current_reqs if ln not in to_remove]
                bak = write_requirements(new_list)
                if bak:
                    st.info(f"Backup created: {bak}")
                st.success("Removed selected entries from requirements.txt")
                st.experimental_rerun()

    # Personality Prompt
    base_prompt = f"You are Nexus, an AI Tutor specialized in {subject}."
    system_prompt = base_prompt
    if st.session_state.file_context:
        system_prompt += f"\n\nContext from student notes: {st.session_state.file_context[:3000]}"
    if st.session_state.image_context:
        system_prompt += f"\n\nContext from uploaded image: {st.session_state.image_context}"

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
                if client is None:
                    full_response = "OpenAI API key not configured. Please set OPENAI_API_KEY to enable responses."
                    message_placeholder.markdown(full_response)
                else:
                    # Reduce history to last `max_history` messages to save tokens
                    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-max_history:]

                    # Retry with exponential backoff for transient rate/quota errors
                    attempts = 0
                    max_attempts = 3
                    backoff = 1
                    succeeded = False
                    while attempts < max_attempts and not succeeded:
                        try:
                            stream = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                stream=True,
                                max_tokens=max_tokens,
                                temperature=0.7,
                            )
                            for chunk in stream:
                                delta = getattr(chunk.choices[0], "delta", None)
                                if delta and getattr(delta, "content", None):
                                    full_response += delta.content
                                    message_placeholder.markdown(full_response + "▌")

                            message_placeholder.markdown(full_response)
                            succeeded = True
                        except Exception as e:
                            attempts += 1
                            # If final attempt, surface error; otherwise back off and retry
                            if attempts >= max_attempts:
                                err_msg = f"OpenAI API error: {e}"
                                full_response = "Sorry — I'm temporarily unable to generate a response. Please try again later."
                                message_placeholder.markdown(full_response)
                                st.error("OpenAI request failed — check your API key, billing, and quota. See details below.")
                                st.exception(e)
                                break
                            time.sleep(backoff)
                            backoff *= 2

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


if __name__ == "__main__":
    # Allow Streamlit to run the app even if ScriptRunContext isn't detected
    # This avoids false positives where the context import/path can't be resolved
    main()

