import streamlit as st
import os

# Import functions from your custom modules
from utils import temp_directory, save_uploaded_file
from Summarizer import (
    transcribe_audio,
    summarize_text,
)

# --- Streamlit App Config ---
st.set_page_config(
    page_title="AI Video Summarizer",
    layout="wide",
    page_icon="🎬"
)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208759.png", width=120)  # 🎬 nice icon
    st.title("🎥 Video Summarizer")
    st.markdown("**Powered by AI 🤖**")
    st.divider()
    st.markdown("### 📌 Features")
    st.markdown("- 📝 Text Summaries\n- 🎧 Audio Summaries\n- 🎬 Condensed Video")
    st.divider()
    st.info("💡 Tip: Longer videos may take more time to process.")
    st.success("⚡ Optimized for ML/AI Projects")

# --- HEADER ---
st.markdown(
    "<h1 style='text-align: center;'>🎥 AI-Powered Video Summarization</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>"
    "Upload your video and instantly generate smart summaries in <b>text, audio,</b> or <b>video</b> formats."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# Setup temp dir
temp_dir = temp_directory()

# --- Upload Section ---
st.subheader("📤 Upload Your Video")
uploaded_file = st.file_uploader(
    "Choose a video file...",
    type=["mp4", "mov", "avi", "mkv"],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

if uploaded_file is not None:
    # Save uploaded file
    temp_video_path = save_uploaded_file(uploaded_file, temp_dir)

    # Show preview in two columns
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(temp_video_path)
    with col2:
        st.info(f"✅ File uploaded: **{uploaded_file.name}**")
        st.metric("File Size", f"{uploaded_file.size / 1e6:.2f} MB")
        st.metric("Format", uploaded_file.type)

    st.divider()

    # --- Summary Options ---
    st.subheader("⚙️ Choose Summary Type")
    summary_choice = st.radio(
        "Pick one:",
        ("📝 Text Summary", "🎧 Audio Summary", "🎬 Video Summary"),
        horizontal=True,
        key="my_radio"
    )

    st.divider()

    # --- Generate Button ---
    if st.button(f"🚀 Generate {summary_choice}", use_container_width=True):
        progress = st.progress(0)
        with st.spinner("⚡ Working hard... This may take a few minutes ⏳"):

            # Update progress in steps (for nicer UX)
            for pct in range(0, 101, 25):
                progress.progress(pct)

            # --- Text Summary ---
            if summary_choice.startswith("📝"):
                transcribed_text = transcribe_audio(temp_video_path)
                if transcribed_text:
                    summary_text = summarize_text(transcribed_text)
                    if summary_text:
                        st.success("🎉 Summary Generated Successfully!")
                        
                        tab1, tab2 = st.tabs(["✨ Final Summary", "📜 Full Transcription"])
                        with tab1:
                            st.write(summary_text)
                        with tab2:
                            st.write(transcribed_text)

