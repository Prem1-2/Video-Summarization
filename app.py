import streamlit as st
import os
import tempfile    # Import tempfile
import atexit      # Import atexit for cleanup
import shutil      # Import shutil to remove directories
import yt_dlp
import pandas as pd

# Import functions from your custom modules
from summary import (
    transcribe_audio,
    summarize_text,
    text_to_audio,
    detect_scenes_fast,
    select_key_scenes,
    create_video_summary_ffmpeg

)

# --- Streamlit App Config ---
st.set_page_config(
    page_title="AI Video Summarizer",
    layout="wide",
    page_icon="🎬"
)

# NEW: Setup Session-Wide Temporary Directory
@st.cache_resource
def get_temp_dir():
    """
    Creates a single temporary directory for the entire app session.
    This directory will be cleaned up when the app server stops.
    """
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")
    
    # Register the cleanup function to run when the app exits
    atexit.register(cleanup_dir, temp_dir=temp_dir)
    return temp_dir

def cleanup_dir(temp_dir):
    """
    Recursively deletes the temporary directory.
    """
    if os.path.exists(temp_dir):
        print(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

# Get or create the session's temp directory
temp_dir = get_temp_dir()

# Re-add your save function (no longer in utils.py) 
def save_uploaded_file(uploaded_file, temp_dir):
    """Saves uploaded file to the session's temp directory."""
    if uploaded_file:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None



def download_video_from_url(url, save_dir):
    """
    Downloads the best audio-only stream from a URL to the specified directory
    using yt-dlp. Returns the file path of the downloaded audio.
    """
    try:
        st.info(f"Accessing link with yt-dlp...")
        
        file_path_template = os.path.join(save_dir, '%(title)s.%(ext)s')

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': file_path_template,
            'quiet': True,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        st.success(f"Download complete: {info.get('title')}")
        return filename
        
    except Exception as e:
        st.error(f"Error downloading with yt-dlp: {e}")
        return None
    

def download_full_video(url, save_dir):
    try:
        st.info("Downloading full video for visual summary...")
        file_template = os.path.join(save_dir, "%(title)s.%(ext)s")

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': file_template,
            'quiet': True,
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        st.success("Full video downloaded.")
        return filename

    except Exception as e:
        st.error(f"Error downloading full video: {e}")
        return None



with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208759.png", width=120)
    st.title("🎥 Video Summarizer")
    st.markdown("**Powered by AI 🤖**")
    st.divider()
    st.markdown("### 📌 Features")
    st.markdown("- 📝 Text Summaries\n- 🎧 Audio Summaries\n- 🎬 Condensed Video")
    st.divider()
    st.info("💡 Tip: Longer videos may take more time to process.")
    st.success("⚡ Optimized for ML/AI Projects")


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


st.subheader("🔗 Provide Your Video Link")
video_url = st.text_input(
    "Paste a video URL...",
    help="Currently supports YouTube URLs."
)

if video_url:
    st.divider()

    st.subheader("⚙️ Choose Summary Type")
    summary_choice = st.radio(
        "Pick one:",
        ("📝 Text Summary", "🎧 Audio Summary", "🎬 Video Summary"),
        horizontal=True,
        key="my_radio"
    )

    st.divider()

    # Generate Button 
    if st.button(f"🚀 Generate {summary_choice}", use_container_width=True):
        progress = st.progress(0, "Starting...")
        with st.spinner("⚡ Working hard... This may take a few minutes ⏳"):

            # Download Step 
            progress.progress(10, "Downloading video audio...")
            temp_video_path = download_video_from_url(video_url, temp_dir)

            if temp_video_path is None:
                st.stop() 
            
            st.divider()

            # Text Summary 
            if summary_choice.startswith("📝"):
                progress.progress(25, "1/2 - Transcribing audio...")
                
                # Now captures language 
                transcribed_text, detected_lang = transcribe_audio(temp_video_path, temp_dir)
                
                if transcribed_text:

                    progress.progress(75, "3/3 - Processing summary...") 
            
                    summary_text = summarize_text(transcribed_text, source_lang=detected_lang)
                    
                    if summary_text:
                        progress.progress(100, "Done!")
                        st.success("🎉 Summary Generated Successfully!")
                        
                        tab1, tab2 = st.tabs(["✨ Final Summary", "📜 Full Transcription"])
                        with tab1:
                            st.write(summary_text)
                        with tab2:
                            st.write(transcribed_text)

            # --- Audio Summary Logic ---
            elif summary_choice.startswith("🎧"):
                progress.progress(25, "1/3 - Transcribing audio...")
                
                transcribed_text, detected_lang = transcribe_audio(temp_video_path, temp_dir)
                
                if transcribed_text:
                    
                    progress.progress(60, "2/3 - Processing summary...")
                    
                    summary_text = summarize_text(transcribed_text, source_lang=detected_lang)
                    
                    if summary_text:
                        progress.progress(80, "3/3 - Converting summary to audio...")
                        temp_audio_path = os.path.join(temp_dir, "summary.mp3")
                        
                        if text_to_audio(summary_text, temp_audio_path):
                            progress.progress(100, "Done!")
                            st.success("Audio Summary Generated Successfully! 🎉")
                            
                            st.subheader("🔊 Audio Summary")
                            st.audio(temp_audio_path)
                            
                            with st.expander("Show Summary Text"):
                                st.write(summary_text)
                        else:
                            st.error("Could not generate audio file. Check logs for details.")

            # --- VIDEO SUMMARY OPTION ---
            elif summary_choice.startswith("🎬"):
                progress.progress(10, "Downloading full video...")
                full_video_path = download_full_video(video_url, temp_dir)

                if full_video_path is None:
                    st.error("🚨 Could not download full video.")
                    st.stop()

                progress.progress(40, "Detecting scenes...")
                scenes = detect_scenes_fast(full_video_path)

                if not scenes:
                    st.error("⚠ No scenes detected. Try increasing threshold.")
                    st.stop()

                progress.progress(60, "Selecting best scenes...")
                key_scenes = select_key_scenes(scenes, max_scenes=5)

                progress.progress(80, "Creating summary video...")
                summary_video_path = os.path.join(temp_dir, "video_summary.mp4")
                final_path = create_video_summary_ffmpeg(full_video_path, key_scenes, summary_video_path)

                progress.progress(100, "Done 🎉")
                st.success("🎬 Video Summary Created Successfully!")
                st.video(final_path)



###FeedBacks#######

COMMENTS_FILE = "comments.csv"

# Create comments file if it doesn't exist
if not os.path.exists(COMMENTS_FILE):
    df = pd.DataFrame(columns=["Name", "Comment"])
    df.to_csv(COMMENTS_FILE, index=False)

st.write("## 💬 User Feedback & Comments")

with st.form("comment_form"):
    name = st.text_input("Your Name")
    comment = st.text_area("Your Comment about this website")
    submit_comment = st.form_submit_button("Submit")

if submit_comment:
    if name.strip() == "" or comment.strip() == "":
        st.warning("⚠ Please fill all fields before submitting.")
    else:
        df = pd.read_csv(COMMENTS_FILE)
        new_row = pd.DataFrame({"Name": [name], "Comment": [comment]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(COMMENTS_FILE, index=False)
        st.success("🎉 Thank you! Your comment has been submitted.")

st.write("---")
st.write("### ⭐ User Comments")

# Display all comments
comments_df = pd.read_csv(COMMENTS_FILE)

for index, row in comments_df.iterrows():
    st.info(f"**{row['Name']}** says:\n\n{row['Comment']}")

            