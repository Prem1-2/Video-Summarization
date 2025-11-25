# Video-Summarization Web Application

A Streamlit-based AI application that summarizes long videos into short video highlights, text summaries, and audio summaries. The application extracts transcript from YouTube videos, performs summarization using transformer models, and generates both video and audio summaries.

🚀 Features

🎬 Video-to-Video Summarization

. Automatically detects key scenes using PySceneDetect

. Generates a short highlight video with visuals

📝 Text Summary Generation

. Extracts transcript using Whisper model

. Produces concise text summaries

🔉 Audio Summary

. Converts text summary into speech using Elevenlabs

🔗 YouTube URL Support

. Paste any video link and get summaries instantly

💬 User Feedback System

. Allows users to submit comments & reviews

. Displays all feedback on the site

🧠 Technologies Used

Area                  Technology 
Frontend / UI         Streamlit 
Model                 Whisper 
Video Processing      FFMPEG, PySceneDetect 
Audio Processing      Elevenlabs 
Data Handling         Pandas 
Deployment            Streamlit Cloud / Local
