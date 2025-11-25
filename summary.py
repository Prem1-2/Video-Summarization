import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from faster_whisper import WhisperModel
import torch 
from transformers import pipeline 
import sys 
from elevenlabs import save
from groq import Groq
from elevenlabs.client import ElevenLabs
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image
import numpy as np
import subprocess
import re
from dotenv import load_dotenv


load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")


# ---- Load model only once ----
@st.cache_resource
def load_whisper_model():
    download_path = os.path.join(os.getcwd(), "models_cache")
    os.makedirs(download_path, exist_ok=True)

    print("🔄 Loading Whisper Model... (only once)")
    model = WhisperModel("tiny", device="cuda", compute_type="float16",
                         download_root=download_path)
    return model


def transcribe_audio(audio_path, temp_dir):
    """
    Transcribes a given audio file using faster-whisper.
    """
    try:
        model = load_whisper_model()  

        print(f"Transcribing file: {audio_path}")
        sys.stdout.flush()

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=False
        )

        result_text = " ".join([segment.text for segment in segments])

        print(f"Transcription complete. Detected language: {info.language}")
        sys.stdout.flush()

        transcript = f"\nTHE TRANSCRIPT IS:\n\n{result_text.strip()}"
        return transcript, info.language

    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.stdout.flush()
        return None, None




# --- THIS IS THE UPDATED FUNCTION ---

def translate_text(text, source_lang):
    """
    Translates text from a source language to English using m2m100_418M
    FORCING CPU to avoid VRAM conflicts with Whisper.
    """
    
    clean_text = text.replace("THE TRANSCRIPT IS:", "").strip()
    
    print(f"✅ translate_text: Starting translation from '{source_lang}' to 'en'...")
    st.info(f"1b/3 - Translating text from '{source_lang}' to 'en'...")
    

    device = "cpu"
    print(f"✅ translate_text: Forcing device '{device}' to ensure stability.")
    st.warning("Translation is running on CPU to prevent crashes. This step may be slow...")

    sys.stdout.flush()
    
    translator_pipeline = None
    model_name = "Helsinki-NLP/opus-mt-hi-en"

    try:
        print(f"✅ translate_text: Attempting to load model '{model_name}' onto device '{device}'...")
        sys.stdout.flush()
        
        translator_pipeline = pipeline(
            "translation",
            model=model_name,
            device=device  
        )
        
        print(f"✅ translate_text: Model loaded successfully onto '{device}'.")
        sys.stdout.flush()

    except Exception as e:
        print(f"⚠️ translate_text: CRITICAL ERROR loading model: {e}")
        sys.stdout.flush()
        st.error(f"Failed to load translation model: {e}")
        return None
    

    # IF MODEL LOADED, PROCEED 
    try:
        max_chunk_size = 400 
        words = clean_text.split()
        chunks = [" ".join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
        
        translated_chunks = []
        print(f"✅ translate_text: Starting translation of {len(chunks)} chunks...")
        sys.stdout.flush()
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Translating chunk {i}/{len(chunks)}...")
            sys.stdout.flush()
            
            result = translator_pipeline(chunk, src_lang=source_lang, tgt_lang="en")
            
            translated_chunks.append(result[0]['translation_text'])
        
        translated_text = " ".join(translated_chunks)
        print("✅ Translation complete.")
        sys.stdout.flush()
        return translated_text

    except Exception as e:
        print(f"⚠️ translate_text: Error *during* translation: {e}")
        sys.stdout.flush()
        st.error(f"Error during translation process: {e}")
        return None
    finally:
        # --- Cleanup ---
        if translator_pipeline:
            print("✅ translate_text: Cleaning up translation model...")
            sys.stdout.flush()
            del translator_pipeline
            print("✅ translate_text: Cleanup complete.")
            sys.stdout.flush()

    
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def summarize_text(text, source_lang="en"):
    """
    Summarizes text. If source_lang is not 'en', it translates first.
    """

    print(f"✅ summarize_text: Received text. Language is '{source_lang}'.")
    sys.stdout.flush()
    text_to_summarize = ""

    # ------- LANGUAGE HANDLING -------
    if source_lang != "en" and source_lang is not None:
        try:
            print("✅ summarize_text: Language is not 'en'. Calling translate_text...")
            sys.stdout.flush()
            translated_text = translate_text(text, source_lang)
            if translated_text is None:
                return None
            text_to_summarize = translated_text
            st.info("2/3 - Summarizing translated text...")
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return None
    else:
        print("✅ summarize_text: Language is 'en'. Skipping translation...")
        sys.stdout.flush()
        text_to_summarize = text.replace("THE TRANSCRIPT IS:", "").strip()
        st.info("2/3 - Summarizing English text...")

    print("✅ summarize_text: Calling Groq LLaMA model for summarization...")
    sys.stdout.flush()

    # ------- SPLIT INTO CHUNKS -------
    max_chunk_size = 500
    words = text_to_summarize.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    summaries = []


    # GROQ SUMMARIZATION REQUEST (Text Section)

    for idx, chunk in enumerate(chunks, 1):
        summary_prompt = f"""
        You are a professional summarizer. Your job is to summarize the given transcript.
        Rules:
        - Create a clean bullet point summary.
        - Remove filler words & repetitions.
        - Maintain the original meaning
        - Final summary must be in English.
        - 200 to 250 words max.

        Text to summarize:
        {chunk}

        Now produce the summary:
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": summary_prompt}]
            )

            summaries.append(response.choices[0].message.content.strip())
            print(f"✅ summarize_text: Processed chunk {idx}/{len(chunks)}")
            sys.stdout.flush()

        except Exception as e:
            print(f"⚠️ Error during Groq summarization: {e}")
            st.error(f"Error during Groq summarization: {e}")
            return None

    # ------- COMBINE MULTIPLE CHUNKS -------
    if len(summaries) > 1:
        combined = " ".join(summaries)

        final_prompt = f"""
        Combine these partial summaries into a cohesive summary.
        - Remove duplicates & repetition
        - Produce clean formatted bullet points
        - Limit final answer to 8-10 lines and remove bullets

        Partial summaries:
        {combined}

        Final Combined Summary:
        """

        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": final_prompt}]
        )

        return final_response.choices[0].message.content.strip()

    return summaries[0]


# Audio Section

def text_to_audio(text, audio_path):
    """
    Converts text to speech using ElevenLabs (High Quality) and saves it as an MP3 file.
    """
    print("✅ text_to_audio: Converting final summary to speech with ElevenLabs...")
    sys.stdout.flush()
    st.info("3/3 - Generating realistic AI voice... 🔉")

    try:
        api_key = "sk_4cc22ec775e06d19cfa966bffae711c1e2acda6089022617" 

        clean_summary = text.replace(".", "").strip().replace("\n\n", "\n")

        client = ElevenLabs(api_key=api_key)

        audio = client.text_to_speech.convert(
            text=clean_summary,
            voice_id="21m00Tcm4TlvDq8ikWAM", 
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
            
        )

        # Save the audio file
        save(audio, audio_path)
        print("✅ text_to_audio: MP3 file saved.")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"⚠️ Error during Text-to-Speech conversion: {e}")
        sys.stdout.flush()
        st.error(f"Error during Text-to-Speech conversion: {e}")
        return False



# FAST SCENE DETECTION USING FFMPEG (Video Section)

def detect_scenes_fast(video_path, threshold=0.4):
    """
    Detects scene boundaries using FFmpeg's built-in scene change detector.
    Returns list of (start_seconds, end_seconds)
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]

    print("⏳ Detecting scenes...", flush=True)
    output = subprocess.run(cmd, stderr=subprocess.PIPE, text=True).stderr

    timestamps = re.findall(r"pts_time:(\d+\.\d+)", output)

    if not timestamps:
        print("⚠ No scenes detected. Returning default 60 sec window.")
        return [(0, 60)]

    scenes = []
    prev = 0.0

    for ts in timestamps:
        ts = float(ts)
        scenes.append((prev, ts))
        prev = ts

    return scenes



# PICK TOP N LONGEST SCENES

def select_key_scenes(scenes, max_scenes=5):
    print(f"🎯 Selecting top {max_scenes} important scenes...", flush=True)
    return sorted(scenes, key=lambda x: x[1] - x[0], reverse=True)[:max_scenes]



# SUPER-FAST VIDEO SUMMARY WITH FFMPEG CONCAT + COPY

def create_video_summary_ffmpeg(video_path, scenes, output_path="summary_video.mp4"):
    print("✂ Creating summary video clips...", flush=True)

    temp_list = "clips.txt"
    with open(temp_list, "w") as f:
        for i, (start, end) in enumerate(scenes):
            clip_path = f"clip{i}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c", "copy",      # No re-encoding = super fast
                clip_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            f.write(f"file '{clip_path}'\n")

    print("🔗 Merging clips...", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_list,
        "-c", "copy", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup temporary files
    for i in range(len(scenes)):
        os.remove(f"clip{i}.mp4")
    os.remove(temp_list)

    print(f"🎉 Summary video saved as: {output_path}")
    return output_path


# MAIN PIPELINE CALL EXAMPLE

def summarize_video(video_path, output="summary_video.mp4"):
    scenes = detect_scenes_fast(video_path)
    key_scenes = select_key_scenes(scenes)
    return create_video_summary_ffmpeg(video_path, key_scenes, output)







