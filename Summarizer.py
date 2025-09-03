import streamlit as st
import whisper
#from transformers import pipeline
#from gtts import gTTS
from moviepy.editor import VideoFileClip, concatenate_videoclips
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
import math
from faster_whisper import WhisperModel

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")



def transcribe_audio(video_path, chunk_length=360):
    try:
        model = WhisperModel("base", device="cuda", compute_type="float16")

        # model = WhisperModel("tiny", device="cpu", compute_type="int8")  # keep base for accuracy
        result_text = ""

        # Load video and extract audio duration
        video = VideoFileClip(video_path)
        duration = math.ceil(video.duration)  # in seconds

        # Process video in chunks to speed up CPU transcription
        for start in range(0, duration, chunk_length):
            end = min(start + chunk_length, video.duration)
            audio_clip = video.subclip(start, end).audio
            audio_file = f"temp_chunk_{start}_{end}.wav"
            audio_clip.write_audiofile(audio_file, verbose=False, logger=None)

            # Transcribe chunk
            segments,info = model.transcribe(
                audio_file,              
                word_timestamps=False,      # skip word-level timestamps
                language="en",              # lock language to English
            )
            chunk_text = " ".join([segment.text for segment in segments])
            result_text += chunk_text + " "

            # Delete temporary chunk file after transcription
            if os.path.exists(audio_file):
                os.remove(audio_file)

            print(f"✅ Processed chunk {start}-{end} sec")

        return f" \n✨✨✨ THE TRANSCRIPT IS ✨✨✨ : \n\n{result_text.strip()}"
        
    except Exception as e:
        print(f"⚠️ Error during transcription: {e}")
        return None


# Example usage
text = transcribe_audio("temp/Future is Here AI - Sundar Pichai.mp4")
# print(text)



def summarize_text(text):
    
    try:
        llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task= "text-generation")

        meta_llm = ChatHuggingFace(llm=llm)
    
        max_chunk_size = 1500  # approx words, adjust if needed
        words = text.split()
        chunks = [" ".join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

        summaries = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = f"Summarize the following text:\n\n{chunk}\n\nSummary:"
            response = meta_llm.invoke(prompt)
            summaries.append(response.content.strip())
            print(f"\n✅ Processed chunk {idx}/{len(chunks)}")

        # If multiple summaries, summarize them again into a final summary
        if len(summaries) > 1:
            combined_text = " ".join(summaries)
            final_prompt = f"Combine the following partial summaries into one coherent final summary:\n\n{combined_text}\n\nFinal Summary:"
            final_response = meta_llm.invoke(final_prompt)
            return f"\n⭐⭐⭐⭐⭐ 📝 THE SUMMARY IS ⭐⭐⭐⭐⭐\n\n{final_response.content}"
        else:
            return f"\n⭐⭐⭐⭐⭐ 📝 THE SUMMARY IS ⭐⭐⭐⭐⭐\n\n{summaries[0]}"

    except Exception as e:
        print(f"⚠️ Error during summarization: {e}")
        return None
    
Summarized_text = summarize_text(text)
print(Summarized_text)



# def clone_voice_to_audio(text, speaker_wav_path, output_audio_path):
#     """
#     Converts text to speech by cloning the voice from a provided audio sample.
#     Saves the result as an MP3 file.
#     """
#     st.info("3/4 - Cloning voice and generating audio... 🗣️")
#     try:
#         # Note: The first time you run this, it will download the model.
#         # This can take a while and requires an internet connection.
#         # Use gpu=True if you have a CUDA-enabled GPU for much faster performance.
#         tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

#         # Generate speech by cloning the voice from the speaker_wav_path
#         tts.tts_to_file(
#             text=text,
#             speaker_wav=speaker_wav_path,
#             language="en", # Specify the language of the text
#             file_path=output_audio_path
#         )
#         return True
#     except Exception as e:
#         st.error(f"Error during voice cloning: {e}")
#         st.warning("Please ensure you have a good internet connection for the initial model download.")
#         return False
    



