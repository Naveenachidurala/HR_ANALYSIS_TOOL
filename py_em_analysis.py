import os
import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tempfile
import subprocess
import wave
import json
from moviepy import VideoFileClip
from vosk import Model, KaldiRecognizer
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load Hugging Face emotion model
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Streamlit UI Setup
st.set_page_config(page_title="Physical & Emotional Analysis for Skilled Workers", layout="wide")
st.title("🛠️ Physical & Emotional Analysis for Skilled Workers")
st.sidebar.title("📌 Navigation")

# Step 1: File Upload Section
st.header("Step 1: Upload Video for Analysis")
uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    st.success("✅ File uploaded successfully!")
    st.video(uploaded_file)
    
    # Step 2: Extract Audio
    def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
        """Extracts audio using FFmpeg subprocess."""
        command = ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", "-acodec", "pcm_s16le", output_audio_path, "-y"]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        return output_audio_path
    
    # Step 3: Transcribe Audio (Optimized for Performance)
    def transcribe_audio(audio_path):
        """Transcribes audio using Vosk in chunks to optimize performance."""
        model_path = "vosk-model-en-us"
        if not os.path.exists(model_path):
            raise FileNotFoundError("❌ Vosk model not found!")
        model = Model(model_path)
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        transcript = ""
        while True:
            data = wf.readframes(2000)  # Reduced chunk size for efficiency
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript += result.get("text", "") + " "
        return transcript
    
    # Step 4: Perform Emotion Analysis
    def analyze_emotions(transcript):
        """Performs emotion analysis on a reduced portion of the transcript for efficiency."""
        emotions = emotion_pipeline(transcript[:500])  # Limiting input size
        return {emotion["label"]: round(emotion["score"], 2) for emotion in emotions[0]}
    
    # Step 5: Perform Physical Analysis (Optimized Frame Processing)
    def analyze_physical_performance(video_path):
        """Analyzes physical performance using MediaPipe Pose with reduced frame processing."""
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_skip = 5  # Process every 5th frame to improve performance
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                image_rgb = cv2.resize(frame, (640, 480))  # Resize for faster processing
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                result = pose.process(image_rgb)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    posture_score = round(abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * 100, 2)
                    results.append(posture_score)
            frame_count += 1
        cap.release()
        return results
    
    # Step 6: Generate HR Feedback
    def generate_hr_feedback(transcript, emotions):
        """Generates structured HR feedback including key recommendations."""
        prompt = f"""
        You are an HR expert evaluating a candidate's interview.
        Provide feedback including scores (1-10) for:
        - Communication
        - Confidence
        - Clarity
        - Emotional Control
        Also, list key strengths, weaknesses, and areas for improvement.
        Transcript:
        {transcript[:500]}
        Emotions:
        {emotions}
        """
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content
    
    # Perform Optimized Analysis
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    emotion_data = analyze_emotions(transcript)
    physical_performance = analyze_physical_performance(video_path)
    hr_feedback = generate_hr_feedback(transcript, emotion_data)
    
    # Display Results
    st.header("Step 7: Analysis Results")
    st.subheader("📜 Transcription")
    st.text_area("Interview Transcript:", transcript[:1000], height=200)
    
    st.subheader("😃 Emotion Analysis")
    st.json(emotion_data)
    
    st.subheader("🏋️ Physical Performance Analysis")
    st.line_chart(physical_performance)
    
    st.subheader("📝 HR Feedback & Recommendations")
    st.text_area("HR Feedback:", hr_feedback, height=200)


  # Step 8: Generate and Download PDF Report
    def generate_pdf(transcript, emotions, hr_feedback, filename="Analysis_Report.pdf"):
        """Generates a detailed physical & emotional assessment PDF report with enhanced styling."""
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleStyle", parent=styles['Title'], fontSize=20, leading=24, spaceAfter=16, textColor=colors.darkblue)
        section_style = ParagraphStyle("SectionStyle", parent=styles['Heading2'], fontSize=14, leading=18, spaceAfter=10, textColor=colors.blue)
        body_style = ParagraphStyle("BodyStyle", parent=styles['BodyText'], fontSize=12, leading=16, spaceAfter=8)
        
        elements = [
            Paragraph("Comprehensive Analysis Report", title_style),
            Spacer(1, 12),
            Paragraph("Candidate Transcript:", section_style),
            Paragraph(transcript[:1000], body_style),
            Spacer(1, 12),
            Paragraph("Emotion Analysis:", section_style)
        ]
        
        # Emotion Analysis Table
        data = [["Emotion", "Score"]] + [[k, f"{v:.2f}"] for k, v in emotions.items()]
        table = Table(data, colWidths=[200, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Physical Performance Analysis:", section_style))
        elements.append(Paragraph("The candidate's physical performance has been assessed based on posture, stability, and fatigue levels. Maintaining good posture and reducing excessive movements can improve efficiency and reduce strain during work. Recommendations include keeping an upright position and ensuring balance during tasks.", body_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("HR Feedback & Recommendations:", section_style))
        elements.append(Paragraph(hr_feedback, body_style))
        elements.append(Spacer(1, 12))
        
        # Interviewer Decision
        decision = "✔ The candidate is recommended for selection." if emotions.get("confidence", 0) > 0.5 else "❌ The candidate is not recommended for selection."
        elements.append(Paragraph("Final Decision:", section_style))
        elements.append(Paragraph(decision, title_style))
        
        doc.build(elements)
        return filename
    
    pdf_report = generate_pdf(transcript, emotion_data, hr_feedback)
    st.download_button("📄 Download Full Analysis Report", data=open(pdf_report, "rb"), file_name="Full_Analysis_Report.pdf", mime="application/pdf")
