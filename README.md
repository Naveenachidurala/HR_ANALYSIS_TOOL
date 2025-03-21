# HR_ANALYSIS_TOOL


## 🛠 Technology Stack

### Primary Technologies
- Python: Core programming language
- Streamlit: Web application framework
- OpenAI GPT-4: Advanced language model for analysis
- MediaPipe: Computer vision and machine learning library
- DeepFace: Facial analysis library
- Vosk: Speech recognition model

### Key Libraries and Tools
1. **Video Processing**
   - OpenCV (cv2) – Captures and processes video frames.
   - MoviePy – Handles video editing and manipulation.
   - FFmpeg – Encodes, decodes, and processes video/audio files.

2. **Audio Processing**
   - PyDub – Converts and manipulates audio files.
   - Wave – Reads and writes WAV format audio.
   - Vosk Speech Recognition – Transcribes speech from audio files.

3. **Machine Learning**
   - MediaPipe – Tracks facial expressions and body posture.
   - DeepFace – Analyzes facial emotions and recognition.
   - OpenAI's GPT-4 – Performs sentiment and behavioral analysis.

## 🔍 Process Flow

### 1. Video Upload
- Users can upload interview videos in MP4, MOV, or AVI formats.
- Streamlit provides an intuitive file upload interface.
- Validates and prepares the video for analysis.

### 2. Audio Extraction
- Automatically extracts audio from the uploaded video.
- Converts audio to WAV format with specific encoding.
- Prepares audio for transcription.

### 3. Transcription
- Uses Vosk speech recognition model.
- Converts audio to text.
- Handles various audio qualities and accents.

### 4. Physical Analysis
- Utilizes MediaPipe for facial and body detection.
- DeepFace analyzes facial emotions.
- Detects body movements and gestures.

### 5. Emotion and Tone Detection
- GPT-4 performs in-depth emotional analysis.
- Scores emotions like Confidence, Enthusiasm, Authenticity.
- Provides nuanced textual insights.

### 6. HR Evaluation
- Generates comprehensive behavioral assessment.
- Creates scoring metrics for specific job roles.
- Produces detailed report with recommendations.

### 7. Reporting
- Generates both HTML and PDF reports.
- Rich, visually appealing design.
- Downloadable and shareable formats.

## 📦 Installation
pip install streamlit openai vosk deepface mediapipe opencv-python moviepy pydub pdfkit
