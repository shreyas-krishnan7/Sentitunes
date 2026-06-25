# SentiTunes: AI-Based Mood Playlist Generator

SentiTunes is an AI-powered web app that analyzes your mood through text, voice, or facial expressions and recommends personalized music playlists. Whether you're feeling happy, sad, nostalgic, or anything in between, SentiTunes finds the perfect tunes for your emotions—instantly!

---

## 🚀 Features

- **Multi-Modal Mood Detection:**  
  Detects your mood using advanced NLP (VADER, KNN, keyword, fuzzy matching) and Computer Vision (DeepFace).
- **Triple Input Methods:**  
  - 📝 Type your feelings
  - 🎙️ Speak your emotions
  - 📹 Use facial expression analysis
- **Real-time Facial Analysis:**  
  Uses OpenCV and DeepFace for accurate emotion detection from webcam feed.
- **Personalized Playlist Recommendations:**  
  Suggests YouTube Music playlists tailored to your mood, genre, and time preferences.
- **Modern, Responsive UI:**  
  Built with Streamlit for a smooth and interactive experience.
- **Confidence Scoring:**  
  See how confident the AI is in its mood predictions across all detection methods.

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/mood_playlist_app.git
    cd mood_playlist_app
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

Required packages now include:
```
streamlit
opencv-python
deepface
tensorflow
speechrecognition
pyaudio
ytmusicapi
vaderSentiment
fuzzywuzzy
python-Levenshtein
```

4. **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 🎤 Usage

- **Text Input:**  
  Type your feelings and click "Analyze My Mood & Find Music".
- **Voice Input:**  
  Click "Speak Your Feelings", record your voice, and let SentiTunes do the rest!
- **Facial Expression:**  
  Click "Facial Expression", allow camera access, and let the AI analyze your mood.
- **Get Playlists:**  
  Instantly receive music playlists that match your detected mood.

---

## 🧠 How It Works

- **Mood Detection Methods:**
  - Text Analysis: VADER sentiment, KNN classification, keyword matching
  - Voice Input: Speech-to-text followed by sentiment analysis
  - Facial Analysis: DeepFace emotion detection with OpenCV
- **Playlist Recommendation:**  
  Fetches relevant playlists from YouTube Music based on detected mood.
- **Confidence Display:**  
  Shows confidence scores for each detection method used.

---

## 📦 Project Structure

```
mood_playlist_app/
│
├── app.py                      # Streamlit frontend
├── main.py                     # Core logic: mood detection, playlist fetching
├── voice_input.py             # Voice-to-text logic
├── facial_emotion_detections.py # Facial emotion detection logic
├── static/images/             # App images
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

---

## 🙏 Credits

- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [DeepFace](https://github.com/serengil/deepface)
- [TensorFlow](https://www.tensorflow.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [YTMusicAPI](https://ytmusicapi.readthedocs.io/)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

---

## 📣 Future Enhancements

- Spotify/Apple Music integration
- User mood analytics and history
- Multi-language support
- Enhanced emotion detection with multiple ML models
- Gesture recognition for music control

---