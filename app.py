import streamlit as st
import time
import os
from main import detect_mood_genre_keywords, determine_final_mood, extract_time_range, fetch_playlists, recommend_music
from main import train_knn_model
from voice_input import get_voice_input
import threading
from facial_emotion_detections import streamlit_facial_emotion_interface

# Set page config
st.set_page_config(
    page_title="SentiTunes",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .main {
        background-color: #1c1c1e;
        color: white;
        text-align: center;
        padding: 30px 10px;
    }
    .square-img {
        width: 200px;
        height: 200px;
        object-fit: cover;
        border: 3px solid lightgrey;
    }
    .title {
        font-size: 4.5rem;
        font-weight: bold;
        color: #ff4b94;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 2rem;
        color: #f5c2d0;
    }
    .mood-card {
        background-color: #292930;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .playlist-card {
        background-color: #292930;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .playlist-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .song-card {
        background-color: #1F1F23;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 8px 0;
        border-left: 4px solid #ff4b94;
    }
    .recommendation-title {
        font-size: 24px;
        color: #f5c2d0;
        margin: 15px 0;
    }
    .voice-button {
        background-color: #ff4b94;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        margin: 10px 0;
    }
    .voice-status {
        background-color: #292930;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ff4b94;
    }
    .input-method-selector {
        background-color: #292930;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource  # This will cache the model
def load_knn_model():
    knn_model, _, _ = train_knn_model()
    return knn_model

# Initialize session state for voice input
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""
if 'voice_status' not in st.session_state:
    st.session_state.voice_status = ""
if 'input_method' not in st.session_state:
    st.session_state.input_method = "text"
if 'voice_recording' not in st.session_state:
    st.session_state.voice_recording = False

knn_model = load_knn_model()

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("static/images/bawra.jpg", width=300)

with col2:
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">SentiTunes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An AI powered playlist generating platform based on your mood</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.image("static/images/gustakh.jpg", width=300)
    
# Input section with enlarged title
st.markdown(
    '<p style="font-size:35px; font-weight:600; color:white;">How was your day?</p>',
    unsafe_allow_html=True
)

# Input method selector
st.markdown('<div class="input-method-selector">', unsafe_allow_html=True)
st.markdown("### Choose your input method:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📝 Type Your Feelings", key="text_input_btn", use_container_width=True):
        st.session_state.input_method = "text"
        st.session_state.voice_text = ""
        st.session_state.voice_status = ""

with col2:
    if st.button("🎙️ Speak Your Feelings", key="voice_input_btn", use_container_width=True):
        st.session_state.input_method = "voice"

with col3:
    if st.button("📹 Facial Expression", key="facial_input_btn", use_container_width=True):
        st.session_state.input_method = "facial"

st.markdown('</div>', unsafe_allow_html=True)

# Voice input section
def handle_voice_input():
    """Handle voice input in a separate thread"""
    try:
        st.session_state.voice_status = "🔊 Adjusting for ambient noise..."
        st.session_state.voice_recording = True
        
        # Get voice input
        voice_text = get_voice_input()
        
        if voice_text:
            st.session_state.voice_text = voice_text
            st.session_state.voice_status = f"✅ Recognized: {voice_text}"
        else:
            st.session_state.voice_status = "❌ Could not recognize speech. Please try again."
            
    except Exception as e:
        st.session_state.voice_status = f"❌ Error: {str(e)}"
    finally:
        st.session_state.voice_recording = False

if st.session_state.input_method == "facial":
    # Run the facial emotion detection interface
    mood = streamlit_facial_emotion_interface()
    user_feelings = mood if mood else ""
elif st.session_state.input_method == "voice":
    st.markdown("### 🎙️ Voice Input")
    
    # Voice input controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎤 Start Recording", key="start_recording", disabled=st.session_state.voice_recording):
            # Start voice recording
            st.session_state.voice_status = "🎙️ Click 'Record Voice' and speak when ready..."
            
        if st.button("🔴 Record Voice", key="record_voice", disabled=st.session_state.voice_recording):
            with st.spinner("🎙️ Listening... Please speak now!"):
                try:
                    voice_text = get_voice_input()
                    if voice_text:
                        st.session_state.voice_text = voice_text
                        st.session_state.voice_status = f"✅ Successfully recorded: '{voice_text}'"
                    else:
                        st.session_state.voice_status = "❌ Could not understand speech. Please try again or use text input."
                except Exception as e:
                    st.session_state.voice_status = f"❌ Error during recording: {str(e)}"
    
    with col2:
        if st.button("🗑️ Clear Recording", key="clear_voice"):
            st.session_state.voice_text = ""
            st.session_state.voice_status = "Recording cleared."
            
        if st.button("📝 Switch to Text", key="switch_to_text"):
            st.session_state.input_method = "text"
            st.session_state.voice_status = ""
    
    # Display voice status
    if st.session_state.voice_status:
        st.markdown(
            f'<div class="voice-status">{st.session_state.voice_status}</div>',
            unsafe_allow_html=True
        )
    
    # Display recognized text in an editable text area
    if st.session_state.voice_text:
        st.markdown("### ✏️ Recognized Text (You can edit this):")
        user_feelings = st.text_area(
            "Edit your recognized speech if needed:",
            value=st.session_state.voice_text,
            height=150,
            key="voice_text_edit"
        )
    else:
        user_feelings = ""
        st.info("👆 Click 'Record Voice' to start speaking, or switch to text input.")

else:  # Text input method
    st.markdown("### 📝 Text Input")
    user_feelings = st.text_area(
        "Share your thoughts and emotions...", 
        height=200, 
        placeholder="Describe about your day here... add info about any specific genre you want to hear or music from a particular timeperiod. If in dilemma leave it to us we will do that job for you!!!",
        key="text_input_area"
    )

# Instructions
st.markdown(
    """
    <div style="background-color: #292930; padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h4 style="color: #f5c2d0;">💡 Tips for better results:</h4>
    <ul style="color: #cccccc;">
        <li><strong>Voice Input:</strong> Speak clearly and describe your feelings naturally</li>
        <li><strong>Text Input:</strong> Mention specific genres, time periods, or detailed emotions</li>
        <li><strong>Examples:</strong> "I'm feeling sad and want to listen to some old Bollywood songs" or "I'm happy and want upbeat pop music"</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Submit button
# Submit button with all input methods handling
if st.button("🎵 Analyze My Mood & Find Music", key="submit", use_container_width=True):
    # Get input based on method
    if st.session_state.input_method == "facial":
        user_input = st.session_state.emotion_result if st.session_state.emotion_result else ""
        confidence = st.session_state.emotion_confidence if hasattr(st.session_state, 'emotion_confidence') else 0
    else:
        user_input = user_feelings.strip() if user_feelings else ""
        confidence = 0

    if user_input:
        with st.spinner("🎭 Analyzing your mood and finding the perfect tunes..."):
            try:
                # Get mood analysis results
                mood, mood_results, confidence_scores = determine_final_mood(user_input, knn_model)
                
                # For facial input, override the mood with detected emotion
                if st.session_state.input_method == "facial":
                    mood = user_input
                    confidence_scores = {'facial': confidence}
                
                # Extract preferences
                _, genre, _ = detect_mood_genre_keywords(user_input)
                years = extract_time_range(user_input)
                country = 'IN'  # Default to India
                
                # Display analysis results
                st.markdown('<div class="mood-card">', unsafe_allow_html=True)
                st.markdown("### 🎭 Mood Analysis Results")
                st.markdown(f"**Detected Mood:** {mood.title()}")
                
                # Show input method used
                input_method_icons = {
                    "voice": "🎙️ Voice Input",
                    "text": "📝 Text Input",
                    "facial": "📹 Facial Expression"
                }
                st.markdown(f"**Input Method:** {input_method_icons[st.session_state.input_method]}")
                
                # Analysis details
                st.markdown("#### Analysis Details:")
                
                method_names = {
                    'vader': 'Sentiment Analysis',
                    'keyword': 'Keyword Detection',
                    'fuzzy': 'Fuzzy Matching',
                    'knn': 'ML Classification',
                    'facial': 'Facial Expression'
                }

                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.input_method == "facial":
                        st.markdown(f"• Facial Analysis: **{mood.upper()}** (confidence: {confidence:.0%})")
                    else:
                        for method, result in mood_results.items():
                            if result:
                                conf = confidence_scores.get(method, 0)
                                conf_str = f"{conf:.0%}" if conf else "N/A"
                                st.markdown(f"• {method_names[method]}: **{result.upper()}** (confidence: {conf_str})")

                with col2:
                    if genre:
                        st.markdown(f"**Genre Detected:** {genre.title()}")
                    st.markdown(f"**Time Range:** {'Recent' if years == 1 else f'Last {years} years'}")
                    st.markdown(f"**Region:** {country}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Get and display playlist recommendations
                playlists = fetch_playlists(mood, genre, country, years)
                
                if playlists and len(playlists) > 0:
                    st.markdown('<div class="mood-card">', unsafe_allow_html=True)
                    st.markdown(f"## 🎵 Music For Your {mood.title()} Mood")
                    
                    for playlist in playlists:
                        if isinstance(playlist, dict) and 'title' in playlist and 'url' in playlist:
                            st.markdown(
                                f"""
                                <div class="playlist-card">
                                    <h3>🎵 {playlist['title']}</h3>
                                    <a href="{playlist['url']}" target="_blank" style="color: #ff4b94; text-decoration: none;">
                                        🔗 Open in YouTube Music
                                    </a>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        f"<p style='color: #888; font-size: 12px; margin-top: 20px;'>Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No playlists found for your mood. Try describing your feelings differently.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    else:
        input_messages = {
            "voice": "Please record your voice first, or switch to text input.",
            "facial": "Please complete the facial expression analysis first.",
            "text": "Please enter your feelings before submitting."
        }
        st.warning(input_messages[st.session_state.input_method])
        
# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #888;">
    <p>🎵 SentiTunes - Your AI-powered mood music companion</p>
    <p>Supports both voice and text input for the best user experience</p>
    </div>
    """,
    unsafe_allow_html=True
)