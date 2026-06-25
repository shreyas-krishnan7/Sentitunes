import cv2
import numpy as np
from deepface import DeepFace
import time
from collections import Counter
import threading
import streamlit as st

class FacialEmotionDetector:
    def __init__(self):
        self.emotions_detected = []
        self.confidence_scores = []
        self.is_recording = False
        self.frame_count = 0
        self.detection_duration = 3  # seconds
        self.fps_target = 10  # frames per second for analysis
        
        # Emotion mapping from DeepFace to your mood system
        self.emotion_to_mood_mapping = {
            'angry': 'angry',
            'disgust': 'angry',
            'fear': 'anxious',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'happy',
            'neutral': 'chill'
        }
    
    def detect_face_emotion(self, frame):
        """
        Detect emotion from a single frame using DeepFace
        """
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                rgb_frame, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                result = result[0]  # Take the first face
            
            # Get emotion with highest confidence
            emotions = result['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion] / 100.0  # Convert to 0-1 scale
            
            return dominant_emotion, confidence, emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None, 0, {}
    
    def analyze_face_emotions_realtime(self, placeholder_status, placeholder_preview):
        """
        Capture and analyze emotions for 3 seconds
        """
        self.emotions_detected = []
        self.confidence_scores = []
        self.is_recording = True
        self.frame_count = 0
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None, 0, "Error: Could not access camera"
        
        start_time = time.time()
        last_analysis_time = 0
        analysis_interval = 1.0 / self.fps_target  # Analyze every 1/fps_target seconds
        
        try:
            while self.is_recording and (time.time() - start_time) < self.detection_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_time = self.detection_duration - elapsed_time
                
                # Update preview (convert BGR to RGB for streamlit)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder_preview.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Analyze emotion at intervals
                if current_time - last_analysis_time >= analysis_interval:
                    emotion, confidence, all_emotions = self.detect_face_emotion(frame)
                    
                    if emotion:
                        self.emotions_detected.append(emotion)
                        self.confidence_scores.append(confidence)
                        self.frame_count += 1
                        
                        # Update status
                        placeholder_status.markdown(
                            f"""
                            <div style="background-color: #292930; padding: 15px; border-radius: 10px; text-align: center;">
                                <h3 style="color: #ff4b94;">🎭 Analyzing Your Expression...</h3>
                                <p style="color: white; font-size: 18px;">Time Remaining: {remaining_time:.1f}s</p>
                                <p style="color: #f5c2d0;">Current Emotion: {emotion.title()}</p>
                                <p style="color: #cccccc;">Confidence: {confidence:.1%}</p>
                                <p style="color: #888;">Frames Analyzed: {self.frame_count}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    last_analysis_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except Exception as e:
            return None, 0, f"Error during analysis: {str(e)}"
        finally:
            cap.release()
            self.is_recording = False
        
        # Analyze results
        if self.emotions_detected:
            final_emotion, final_confidence = self.determine_final_emotion()
            mapped_mood = self.emotion_to_mood_mapping.get(final_emotion, 'neutral')
            return mapped_mood, final_confidence, None
        else:
            return None, 0, "No emotions detected. Please try again."
    
    def determine_final_emotion(self):
        """
        Determine the final emotion based on collected data
        Uses both frequency and confidence-weighted scoring
        """
        if not self.emotions_detected:
            return None, 0
        
        # Method 1: Most frequent emotion
        emotion_counts = Counter(self.emotions_detected)
        most_frequent = emotion_counts.most_common(1)[0][0]
        
        # Method 2: Confidence-weighted average
        emotion_confidence_sums = {}
        emotion_confidence_counts = {}
        
        for emotion, confidence in zip(self.emotions_detected, self.confidence_scores):
            if emotion not in emotion_confidence_sums:
                emotion_confidence_sums[emotion] = 0
                emotion_confidence_counts[emotion] = 0
            
            emotion_confidence_sums[emotion] += confidence
            emotion_confidence_counts[emotion] += 1
        
        # Calculate average confidence for each emotion
        emotion_avg_confidence = {}
        for emotion in emotion_confidence_sums:
            emotion_avg_confidence[emotion] = (
                emotion_confidence_sums[emotion] / emotion_confidence_counts[emotion]
            )
        
        # Find emotion with highest average confidence
        highest_confidence_emotion = max(emotion_avg_confidence, key=emotion_avg_confidence.get)
        
        # Decide between frequency and confidence-based result
        if most_frequent == highest_confidence_emotion:
            # Both methods agree
            final_emotion = most_frequent
            final_confidence = emotion_avg_confidence[final_emotion]
        else:
            # Methods disagree - use confidence-weighted if confidence is significantly higher
            freq_confidence = emotion_avg_confidence[most_frequent]
            high_confidence = emotion_avg_confidence[highest_confidence_emotion]
            
            if high_confidence > freq_confidence + 0.1:  # 10% threshold
                final_emotion = highest_confidence_emotion
                final_confidence = high_confidence
            else:
                final_emotion = most_frequent
                final_confidence = freq_confidence
        
        return final_emotion, final_confidence
    
    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False

def streamlit_facial_emotion_interface():
    """
    Streamlit interface for facial emotion detection
    """
    st.markdown("### 😊 Facial Emotion Detection")
    st.markdown("Let AI analyze your facial expressions to detect your current mood!")
    
    # Instructions
    st.markdown(
        """
        <div style="background-color: #292930; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #f5c2d0;">📋 Instructions:</h4>
        <ul style="color: #cccccc;">
            <li>Click "Start Emotion Detection" to begin</li>
            <li>Position your face clearly in the camera view</li>
            <li>The system will analyze your expressions for 3 seconds</li>
            <li>Make sure you have good lighting</li>
            <li>Look directly at the camera for best results</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize detector
    if 'emotion_detector' not in st.session_state:
        st.session_state.emotion_detector = FacialEmotionDetector()
    
    # Initialize session state for emotion detection
    if 'emotion_result' not in st.session_state:
        st.session_state.emotion_result = None
    if 'emotion_confidence' not in st.session_state:
        st.session_state.emotion_confidence = 0
    if 'emotion_error' not in st.session_state:
        st.session_state.emotion_error = None
    if 'emotion_detecting' not in st.session_state:
        st.session_state.emotion_detecting = False
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📹 Start Emotion Detection", key="start_emotion_detection", 
                    disabled=st.session_state.emotion_detecting):
            st.session_state.emotion_detecting = True
            st.session_state.emotion_result = None
            st.session_state.emotion_confidence = 0
            st.session_state.emotion_error = None
            
            # Create placeholders for status and preview
            status_placeholder = st.empty()
            preview_placeholder = st.empty()
            
            # Show initial status
            status_placeholder.markdown(
                """
                <div style="background-color: #292930; padding: 15px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #ff4b94;">🎭 Initializing Camera...</h3>
                    <p style="color: white;">Please wait while we access your camera</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Start emotion detection
            try:
                mood, confidence, error = st.session_state.emotion_detector.analyze_face_emotions_realtime(
                    status_placeholder, preview_placeholder
                )
                
                if error:
                    st.session_state.emotion_error = error
                elif mood:
                    st.session_state.emotion_result = mood
                    st.session_state.emotion_confidence = confidence
                    
                    # Show success message
                    status_placeholder.markdown(
                        f"""
                        <div style="background-color: #292930; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #00ff00;">✅ Analysis Complete!</h3>
                            <p style="color: white; font-size: 20px;">Detected Mood: <strong>{mood.title()}</strong></p>
                            <p style="color: #f5c2d0;">Confidence: {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.session_state.emotion_error = "No clear emotion detected"
                    
            except Exception as e:
                st.session_state.emotion_error = f"Unexpected error: {str(e)}"
            
            finally:
                st.session_state.emotion_detecting = False
                # Clear the preview after detection
                time.sleep(2)
                preview_placeholder.empty()
    
    with col2:
        if st.button("🔄 Retry Detection", key="retry_emotion_detection", 
                    disabled=st.session_state.emotion_detecting):
            st.session_state.emotion_result = None
            st.session_state.emotion_confidence = 0
            st.session_state.emotion_error = None
            st.rerun()
    
    with col3:
        if st.button("📝 Switch to Text", key="switch_from_emotion"):
            st.session_state.input_method = "text"
            st.session_state.emotion_result = None
            st.session_state.emotion_confidence = 0
            st.session_state.emotion_error = None
            st.rerun()
    
    # Display results
    if st.session_state.emotion_error:
        st.error(f"❌ {st.session_state.emotion_error}")
        st.markdown(
            """
            <div style="background-color: #292930; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4 style="color: #f5c2d0;">💡 Troubleshooting Tips:</h4>
            <ul style="color: #cccccc;">
                <li>Make sure your camera is not being used by another application</li>
                <li>Check that your browser has camera permissions</li>
                <li>Ensure good lighting on your face</li>
                <li>Try moving closer to the camera</li>
                <li>If problems persist, use text or voice input instead</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    elif st.session_state.emotion_result:
        st.success(f"✅ Detected Mood: **{st.session_state.emotion_result.title()}**")
        st.info(f"🎯 Confidence: **{st.session_state.emotion_confidence:.1%}**")
        
        # Show emotion mapping info
        st.markdown(
            f"""
            <div style="background-color: #292930; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4 style="color: #f5c2d0;">🎭 Emotion Analysis Details:</h4>
            <p style="color: #cccccc;">
                <strong>Input Method:</strong> 📹 Facial Expression Analysis<br>
                <strong>Detection Duration:</strong> 3 seconds<br>
                <strong>Frames Analyzed:</strong> {st.session_state.emotion_detector.frame_count}<br>
                <strong>Final Mood:</strong> {st.session_state.emotion_result.title()}<br>
                <strong>Confidence Score:</strong> {st.session_state.emotion_confidence:.1%}
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Return the result for use in main app
        return st.session_state.emotion_result
    
    return None

# Test function for standalone usage
def test_facial_emotion_detection():
    """
    Test the facial emotion detection in a standalone manner
    """
    detector = FacialEmotionDetector()
    
    print("🎭 Testing Facial Emotion Detection")
    print("Press 'q' to quit during detection")
    
    # Create dummy placeholders for testing
    class DummyPlaceholder:
        def markdown(self, text, **kwargs):
            print(f"Status: {text}")
        def image(self, image, **kwargs):
            pass  # Skip image display in console
    
    status_placeholder = DummyPlaceholder()
    preview_placeholder = DummyPlaceholder()
    
    try:
        mood, confidence, error = detector.analyze_face_emotions_realtime(
            status_placeholder, preview_placeholder
        )
        
        if error:
            print(f"❌ Error: {error}")
        elif mood:
            print(f"✅ Detected Mood: {mood.title()}")
            print(f"🎯 Confidence: {confidence:.1%}")
        else:
            print("❌ No emotion detected")
            
    except KeyboardInterrupt:
        print("\n⏹️ Detection stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_facial_emotion_detection()