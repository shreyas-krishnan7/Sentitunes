import speech_recognition as sr

def get_voice_input():
    """
    Get voice input from the user using the microphone.
    Returns:
        str: The recognized text from the user's voice input, or None if recognition fails.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🔊 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎙️ Speak now (max 10 sec)...")
            audio = recognizer.listen(source, phrase_time_limit=10)
    except Exception as e:
        print(f"❌ Error accessing microphone: {e}")
        return None

    try:
        print("🔄 Recognizing...")
        text = recognizer.recognize_google(audio)
        print("✅ You said:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Sorry, could not understand your voice.")
        return None
    except sr.RequestError as e:
        print(f"❌ Voice service error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None