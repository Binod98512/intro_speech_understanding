import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Transcribe speech from a wave file using SpeechRecognition.
    '''
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio, language=language)
    return text
