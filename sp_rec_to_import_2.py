import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

 

def speak(text):
    while True:
        try:
            speech = gTTS(text=text, lang='en', slow=False)
            speech.save('response.mp3')
            #os.system("response.mp3 --silent")
            playsound('response.mp3')
            break
        except Exception as e:
            
            if e is not AssertionError:
                print(e)
                speak(text)
            else:
                speak('Sorry, but I could not detect anything... Is there anything else that I can help you with?')
            break
            

