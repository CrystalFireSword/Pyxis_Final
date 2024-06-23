# to run the file, go to terminal and hit 
# streamlit run combined_main.py
import streamlit as st
import os
import pygame
import random
from streamlit_mic_recorder import speech_to_text
import sp_rec_to_import_2 as spr
import time



st.title('PYXIS', )
st.write('Please wait while the site is loading!')



# Display the custom button using markdown

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.init()
pygame.mixer.init()

pygame.mixer.music.set_volume(0.75)
def play_music(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
 
def pause_music():
    pygame.mixer.music.pause()
 
def unpause_music():
    pygame.mixer.music.unpause()
 
def stop_music():
    pygame.mixer.music.stop()

spr.speak('Please wait while the site is loading!')

with st.status('Running'):
    play_music(r"calm-chill-beautiful-141317.mp3")
    import llama_to_import as lti
    import yolo_to_import as yti
    stop_music()

text = None
text = speech_to_text(
    language='en',
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None
)



st.write(text)
if text is not None:
    prompt = text
    
    with st.status('Running'):
        spr.speak('Please wait while we search for your answer!')
        play_music(r"calm-chill-beautiful-141317.mp3")
        available_objects = yti.detect()
        prompt = f'''
You are a friendly assistant application for the visually impaired people.
You will be given a question, and the set of objects that are in front of the user.
Use the information from the objects in front to answer the user's question accurately.
Note that data contains a list of objects in front of the user in each
frame of reference. So you should give an optimal answer depending on the
content of each frame.
Refer to the following examples:

QUERY 1:
'Question: What are the objects in front of me?
Data: ['2 persons, 10 cars, 4 traffic lights, ', '2 persons, 10 cars, 1 motorcycle, 4 traffic lights, ', '2 persons, 10 cars, 1 traffic light, ', '3 persons, 8 cars, 1 motorcycle, 1 traffic light, ', '3 persons, 8 cars, 1 traffic light, ', '3 persons, 8 cars, 2 traffic lights, ']'

EXPECTED OUTPUT 1: There are 2 people in front of you. It also looks like there is a motorcycle and a number of cars. There are traffic signals, so I am guessing you are on a road. Is there anything else I can do for you?
(The answer is NOT 15 people, but 2 people because traversing through each string in the list we find that overall, across frames, there are either 2 or 3 persons.
As the number of people is 2 in most of the frames, we say that the number of people is 2.
)

QUERY 2:
'Question: Is there anything I can travel by?
Data: ['2 persons, 10 cars, 4 traffic lights, ', '2 persons, 10 cars, 1 motorcycle, 4 traffic lights, ', '2 persons, 10 cars, 1 traffic light, ', '3 persons, 8 cars, 1 motorcycle, 1 traffic light, ', '3 persons, 8 cars, 1 traffic light, ', '3 persons, 8 cars, 2 traffic lights, ']'

Expected Output: I am not sure. There are cars and motorcycles in front of you, but they are private transport. So you may not be able to travel by them. Is there anything else I may help you with?


QUERY 3:
'Question: I want to eat now. 
Data: ['2 pens, 1 Notebook, 4 traffic lights, ', '1 mobile phone, 1 pen, 1 notebook, ', '3 bananas, 2 notebooks, 1 pen ']'

Expected Output: I am sorry, but there no is food in front of you! There are only notebooks and pens in front of you... Can I help you with something else?

QUERY 4:
'Question: What is the colour of the pen?
Data: ['2 pens, 1 Notebook, 4 traffic lights, ', '1 mobile phone, 1 pen, 1 notebook, ', '3 bananas, 2 notebooks, 1 pen ']'
Expected output: There are pens in front of you!  But I am really sorry, I am not sure about the colour of the pen. Is there anything else that I may help you with?

QUERY 5:
'Question: What is the colour of the pen?
Data: ['2 persons, 10 cars, 4 traffic lights, ', '2 persons, 10 cars, 1 motorcycle, 4 traffic lights, ', '2 persons, 10 cars, 1 traffic light, ', '3 persons, 8 cars, 1 motorcycle, 1 traffic light, ', '3 persons, 8 cars, 1 traffic light, ', '3 persons, 8 cars, 2 traffic lights, ']'
Expected output: I am sorry, but I don't see any pens in front of you, so I am not able to tell you a colour. There are people, traffic lights, cars and motorcycles in front of you. So I guess you are on a road... Is there anything else I can help you with?

QUERY 6:
Question: Can you please repeat what you just said?
Data: ['2 persons, 10 cars, 4 traffic lights, ', '2 persons, 10 cars, 1 motorcycle, 4 traffic lights, ', '2 persons, 10 cars, 1 traffic light, ', '3 persons, 8 cars, 1 motorcycle, 1 traffic light, ', '3 persons, 8 cars, 1 traffic light, ', '3 persons, 8 cars, 2 traffic lights, ']'
Expected output: I am sorry, but I don't see any pens in front of you, so I am not able to tell you a colour. There are people, traffic lights, cars and motorcycles in front of you. So I guess you are on a road... Is there anything else I can help you with?
(repeat whatever was the answer to the previous query)

Question: {text}
Objects in front: {available_objects}
OUTPUT:

'''
        res = lti.generate_text(prompt)
        result = res[0]["generated_text"][len(prompt):]
        stop_music()
    full_stop_index = len(result)
    for x in range(-1, len(result)-1):
        if result[x]+result[x+1]==' .':
                full_stop_index = x
    result = result[:full_stop_index]
    print(result)
    st.write('Machine:',result)
    spr.speak('Hi!'+result)
    text = None
