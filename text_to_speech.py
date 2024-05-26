# Import the required module for text 
# to speech conversion
from gtts import gTTS
import googletrans
import translators as ts
from translate import Translator
from pygame import mixer
import time
# This module is imported so that we can 
# play the converted audio
import os

mixer.init()

f= open('output.txt', 'r')
lang = f.readline()
text = []
for line in f:
    text.append(line)
text = ''.join(text)
print(text)

print("Enter the language code:")
language = input()
translator = Translator(to_lang=language)
result = translator.translate(text)
print(result)


# Language in which you want to convert
# language = 'en'

# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=result, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("text_to_speech_output.mp3")

# Playing the converted file
mixer.music.load("text_to_speech_output.mp3")
mixer.music.play()
while mixer.music.get_busy():  # wait for music to finish playing
    time.sleep(1)
# playsound("text_to_speech_output.mp3")
