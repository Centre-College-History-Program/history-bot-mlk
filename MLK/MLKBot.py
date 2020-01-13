#Source here: https://www.simplifiedpython.net/speech-recognition-python/
import speech_recognition as sr     # import the library
import RPi.GPIO as GPIO
import time
import os
import random

def getResponse():
    a

def main():
    GPIO.setmode(GPIO.BCM) #use the GPIO numbering
    GPIO.setwarnings(False) # Avoids warning channel is already in use

    button = 18 # GPIO pin 18

    GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP) #sets up pin 18 as a button

    i_count = 0 # set up for correct grammar in notification below

    r = sr.Recognizer()
    while True:
        input_state = GPIO.input(button) # primes the button!
        if input_state == False:
            with sr.Microphone() as source:     # mention source it will be either Microphone or audio files.
                print("Speak Anything :")
                audio = r.listen(source)        # listen to the source
                try:
                    text = r.recognize_google(audio)    # use recognizer to convert our audio into text part.
                    print("You said : {}".format(text))
                except:
                    print("Sorry could not recognize your voice")    # In case of voice not recognized  clearly
            time.sleep(1.0)

main()