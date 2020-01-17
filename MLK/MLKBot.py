#Source here: https://www.simplifiedpython.net/speech-recognition-python/
import speech_recognition as sr     # import the library
import RPi.GPIO as GPIO
import subprocess
import time
import os
import random
from Tkinter import *
import tkSnack
from Dataset import Dataset
    
audioProcess = None    

def getResponseObjects(responses, dataset):
    responseObjects = []
    for response in responses:
        responseObject = dataset.getResponseObject(response)
        responseObjects.append(responseObject)
    return responseObjects

def calculateResponseObjectFitness(responseObject, responseFrequency):
    return float(responseFrequency)/len(responseObject["keys"]) 

def getFinalResponse(text, dataset):
    keys = text.split(" ")
    responses, responsesFrequency = dataset.getPossibleResponses(keys)
    responseObjects = getResponseObjects(responses, dataset)
    
    #Calculate the fitness of each response
    responsesFitness = []
    for i in range(len(responseObjects)):
        responseObject = responseObjects[i]
        responseFrequency = responsesFrequency[i]
        responseFitness = calculateResponseObjectFitness(responseObject, responseFrequency)
        responsesFitness.append(responseFitness)
        
    #Find the response with the highest fitness score
    highestFitness = 0
    highestFitnessIndex = -1
    for i in range(len(responsesFitness)):
        responseFitness = responsesFitness[i]
        if responseFitness > highestFitness:
            highestFitness = responseFitness
            highestFitnessIndex = i
    
    #Return the best response
    finalResponse = -1
    if highestFitnessIndex != -1:
        finalResponse = responseObjects[highestFitnessIndex]
    return finalResponse

def playAudio(output):
    global audioProcess
    audioProcess = subprocess.Popen(['omxplayer', '-o', 'alsa', output], stdin=subprocess.PIPE)

def speak(output):
    print(output)

def main():
    dataset = Dataset()
    
    GPIO.setmode(GPIO.BCM) #use the GPIO numbering
    GPIO.setwarnings(False) # Avoids warning channel is already in use

    button = 18 # GPIO pin 18

    GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP) #sets up pin 18 as a button

    i_count = 0 # set up for correct grammar in notification below

    r = sr.Recognizer()
    while True:
        input_state = GPIO.input(button) # primes the button!
        if input_state == False:
            if audioProcess:
                audioProcess.stdin.write('q')
            with sr.Microphone() as source:     # mention source it will be either Microphone or audio files.
                print("Speak Anything :")
                audio = r.listen(source)
                text = ""
                output = "I'm sorry, I couldn't understand you."
                outputType = 't'
                try:
                    text = r.recognize_google(audio)    # use recognizer to convert our audio into text part.
                    print("You said : {}".format(text))
                    finalResponse = getFinalResponse(text, dataset)
                    if finalResponse == -1:
                        output = "I don't know the answer to that."
                    else:
                        output = finalResponse["fact"]
                        outputType = finalResponse["fact_type"]
                except Exception as e:
                    print(e)
                if outputType == 'a':
                    playAudio(dataset.getFilePath(output))
                else:
                    speak(output)
            time.sleep(1.0)

main()