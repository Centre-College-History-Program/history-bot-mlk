#Source here: https://www.simplifiedpython.net/speech-recognition-python/
import speech_recognition as sr     # import the library
import RPi.GPIO as GPIO
import subprocess
from multiprocessing import Process
import time
import os
import random
from Tkinter import *
import tkSnack
from Dataset import Dataset
from gtts import gTTS
#from Real-Time-Voice-Cloning import VoiceCloning.Toolbox

random.seed()
audioProcess = None   
processing_led_1 = 21 
processing_led_2 = 16

def getResponseObjects(responses, dataset):
    responseObjects = []
    for response in responses:
        responseObject = dataset.getResponseObject(response)
        responseObjects.append(responseObject)
    return responseObjects

def calculateResponseObjectFitness(responseObject, responseFrequency):
    return float(responseFrequency)/len(responseObject["keys"])

def flashLights():
    pauseTime = 1
    while True:
        GPIO.output(processing_led_1, True)
        GPIO.output(processing_led_2, True)
        time.sleep(pauseTime)
        GPIO.output(processing_led_1, False)
        GPIO.output(processing_led_2, False)
        time.sleep(pauseTime)
    
def stopLights():
    GPIO.output(processing_led_1, GPIO.LOW)

def getFinalResponse(text, dataset):
    requiredMaxFrequency = 3
    requiredAccuracy = 0.33
    
    keys = text.split(" ")
    responses, responsesFrequency = dataset.getPossibleResponses(keys)
    responseObjects = getResponseObjects(responses, dataset)
    
    #Calculate the best response
    responsesFitness = []
    
    #Find all of the responses which tied with the most keyword matches
    maxFrequency = 0
    bestResponses = []
    for index in range(len(responsesFrequency)):
        frequency = responsesFrequency[index]
        if frequency > maxFrequency:
            bestResponses = []
            bestResponses.append(responseObjects[index])
            maxFrequency = frequency
    
    for i in range(len(bestResponses)):
        responseObject = bestResponses[i]
        responseFitness = calculateResponseObjectFitness(responseObject, maxFrequency)
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
    finalResponse = {}
    finalResponse["fact"] = "I don't know how to respond to that"
    finalResponse["fact_type"] = "t"
    if highestFitnessIndex != -1 and (maxFrequency > requiredMaxFrequency or responsesFitness[highestFitnessIndex] > requiredAccuracy):
        finalResponse = bestResponses[highestFitnessIndex]
    return finalResponse

def playAudio(output):
    global audioProcess
    audioProcess = subprocess.Popen(['omxplayer', '-o', 'alsa', output], stdin=subprocess.PIPE)

def speak(output):
    tts = gTTS(text=output, lang='en')
    print('Saving...')
    tts.save("speech.mp3")
    os.system("mpg321 speech.mp3")

def main():
    dataset = Dataset()
    
    GPIO.setmode(GPIO.BCM) #use the GPIO numbering
    GPIO.setwarnings(False) # Avoids warning channel is already in use

    button = 18 # GPIO pin 18
    button_led = 17

    GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP) #sets up pin 18 as a button
    GPIO.setup(button_led, GPIO.OUT)
    GPIO.setup(processing_led_1, GPIO.OUT)
    GPIO.setup(processing_led_2, GPIO.OUT)
    GPIO.output(button_led, True)

    i_count = 0 # set up for correct grammar in notification below

    r = sr.Recognizer()
    cont = True
    while cont:
        input_state = GPIO.input(button) # primes the button!
        
        if input_state == False:
            
            flashLightsProcess = Process(target=flashLights)
            flashLightsProcess.start()
            
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
                    if text != 'stop':
                        finalResponse = getFinalResponse(text, dataset)
                        output = finalResponse["fact"]
                        outputType = finalResponse["fact_type"]
                    else:
                        cont = False
                except Exception as e:
                    print(e)
                if cont:
                    if outputType == 'a':
                        playAudio(dataset.getFilePath(output))
                    elif outputType == 'r':
                        choice = output[random.randint(0, len(output) - 1)]
                        speak(choice)
                    else:
                        speak(output)
            
            flashLightsProcess.terminate()
            stopLights()
            time.sleep(1.0)
    GPIO.cleanup()

main()
