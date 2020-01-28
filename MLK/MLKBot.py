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
import signal
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
    GPIO.output(processing_led_2, GPIO.LOW)

def getFinalResponse(text, dataset):
    requiredMaxFrequency = 2
    requiredFileKeysAccuracy = 0.33
    requiredGivenKeysAccuracy = 0.5
    
    keys = text.split(" ")
    responses, responsesFrequency = dataset.getPossibleResponses(keys)
    responseObjects = getResponseObjects(responses, dataset)
    
    #Calculate the best response
    responsesFitness = []
    
    #Find all of the responses which tied with the most keyword matches
    maxFrequency = 0
    bestResponses = []
    bestResponeFiles = []
    for index in range(len(responsesFrequency)):
        frequency = responsesFrequency[index]
        if frequency > maxFrequency:
            bestResponses = []
            bestResponeFiles = []
            bestResponses.append(responseObjects[index])
            bestResponeFiles.append(responses[index])
            maxFrequency = frequency
        elif frequency == maxFrequency:
            bestResponses.append(responseObjects[index])
            bestResponeFiles.append(responses[index])
    
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
    finalResponse["file_name"] = '-1.txt'
    if highestFitnessIndex != -1 and (maxFrequency >= requiredMaxFrequency or responsesFitness[highestFitnessIndex] >= requiredFileKeysAccuracy) and float(maxFrequency)/(len(keys)) >= requiredGivenKeysAccuracy:
        finalResponse = bestResponses[highestFitnessIndex]
        finalResponse["file_name"] = bestResponeFiles[highestFitnessIndex]
    return finalResponse

def playAudio(output):
    global audioProcess
    audioProcess = subprocess.Popen("exec play " + output, stdout=subprocess.PIPE, shell=True)

def speak(response, choice = None):
    fileName = response['file_name']
    audioDirectory = "MLK_Speech_Files"
    fileName = fileName[:-4]
    if choice:
        fileName += '_' + str(choice)
    fileName += ".wav"
    filePath = os.path.join(audioDirectory, fileName)
    print(filePath)
    try:
        f = open(filePath, 'r')
        f.close()
        playAudio(filePath)
    except Exception as e:
        print(e)
        output = response['fact']
        if choice:
            output = output[choice]
        tts = gTTS(text=output, lang='en')
        print('Saving...')
        tts.save("speech.mp3")
        os.system("mpg321 speech.mp3")
    
    if choice: 
        print(response['fact'][choice])
    else:
        print(response['fact'])

def main():
    global audioProcess
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
                try:
                    audioProcess.kill()
                except:
                    pass
                audioProcess = None
            with sr.Microphone() as source:     # mention source it will be either Microphone or audio files.
                print("Speak Anything :")
                audio = r.listen(source)
                text = ""
                finalResponse = {}
                finalResponse['fact'] = "I'm sorry, I couldn't understand you."
                finalResponse['fact_type'] = 't'
                finalResponse['file_name'] = '-2.txt'
                try:
                    text = r.recognize_google(audio)    # use recognizer to convert our audio into text part.
                    print("You said : {}".format(text))
                    if text != 'stop':
                        finalResponse = getFinalResponse(text, dataset)
                        fileName = finalResponse["file_name"]
                    else:
                        cont = False
                except Exception as e:
                    print(e)
                flashLightsProcess.terminate()
                stopLights()
                outputType = finalResponse["fact_type"]
                if cont:
                    if outputType == 'a':
                        playAudio(dataset.getFilePath(finalResponse['fact']))
                    elif outputType == 'r':
                        choice = random.randint(0, len(finalResponse['fact']) - 1)
                        speak(finalResponse, choice = choice)
                    else:
                        speak(finalResponse)
            
            time.sleep(1.0)
    GPIO.cleanup()

main()
