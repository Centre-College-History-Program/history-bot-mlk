from Voice_Cloning.command_line_interface import VoiceCloner
import os
import json

datasetDirectory = "Dataset_Files"
outputDirectory = 'MLK_Speech_Files'
voiceCloner = VoiceCloner()
'''
datasetFiles = os.listdir(datasetDirectory)
for fileName in datasetFiles:
    try:
        hasInt = int(fileName[:-4]) #Test to see if the file name has an integer

        #Get the data out of the file
        fileName += ".wav"
        print("Synthesizing ", fileName)
        filePath = os.path.join(datasetDirectory, fileName)
        file = open(filePath, 'r')
        data = json.loads(file.read())
        file.close()
        fact = data['fact']
        fact_type = data['fact_type'] 

        #Synthesize the speech
        if fact_type == 't':
            outputFilePath = os.path.join(outputDirectory, fileName[:-4])
            voiceCloner.synthesize(fact)
            voiceCloner.vocode(outputFilePath)
        elif fact_type == 'r':
            index = 0
            while index < len(fact):
                text = fact[index]
                outputFilePath = os.path.join(outputDirectory, fileName[:-4] + "_" + str(index))
                voiceCloner.synthesize(text)
                voiceCloner.vocode(outputFilePath)
    except Exception as e:
        print(e)
'''
fileName = '-1.wav'
voiceCloner.synthesize("I don't know how to respond to that.")
voiceCloner.vocode(os.path.join(outputDirectory, fileName))
fileName = '-2.wav'
voiceCloner.synthesize("I couldnt understand you.")
voiceCloner.vocode(os.path.join(outputDirectory, fileName))
