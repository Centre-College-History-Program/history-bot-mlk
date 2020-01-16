import os
import json

class Dataset:
    
    def __init__(self):
        self.datasetDirectory = "Dataset_Files"
        datasetFileName = "dataset.txt"
        datasetFile = open(os.path.join(self.datasetDirectory, datasetFileName), 'r')
        index = datasetFile.read()
        datasetFile.close()
        self.dataset = json.loads(index)
    
    def swap(self, array, index1, index2):
        tempValue = array[index1]
        array[index1] = array[index2]
        array[index2] = tempValue
    
    def insertSort(self, arrayInts, arrayVals):
        endValue = 1
        while endValue < len(arrayInts):
            index = endValue
            cont = True
            while cont:
                if arrayInts[endValue] > arrayInts[index - 1]:
                    index -= 1
                    cont = index == 0
                else:
                    cont = False
            swap(arrayInts, endValue, index)
            swap(arrayVals, endValue, index)
            endValue += 1
    
    def getResponseObject(self, responseFileName):
        responseFile = open(os.path.join(self.datasetDirectory, responseFileName), 'r')
        responseObject = json.loads(responseFile.read())
        responseFile.close()
        return responseObject
    
    def getPossibleResponses(self, keys):
        responses = []
        responsesFrequency = []
        for key in keys:
            try:
                key = key.lower()
                newResponses = self.dataset[key]
                for response in newResponses:
                    if response in responses:
                        responsesFrequency[responses.index(response)] += 1
                    else:
                        responses.append(response)
                        responsesFrequency.append(1)
            except:
                pass
        self.insertSort(responsesFrequency, responses)
        return responses, responsesFrequency
    
    def getFilePath(self, fileName):
        return os.path.join(self.datasetDirectory, fileName)