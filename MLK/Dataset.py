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
        return responses, responsesFrequency
    
    def getFilePath(self, fileName):
        return os.path.join(self.datasetDirectory, fileName)
