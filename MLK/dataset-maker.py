import json
import os

dataset = None
datasetKeys = None
datasetDirectory = "Dataset_Files"
datasetFileName = "dataset.txt"
factFileMappingName = "fact-file-mapping.txt"
datasetFilePath = os.path.join(datasetDirectory, datasetFileName)
factFileMappingFilePath = os.path.join(datasetDirectory, factFileMappingName)
    
def loadDataset():
    global dataset, datasetKeys
    datasetFile = open(datasetFilePath, 'r')
    index = datasetFile.read()
    dataset = json.loads(index)
    datasetKeys = dataset.keys()
    datasetFile.close()

def getFileName(fact):
    return str(dataset['nextFile']) + '.txt'

def saveDataset():
    datasetFile = open(datasetFilePath, 'w')
    datasetFile.write(json.dumps(dataset))
    datasetFile.close()

def save(fact, keys, essential_keys, fact_type):
    fileName = getFileName(fact)
    for key in keys:
        try:
            dataset[key].append(fileName)
        except:
            array = []
            array.append(fileName)
            dataset[key] = array
    file = open(os.path.join(datasetDirectory, fileName), 'w')
    file_obj = {}
    file_obj['essentialKeys'] = essential_keys
    file_obj['keys'] = keys
    file_obj['fact'] = fact
    file_obj['fact_type'] = fact_type
    dataset['nextFile'] += 1
    file.write(json.dumps(file_obj))
    file.close()

    factFileMappingFile = open(factFileMappingFilePath, 'a')
    factFileMapping = fileName + " - " + fact
    factFileMappingFile.write(factFileMapping)
    factFileMappingFile.close()

def main():
    cont = True
    loadDataset()
    while cont:
        fact = input("Please enter your fact.\n")
        keys = input("Please enter your keys (separate keys with a space.)\n")
        essential_keys = input("Please enter your essential keys (separate keys with a space.)\n")
        fact_type = input("Please enter the fact type (t for text, a for audio).\n")

        keys = keys.split(' ')
        essential_keys = essential_keys.split(' ')
        fact_type = fact_type.strip().lower()

        index = 0
        while index < len(keys):
            key = keys[index]
            if key.strip() == '':
                del keys[index]
            else:
                index += 1

        index = 0
        while index < len(essential_keys):
            key = essential_keys[index]
            if key.strip() == '':
                del essential_keys[index]
            else:
                index += 1

        save(fact, keys, essential_keys, fact_type)
        saveDataset()
        contInput = input("Would you like to continue (Y for yes, N for no)?")
        cont = contInput.lower() == 'y'
main()
