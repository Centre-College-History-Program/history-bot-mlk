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
            dataset[key.lower()].append(fileName)
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
        validFactType = False
        validFactTypes = ['a', 't', 'r']
        while not validFactType:
            fact_type = raw_input("Please enter the fact type (t for text, a for audio, r for a random value from an array).\n")
            validFactType = fact_type in validFactTypes
            if not validFactType:
                print("That fact type is not valid, please give a valid type.")
        if fact_type == 'a':
            fact = raw_input("Please enter the name of your audio file and put the file in the Dataset_Files directory.\n")
        elif fact_type == 'r':
            fact = []
            addAnotherFact = True
            while addAnotherFact:
                newFact = raw_input("Please enter another string that can be chosen (Enter 'q' to quit).")
                if newFact.lower().strip() != 'q':
                    fact.add(newFact)
                else:
                    addAnotherFact = False
        else:
            fact = raw_input("Please enter your fact.\n")
        keys = raw_input("Please enter your keys (separate keys with a space.)\n")
        essential_keys = raw_input("Please enter your essential keys (separate keys with a space.)\n")

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
        contInput = raw_input("Would you like to continue (Y for yes, N for no)?")
        cont = contInput.lower() == 'y'
main()
