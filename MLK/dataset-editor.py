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
    
def saveDataset():
    datasetFile = open(datasetFilePath, 'w')
    datasetFile.write(json.dumps(dataset))
    datasetFile.close()

def changeKey():
    #Get the key information
    oldKey = raw_input('What is the existing spelling of the key?\n')
    newKey = raw_input('What is the new spelling of the key?\n')

    #Get the existing files associated with the keys
    oldKeyFiles = dataset[oldKey]
    newKeyFiles = []
    if newKey in dataset:
        newKeyFiles = dataset[newKey]

    for oldKeyFile in oldKeyFiles:
        if oldKeyFile not in newKeyFiles:
            newKeyFiles.append(oldKeyFile)

    dataset[newKey] = newKeyFiles
    del dataset[oldKey]

    for fileName in oldKeyFiles:
        file = open(os.path.join(datasetDirectory, fileName), 'r')
        data = json.loads(file.read())
        file.close()
        data['keys'].append(newKey)
        data['keys'].remove(oldKey)
        file = open(os.path.join(datasetDirectory, fileName), 'w')
        file.write(json.dumps(data))
        file.close()
        
    
def editKey():
    
    #Get the key and file names
    key = raw_input('What is your key?\n')
    files = raw_input('What files would you like to have this key?\n(E.g. write "1 2 3" if you want this key for 1.txt, 2.txt, and 3.txt)\n')
    
    key = key.lower().strip() #Remove whitespace from the key
    files = files.split(' ') #Split up the file names
    
    #Establish what the existing files associated with the key are
    existingFiles = []
    if key in dataset:
        existingFiles = dataset[key]
        
    #Remove empty file names and remove extra whitespace
    index = 0
    while index < len(files):
        file = files[index]
        if file.strip() == '':
            del files[index]
        else:
            files[index] = file.strip()
            index += 1
    fileNames = []
    
    #Add the appropriate file extension onto the file names
    for file in files:
        fileNames.append(file + ".txt")
        
    #Associate the files witht the key
    existingFiles.extend(fileNames)
    dataset[key] = existingFiles 
    
    #Add the key to the files
    for fileName in fileNames:
        file = open(os.path.join(datasetDirectory, fileName), 'r')
        data = json.loads(file.read())
        file.close()
        data['keys'].append(key)
        file = open(os.path.join(datasetDirectory, fileName), 'w')
        file.write(json.dumps(data))
        file.close()
        
def editFile():
    #Get the file and the new keys
    fileName = raw_input('What is your file (e.g. type 1 for 1.txt)?\n')
    keys = raw_input('What are the new keys (separate the keys with a space)?\n')
    
    keys = keys.split(' ') #Separate the keys
    
    #Adjust the file name and add create the file path
    fileName = fileName.strip() + '.txt'
    filePath = os.path.join(datasetDirectory, fileName)
    
    #Get the existing keys associated with file
    file = open(filePath, 'r')
    data = json.loads(file.read())
    file.close()
    existingKeys = data['keys']
    
    #Filter the existing keys out of the new keys
    newKeys = []
    for key in keys:
        if key not in existingKeys:
            newKeys.append(key)
        else:
            print("Key already exists for file: " + key + "\n")
            
    #Add the new keys to the file
    existingKeys.extend(newKeys)
    data['keys'] = existingKeys
    file = open(filePath, 'w')
    file.write(json.dumps(data))
    file.close()
    
    #Add the file to the keys in the dataset
    for key in newKeys:
        if key not in dataset:
            dataset[key] = []
        dataset[key].append(fileName)

def makeKeysLowercase():
    keys = list(dataset.keys())
    for key in keys:
        if key != 'nextFile':
            data = dataset[key]
            del dataset[key]
            key = key.lower().strip()
            if key in dataset:
                data.extend(dataset[key])
            dataset[key] = data
            
    maxFile = dataset["nextFile"]
    currFile = 1
    while currFile < maxFile:
        currFileName = str(currFile) + ".txt"
        print (currFileName)
        file = open(os.path.join(datasetDirectory, currFileName), 'r')
        data = json.loads(file.read())
        file.close()
        fileKeys = data['keys']
        lowerKeys = []
        for key in fileKeys:
            lowerKeys.append(key.lower())
        data['keys'] = lowerKeys
        file = open(os.path.join(datasetDirectory, currFileName), 'w')
        file.write(json.dumps(data))
        file.close()
        currFile += 1

def handleInput(text):
    if text == 'h':
        printHelp()
    elif text == 'a':
        editKey()
    elif text == 'f':
        editFile()
    elif text == 'l':
        makeKeysLowercase()
    elif text == 'c':
        changeKey()
    else:
        print("That was not a recongizable input\n")

def printHelp():
    helpString = "\n"
    helpString += "q - quit\n"
    helpString += "h - help\n"
    helpString += "a - add files to a key\n"
    helpString += "f - add keys to a file\n"
    helpString += "l - make all keys lowercase\n"
    helpString += "s - change the spelling of a key\n"
    print(helpString)

def main():
    loadDataset()
    cont = True
    print("Welcome to the dataset editor. These are your possible commands:")
    printHelp()
    while cont:
        text = raw_input("Enter your command:\n")
        text = text.lower()
        if text != 'q':
            handleInput(text)
        else:
            cont = False
        saveDataset()
    
main()
