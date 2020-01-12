import json
import os

dataset = None
datasetDirectory = "Dataset_Files"
datasetFileName = "dataset"
datasetFilePath = os.path.join(datasetDirectory, datasetFileName)
    
def loadDataset():
    datasetFile = open(datasetFilePath, 'w')
    a

def save(fact, keys, essential_keys):
    a

def main():
    cont = True
    while cont:
        fact = input("Please enter your fact.")
        keys = input("Please enter your keys (separate keys with a space.)")
        essential_keys = input("Please enter your essential keys (separate keys with a space.)")

        keys = keys.split(' ')
        essential_keys = essential_keys.split(' ')

        save(fact, keys, essential_keys)
        contInput = input("Would you like to continue (Y for yes, N for no)?")
        cont = contInput.trim().lower() == 'y'
main()
