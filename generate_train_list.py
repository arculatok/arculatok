from collections import defaultdict
import os

TRAIN_AUDIO_FOLDER_PATH = "E:\\Jaki\\Msc\\Gepi_Tanulasi_Modszerek\\train\\audio\\"
TESTING_LIST_FILE_PATH = "E:\\Jaki\Msc\\Gepi_Tanulasi_Modszerek\\Speech_Recognition\\arculatok\\testing_list.txt"
VALIDATION_LIST_FILE_PATH = "E:\\Jaki\Msc\\Gepi_Tanulasi_Modszerek\\Speech_Recognition\\arculatok\\validation_list.txt"
GENERATED_FILE_PATH = "E:\\Jaki\\Msc\\Gepi_Tanulasi_Modszerek\\Speech_Recognition\\arculatok\\training_list.txt"

def readFromFileLineByLine(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    return content;

# We want to generate map from lines as follows:
    # key: word (e.g. "bed")
    # value: list of audio file names which belong to the key word
def generateMapsFromLines(arrayOfLines):
    mapFilesToWord = defaultdict(list)
    for line in arrayOfLines:
        splittedLine = line.split('/')
        mapFilesToWord[splittedLine[0]].append(splittedLine[1])
    return mapFilesToWord

def getFileNamesFromFolder(folder):
    files = [f for f in os.listdir(TRAIN_AUDIO_FOLDER_PATH + folder)]
    return files

# In a manner similar to the above, we construct our training set file
def writeGeneratedMapToFile(map):
    generatedFile = open(GENERATED_FILE_PATH, "a")
    for key, value in map.items():
        for file in value:
            generatedFile.write(key + "/" + file + "\n")


if __name__ == "__main__":
    testing_list_map = generateMapsFromLines(readFromFileLineByLine(TESTING_LIST_FILE_PATH))
    validation_list_map = generateMapsFromLines(readFromFileLineByLine(VALIDATION_LIST_FILE_PATH))

    # All possible words
    possibleWords = testing_list_map.keys()
    # Empty dictionary for generating traning set
    mapForTrainSet = defaultdict(list)

    # We have to drop some files (which are for testing and validation) from the traning folder
    # We must do that for all the possible words (30) in for loop
    for word in possibleWords:
        allWordFiles = getFileNamesFromFolder(word)
        filesToDrop = testing_list_map[word] + list((set(validation_list_map[word]) - set(testing_list_map[word])))
        filesRemained = set(allWordFiles) - set(filesToDrop)

        mapForTrainSet[word] = [file for file in filesRemained]

    writeGeneratedMapToFile(mapForTrainSet)