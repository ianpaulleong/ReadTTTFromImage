# SECTION 1: IMPORTING LIBRARIES, HYPERPARAMETERS
# Here's where I import things
import glob
import numpy as np
import torch
from torchvision import transforms
from random import shuffle
from utilityFunctions import loadJpgFile
from utilityFunctions import loadTxtFile

# These are the tunable hyperparameters
batchSize = 4

# SECTION 2: IMPORT TRAINING DATA

# Create lists of all .jpg and .txt in the training data folder
trainDir = 'Images/Train/'
txtFileList = glob.glob(trainDir + '*.txt')
jpgFileList = glob.glob(trainDir + '*.jpg')

# Randomize the order in which the data is loaded
numFiles = len(txtFileList)
fileAccessOrder = np.linspace(0, numFiles-1, numFiles, dtype = int)
shuffle(fileAccessOrder)

# Prepare the lists that will contain the individual batches
jpgBatchList = []
txtBatchList = []

# This determines what preprocessing will happen to the images. It might belong in the hyperparameters section
data_transforms = transforms.Compose([
    transforms.Resize(244),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load the data into the batch lists. 
# Note: currently it's assumed that the number of files is divisible by the batch size. FIX ASAP
for ii in range(int(numFiles/batchSize)):
    print('ALERT: iteration number',ii,'\n')
    baseNum = ii*batchSize
    fileAccessNum = fileAccessOrder[baseNum]
    batchTxtData = loadTxtFile(txtFileList[fileAccessNum])
    batchJpgData = loadJpgFile(data_transforms,jpgFileList[fileAccessNum])
    for jj in range(1,4):
        fullNum = baseNum + jj
        fileAccessNum = fileAccessOrder[fullNum]
        currentTxtData = loadTxtFile(txtFileList[fileAccessNum])
        currentJpgData = loadJpgFile(data_transforms,jpgFileList[fileAccessNum])
        batchTxtData = torch.cat((batchTxtData,currentTxtData),0)
        batchJpgData = torch.cat((batchJpgData,currentJpgData),0)
    txtBatchList.append(batchTxtData)
    jpgBatchList.append(batchJpgData)

# SECTION 3: 
    
    