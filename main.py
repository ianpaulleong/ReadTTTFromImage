# SECTION 1: IMPORTING LIBRARIES, HYPERPARAMETERS
# Here's where I import things
import glob
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from random import shuffle
from torch.optim import lr_scheduler
from utilityFunctions import loadJpgFile
from utilityFunctions import loadTxtFile
from utilityFunctions import train_model

# These are the tunable hyperparameters
batchSize = 4
learning_rate = 0.001
numTrainingIters = 20
reloadData = 0
reloadNetwork = 0

# SECTION 2: IMPORT TRAINING DATA

# If it's loaded, don't reload the data. 
if reloadData:
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
    numBatches = len(txtBatchList)

# SECTION 3: LOAD PRETRAINED NETWORK, MODIFY, TRAIN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if reloadNetwork:
    myModel = torchvision.models.resnet18(pretrained=True)
    num_ftrs = myModel.fc.in_features
    myModel.fc = nn.Linear(num_ftrs, 18)
#    myModel = torchvision.models.alexnet(pretrained=True)
#    num_ftrs = myModel.classifier[6].in_features
#    myModel.classifier[6] = nn.Linear(num_ftrs, 18)
    myModel = myModel.to(device)
myLossFunction = torch.nn.SmoothL1Loss()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
#myOptimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myScheduler = lr_scheduler.StepLR(myOptimizer, step_size=7, gamma=0.1)

model_ft = train_model(device, jpgBatchList, txtBatchList,'train', myModel, myLossFunction, myOptimizer, myScheduler, numTrainingIters)


        