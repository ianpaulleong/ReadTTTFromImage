# SECTION 1: IMPORTING LIBRARIES, HYPERPARAMETERS
# Here's where I import things
import glob
import pickle
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.optim import lr_scheduler
from utilityFunctions import loadJpgFile
from utilityFunctions import loadTxtFile
from utilityFunctions import flipLRTxtData
from utilityFunctions import flipUDTxtData
from utilityFunctions import train_model
from utilityFunctions import compareAValImg

# These are the tunable hyperparameters
batchSize = 4
# learning_rate = .0000001
learning_rate = .001
numTrainingIters = 12
reloadData = 0
recreateNetwork = 0
loadNetworkFromFile = 1
step_size = 3
gamma = 0.1
det_thresholdX = 0.45
det_thresholdO = 0.38

# SECTION 2: IMPORT TRAINING DATA
basic_transform = transforms.Compose([
    transforms.Resize(244),
    transforms.ToTensor()
])
color_transform = transforms.Compose([
    transforms.Resize(244),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

# If it's loaded, don't reload the data. 
# This determines what preprocessing will happen to the images. It might belong in the hyperparameters section
if reloadData:
    # Create lists of all .jpg and .txt in the training data folder
    trainDir = 'Images/Train/'
    txtFileList = glob.glob(trainDir + '*.txt')
    jpgFileList = glob.glob(trainDir + '*.jpg')
    
    # Prepare the lists that will contain the data
    jpgList = []
    txtList = []
    
    # Load the data into the batch lists. 
    numFiles = len(txtFileList)
    for ii in range(numFiles):
        print('File number:',ii,'\n')
        # Load data from files
        curTxtData = loadTxtFile(txtFileList[ii])
        curJpgData = loadJpgFile(basic_transform,jpgFileList[ii])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        
        # Augment through flips!
        curTxtData = flipLRTxtData(curTxtData)
        curJpgData = curJpgData.flip([3])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        curTxtData = flipUDTxtData(curTxtData)
        curJpgData = curJpgData.flip([2])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        curTxtData = flipLRTxtData(curTxtData)
        curJpgData = curJpgData.flip([3])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        
        # Augment through color jitter AND flips!
        curTxtData = loadTxtFile(txtFileList[ii])
        curJpgData = loadJpgFile(color_transform,jpgFileList[ii])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        curTxtData = flipLRTxtData(curTxtData)
        curJpgData = curJpgData.flip([3])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        curTxtData = flipUDTxtData(curTxtData)
        curJpgData = curJpgData.flip([2])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)
        curTxtData = flipLRTxtData(curTxtData)
        curJpgData = curJpgData.flip([3])
        txtList.append(curTxtData)
        jpgList.append(curJpgData)

# SECTION 3: LOAD PRETRAINED NETWORK, MODIFY, TRAIN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if recreateNetwork:
    myModel = torchvision.models.resnet18(pretrained=True)
    num_ftrs = myModel.fc.in_features
    myModel.fc = nn.Linear(num_ftrs, 18)
#    myModel = torchvision.models.alexnet(pretrained=True)
#    num_ftrs = myModel.classifier[6].in_features
#    myModel.classifier[6] = nn.Linear(num_ftrs, 18)
    myModel = myModel.to(device)
elif loadNetworkFromFile:
    with open('retrainedModel.pickle', 'rb') as handle:
        myModel = pickle.load(handle)
    myModel.to(device)

myLossFunction = torch.nn.SmoothL1Loss()
#myLossFunction = torch.nn.MSELoss()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
#myOptimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myScheduler = lr_scheduler.StepLR(myOptimizer, step_size, gamma)

model_ft = train_model(device, batchSize, jpgList, txtList,'train', myModel, myLossFunction, myOptimizer, myScheduler, numTrainingIters)


# SECTION 4: VALIDATION TIME!
# Load the validation images
valJpgList = glob.glob('Images/Val/*.jpg')
valTxtList = glob.glob('Images/Val/*.txt')
# Compare model output to correct answers for a few select images. This should 
# be for-looped but isn't yet.
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,3,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,4,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,6,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,10,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,13,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,15,det_thresholdX,det_thresholdO)
compareAValImg(myModel,basic_transform,device,valTxtList,valJpgList,16,det_thresholdX,det_thresholdO)


# SECTION 5: SAVE TRAINED MODEL
myModel.to('cpu')
with open('retrainedModel.pickle', 'wb') as handle:
    pickle.dump(myModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
myModel.to(device)