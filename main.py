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
from utilityFunctions import train_model
from utilityFunctions import compareAValImg

# These are the tunable hyperparameters
batchSize = 4
learning_rate = .0000001
numTrainingIters = 1
reloadData = 1
reloadNetwork = 0
step_size = 7
gamma = 0.1
det_threshold = 0.45

# SECTION 2: IMPORT TRAINING DATA

# If it's loaded, don't reload the data. 
if reloadData:
    # Create lists of all .jpg and .txt in the training data folder
    trainDir = 'Images/Train/'
    txtFileList = glob.glob(trainDir + '*.txt')
    jpgFileList = glob.glob(trainDir + '*.jpg')
    
    
    # Prepare the lists that will contain the data
    jpgList = []
    txtList = []
    
    # This determines what preprocessing will happen to the images. It might belong in the hyperparameters section
    data_transforms = transforms.Compose([
        transforms.Resize(244),
        #transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Load the data into the batch lists. 
    numFiles = len(txtFileList)
    for ii in range(numFiles):
        print('File number:',ii,'\n')
        batchTxtData = loadTxtFile(txtFileList[ii])
        batchJpgData = loadJpgFile(data_transforms,jpgFileList[ii])
        txtList.append(batchTxtData)
        jpgList.append(batchJpgData)

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
#myLossFunction = torch.nn.MSELoss()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
#myOptimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate, momentum=0.9)
myScheduler = lr_scheduler.StepLR(myOptimizer, step_size, gamma)

model_ft = train_model(device, batchSize, jpgList, txtList,'train', myModel, myLossFunction, myOptimizer, myScheduler, numTrainingIters)


# SECTION 4: VALIDATION TIME!
valJpgList = glob.glob('Images/Val/*.jpg')
valTxtList = glob.glob('Images/Val/*.txt')
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,3, det_threshold)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,4, det_threshold)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,6)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,10)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,13)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,15, det_threshold)
compareAValImg(myModel,data_transforms,device,valTxtList,valJpgList,16)

# SECTION 5: Save the trained model!
with open('retrainedModel.pickle', 'wb') as handle:
    pickle.dump(myModel, handle, protocol=pickle.HIGHEST_PROTOCOL)