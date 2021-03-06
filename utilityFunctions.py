import numpy as np
import torch
import time
import copy
import PIL.Image as PILImage
import matplotlib.pyplot as mplpp
from random import shuffle
import math

def loadTxtFile(theLocAndName):
    '''
    This function reads a comma-delineated .txt file where the first 9 numbers
    locate where the X's are on the Tic-Tac-Toe board in the corresponding
    and the next 9 locate the O's. So for instance, if the board was:
        +-+-+-+
        |X| | |
        +-+-+-+
        | |O| |
        +-+-+-+
        | | |X|
        +-+-+-+
    Then the text file would read:
        1,0,0,
        0,0,0,
        0,0,1,
        
        0,0,0,
        0,1,0,
        0,0,0
    
    This is read into a 1x18 pytorch tensor of type float. There's a reason why 
    it's two dimensions (1x18) instead of one dimension (size: 18); I can't 
    remember what that reason is. It breaks if I don't do it this way, though.
    
    Later note: it's because I stack board states for multiple training 
    examples along the first dimension.
    
    This function takes as input a string containing the path (i.e. 
    'Images/Val\\IMG_20191126_181744.txt') to the .txt file being loaded.
    '''
    openedFile = open(theLocAndName)
    theStrData = openedFile.read()
    openedFile.close()
    theStrData.replace('\n','')
    theTextData = torch.tensor(list(map(float,theStrData.split(','))))
    #theTextData = theTextData.long()
    return theTextData.unsqueeze(0)


def flipUDTxtData(theTxtDataTensor):
    '''
    This is used to augment the data. It flips board position along the 
    horizontal axis. 
    '''
    outTensor = theTxtDataTensor[[0],[6,7,8,3,4,5,0,1,2,15,16,17,12,13,14,9,10,11]]
    return outTensor.unsqueeze(0)


def flipLRTxtData(theTxtDataTensor):
    '''
    This is used to augment the data. It flips board position along the 
    vertical axis.
    '''
    outTensor = theTxtDataTensor[[0],[2,1,0,5,4,3,8,7,6,11,10,9,14,13,12,17,16,15]]
    return outTensor.unsqueeze(0)


def loadJpgFile(transformer, theLocAndName):
    '''
    This loads data from a .jpg file into a pytorch tensor. It requires a 
    torchvision transformer and the path to the jpg file being loaded. After 
    doing whatever preprocessing done by the transformer, it then outputs a
    pytorch tensor of size 1x3xNxM, where N and M is the size of the image 
    (assuming three color channels, which seems reasonable to me). 
    
    Like with the .txt file loader, I also add another dimension of size 1 to 
    the front because who knows why. 
    
    Oh, I just remembered! It lets me easily stack multiple images into a pile 
    of images to use for training. Side note: this also means that currently if
    the orientation or length/width ratio of the training images are not all
    identical, it'll break. That's something to fix in the future. Portrait AND
    Landscape please.
    '''
    image = PILImage.open(theLocAndName)
    image = transformer(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def plotThisTensorImg(inTens):
    '''
    A quick way to visualize the loaded .jpg files. I probably used this while
    debugging, but I think it's obsolete now.
    '''
    inNp = inTens.detach().numpy()
    inNp = inNp.transpose((1,2,0))
    mplpp.imshow(inNp)


def loadAValImgAndExtractBoardState(model,transformer,pathToFile,device):
    '''
    This function is used as part of the validation function I put together 
    below. First it loads a .jpg file, then passes it through the CNN to 
    extract the board state, and finally reshapes the resulting tensor.
    '''
    loadedImg = loadJpgFile(transformer,pathToFile)
    loadedImg = loadedImg.to(device)
    return model(loadedImg).reshape([6,3])


def compareAValImg(model,transformer,device,txtList,jpgList,whichImg,thresholdX = -1,thresholdO = -1):
    '''
    This is the function I used to validate my model. For a single validation
    image, compare the model output to truth data. 
    
    There's a lot of inputs, so let's go over them:
        model: the CNN whose performance I want to validate
        transformer: a torchvision transformer that preprocesses images
        device: this indicates whether we're on the CPU or the GPU
        txtList: a list of paths for the .txt files that contain the 
                 validation truth data
        jpgList: a list of paths for the .jpg image files to be used for 
                 validation
        whichImg: the index of the specific .jpg file to be used for this one
                  validation instance
        thresholdX: the threshold value above which it will be assumed that a 
                    square contains an X.
        thresholdO: the threshold value above which it will be assumed that a 
                    square contains an O. I determined through testing that it
                    made sense to have seperate threshold values for X and O.
    
    If the thresholded model output perfectly matches the truth data, the 
    function simply prints a happy message. If not, then the model data, the 
    thresholded model data, the truth data, and the location of errors are 
    printed out.
    
    This function was designed to also work if no threshold values for X and O
    were given. In this case, it would simply display the truth data along with
    the scores for X and O for all positions.
    
    '''
    cowImg = loadAValImgAndExtractBoardState(model,transformer,jpgList[whichImg],device)
    cowTxt = loadTxtFile(txtList[whichImg])
    cowTxt = cowTxt.reshape([6,3])
    cowTxt = cowTxt.int().to(device).byte()
    if thresholdX == -1:
        print('Output Scores:')
        print(cowImg)
        print('Truth Data:')
        print(cowTxt)
    else:
        xTens = cowImg[0:3,0:3] > thresholdX
        oTens = cowImg[3:6,0:3] > thresholdO
        xoTens = torch.cat([xTens,oTens],0)
        areTheyTheSameTens = (cowTxt == xoTens)
        areTheyTheSame = areTheyTheSameTens.all().item()
        if areTheyTheSame == 1:
            print("No Errors!")
        else:
            print('Output Scores:')
            print(cowImg)
            print('Output:')
            print(xoTens)
            print('Truth Data:')
            print(cowTxt)
            print('Errors at:')
            print(~areTheyTheSameTens)
    print('\n')



# This function was copied with minor modifications from Sasank Chilamkurthy's 
# Transfer Learning for Computer Vision Tutorial. I did the bare minimum to get
# it working for training only, and I've kludged out all the bits that relate
# to using this function for validation. I really need to clean it up and/or
# reactivate the validation part of this function. Eventually.
# 
## BSD 3-Clause License
#
#Copyright (c) 2017, Pytorch contributors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def train_model(device, batchSize, inPictures, truthData, phase, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10^15

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Do things differently for training and evaluation
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        #running_corrects = 0

        # Iterate over data.
        numFiles = len(inPictures)
        numBatches = math.floor(numFiles/batchSize)
        fileAccessOrder = np.linspace(0, numFiles-1, numFiles, dtype = int)
        shuffle(fileAccessOrder)
        for ii in range(numBatches):
            baseNum = ii*batchSize
            fileAccessNum = fileAccessOrder[baseNum]
            batchJpgData = inPictures[fileAccessNum]
            batchTxtData = truthData[fileAccessNum]
            for jj in range(1,4):
                fullNum = baseNum + jj
                fileAccessNum = fileAccessOrder[fullNum]
                currentJpgData = inPictures[fileAccessNum]
                currentTxtData = truthData[fileAccessNum]
                batchJpgData = torch.cat((batchJpgData,currentJpgData),0)
                batchTxtData = torch.cat((batchTxtData,currentTxtData),0)
                
            aPictureBatch = batchJpgData.to(device)
            aTruthDataBatch = batchTxtData.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outData = model(aPictureBatch)
                #loss = criterion(outData, 8*aTruthDataBatch)
                #loss = criterion(outData, aTruthDataBatch.round())
                loss = criterion(outData, aTruthDataBatch)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * batchSize
            #running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        numData = batchSize*numBatches
        epoch_loss = running_loss / numData
        #epoch_acc = running_corrects.double() / numData

#        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#            phase, epoch_loss, epoch_acc))
        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights (I NEED TO DO STUFF TO MAKE THIS WORK BUT IT IS NOT THIS DAY!)
    # model.load_state_dict(best_model_wts)
    return model
    