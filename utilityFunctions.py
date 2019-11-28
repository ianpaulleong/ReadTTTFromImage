import numpy as np
import torch
import time
import copy
import PIL.Image as PILImage
import matplotlib.pyplot as mplpp

def loadTxtFile(theLocAndName):
    openedFile = open(theLocAndName)
    theStrData = openedFile.read()
    openedFile.close()
    theStrData.replace('\n','')
    theTextData = torch.tensor(list(map(float,theStrData.split(','))))
    #theTextData = theTextData.long()
    return theTextData.unsqueeze(0)

def loadJpgFile(transformer, theLocAndName):
    image = PILImage.open(theLocAndName)
    image = transformer(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def plotThisTensorImg(inTens):
    inNp = inTens.detach().numpy()
    inNp = inNp.transpose((1,2,0))
    mplpp.imshow(inNp)

def checkAValImg(model,transformer,pathToFile,device):
    loadedImg = loadJpgFile(transformer,pathToFile)
    loadedImg = loadedImg.to(device)
    return model(loadedImg).reshape([6,3])
    
# This was copied with minor modifications from Sasank Chilamkurthy's Transfer Learning for Computer Vision Tutorial:
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

def train_model(device, inPictures, truthData, phase, model, criterion, optimizer, scheduler, num_epochs=25):
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
        numBatches = len(inPictures)
        batchSize = truthData[0].size()[0]
        for ii in range(numBatches):
            aPictureBatch = inPictures[ii].to(device)
            aTruthDataBatch = truthData[ii].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outData = model(aPictureBatch)
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
    