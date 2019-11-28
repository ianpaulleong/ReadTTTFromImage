import numpy as np
import torch
import PIL.Image as PILImage
import matplotlib.pyplot as mplpp

def loadTxtFile(theLocAndName):
    openedFile = open(theLocAndName)
    theStrData = openedFile.read()
    openedFile.close()
    theStrData.replace('\n','')
    theTextData = torch.tensor(list(map(int,theStrData.split(','))))
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
    