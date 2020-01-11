# ReadTTTFromImage
This project exists to train a Convolutional Neural Network to read the current state of a tic tac toe board from a well-cropped and well-oriented image. I perform transfer learning on resnet18, redesigning the output layer to function for regression rather than classification.

Training and validation data should be placed in a folder called 'Images', in the subfolders 'Training' and Val. 

The data I used for training can be found at:

**Releases**:     
https://github.com/ianpaulleong/ReadTTTFromImage/releases

`wget https://github.com/ianpaulleong/readTTTFromImage/releases/download/1.0.1/Images.zip`
