# TDI_Challenge_2017
Build a classifier to classify X-ray images diagnosis


## Data Source
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737
Only images_001.tar.gz, images_0012.tar.gz and images_003.tar.gz were used for this challenge. Those three compressed files contain 24999 images, which are in format of (1024, 1024, 3). 

Download the data to directory 'images'

## Preprocessing
Resize each image into (224, 224, 3), and encode the labels, saved as "Resampled_data.h5".

## Use VGG16 convolutional layer to compute bottleneck features
Split the data into train, validation and test dataset, compute and save the bottlenexk features into "bottleneck_features_224_224_train.npy", "bottleneck_features_224_224_validation.npy" and "bottleneck_features_224_224_test.npy"

## Train the model on top of the features
Build a fully convolutional layer to train the model

## Visualize what the model learn
Show what the model learns by plotting class activation maps
