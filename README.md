# CNN-SVM

A script to extract image features using TensorFlow's trained CNN Inceptionv3 which can be downloaded here: 
https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip 

> extract_features.py is designed to extract features from each image of a database and store them with labels in two .txt files. Note that extraction currently occurs at layer pool_3:0 of inceptionv3 which can be adapted. The way the scripts walks down data subfolders is specific to nested folder structures

> SVM.py is designed to train, evaluate and save a SVM predictor for classification
