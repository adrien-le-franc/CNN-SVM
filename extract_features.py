"""
script designed to extract features calculated with TensorFlow's inceptionv3
from an image data base.

paths to models' .pb file and image data base parent folder should be modified

nb_images states how many images is in the data base and can be found entering
UNIX "find . -type f -ls | wc -l" in parent folder (make sure sub folders
only contain images)

"""

import os
import csv
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import argparse

# paths to your own file (to be adapted)
inception_path = "../../../inceptionv3/tensorflow_inception_graph.pb"
path_to_parent_folder = "../luggage_case/travel_accessoires"

nb_images = 396

# names of files returned by the programm
new_features = "features4.txt"
new_labels = "labels4.txt"
label_dict = "label_dict4.csv"

# dictionaries to label clusters of the data base, will be saved in a csv file
dictionary = {}


def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.

    """
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(data_path, nb_images, verbose=False):
    """
    extract_features computed inceptionv3 bottleneck feature for a list of images

    returns: 2-d array in the shape of (len(image_paths), 2048)
    
    """
    feature_dimension = 2048

    # to be returned containing images features and labels, labels stored in
    # a 1D array to match sklearn expectations
    features = np.empty((nb_images, feature_dimension))
    labels = np.zeros((nb_images,1))

    with tf.Session() as sess:
        # takes features out of pool_3:0 layer
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        index = -1
        cluster = -1
        for root, dirs, files in os.walk(path_to_parent_folder):
            cluster+=1
            st = str(root).split("/")[-1]
            if cluster>0:
                dictionary[st]=cluster
            print(dictionary)
            for name in files:
                index+=1
                if name.endswith((".jpg", ".jpeg")) and index < nb_images:

                    # fill labels
                    labels[index,0] = cluster

                    # extract features
                    image_path = root +"/"+ name

                    if verbose:
                        print("completed {}%".format(index/nb_images*100))

                    if not gfile.Exists(image_path):
                        tf.logging.fatal('File does not exist %s', image)

                    image_data = gfile.FastGFile(image_path, 'rb').read()
                    feature = sess.run(flattened_tensor, {
                        'DecodeJpeg/contents:0': image_data
                    })

                    # fill labels
                    features[index, :] = np.squeeze(feature)

    # save into txt file
    np.savetxt(new_features,features)
    np.savetxt(new_labels,labels)

    # save dict in csv file
    w = csv.writer(open(label_dict, "w"))
    for key, val in dictionary.items():
        w.writerow([key, val])

    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="computes features")
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-n', '--nb-images', type=int, required=True)
    args = parser.parse_args()

    create_graph(args.model)
    X, Y = extract_features(args.data, args.nb_images, verbose=True)
