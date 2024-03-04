import joblib
from skimage.io import imread
from skimage.transform import resize
import os
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import skimage
from skimage.feature import hog
import random
from collections import Counter


def resize_and_prepare_images(src, width, height, include):
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    
    pklname = f"images_{width}x{height}px.pkl"

    for subdir in os.listdir(src):
        if not subdir in include: continue
        print(subdir)
        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg'}:
                im = imread(os.path.join(current_path, file))
                im = resize(im, (width, height)) #[:,:,::-1]
                data['label'].append(subdir)
                data['filename'].append(file)
                data['data'].append(im)
        
        joblib.dump(data, pklname)

def consult_primitive_classifier(y_train, labels, accuracy, random_state):
    random.seed(random_state)
    labeled = []
    for sample in y_train:
        rand = random.uniform(0, 1.0)
        if rand <= accuracy: 
            labeled.append(sample)
        else: 
            possible_labels = [x for x in labels if x != sample]
            labeled.append(random.choice(possible_labels))
    return labeled

def consult_primitive_classifiers_with_majority_vote(y_train, labels, accuracy_low, accuracy_up, random_state, amount):
    random.seed(random_state)
    accuracies = []
    for _ in range(amount): accuracies.append(random.uniform(accuracy_low, accuracy_up))
    #print(accuracies)
    results = []
    for x in accuracies: results.append(consult_primitive_classifier(y_train, labels, x, (x * random_state)))
    labels = []
    for result in zip(*results): labels.append(majority_vote(result))
    return labels

def majority_vote(data):
    counter = Counter(data)
    return counter.most_common(1)[0][0]



    
 
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])