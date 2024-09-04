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
from tqdm import tqdm


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

# def consult_primitive_classifier(y_train, labels, accuracy, random_state):
#     random.seed(random_state)
#     labeled = []
#     for sample in y_train:
#         rand = random.uniform(0, 1.0)
#         if rand <= accuracy: 
#             labeled.append(sample)
#         else: 
#             possible_labels = [x for x in labels if x != sample]
#             labeled.append(random.choice(possible_labels))
#     return labeled

def consult_primitive_classifiers_with_majority_vote(y_train, labels, annotators):
    results = []
    for x in tqdm(annotators, desc='consulting primitive annotators', leave=False): results.append(x.annotate_multi(y_train, labels))
    labels = []
    for result in zip(*results): 
        label, majority = majority_vote(result)
        for x in zip(result, annotators): 
            annotator_label = x[0]
            annotator = x[1]
            if annotator_label == label: annotator.increase_correct(majority)
        labels.append(label)
    return labels

def majority_vote(data):
    counter = Counter(data)
    votes = counter.most_common(1)[0][1]
    majority = votes / len(data)
    return counter.most_common(1)[0][0], majority

class PrimitveAnnotator():
    def __init__(self, accuracy_low, accuracy_up, random_state):
        accuracy = random.uniform(accuracy_low, accuracy_up)
        self.accuracy = accuracy
        self.annotated = 0
        self.correct = 0
        self.random_state = random_state
    
    def get_accuracy(self):
        return self.accuracy

    def get_performance(self):
        return self.correct / self.annotated
    
    def get_consultations(self):
        return self.annotated
    
    def increase_correct(self, correct):
        self.correct = self.correct + correct
    
    def annotate_one(self, label, possible_labels):
        random.seed(self.random_state / (self.annotated + 1))
        rand = random.uniform(0.0, 1.0)
        self.annotated = self.annotated + 1
        if rand <= self.accuracy: return label
        else: return random.choice([x for x in possible_labels if x != label])
    
    def annotate_multi(self, labels, possible_labels):
        results = []
        for label in labels: results.append(self.annotate_one(label, possible_labels))
        return results
        

    
 
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
        

def get_label_from_probabilities(probabilities, labels):
    #print(probabilities)
    #print(labels)
    result = []
    label_probabilities = []
    for x in probabilities:
        #print(x)
        max = np.max(x)
        index = x.index(max)
        result.append(labels[index])
        label_probabilities.append(max)
    return result, label_probabilities
    

def normalize_probabilities(probabilities):
    max = np.max(probabilities)
    normalized = [(x / max) for x in probabilities]
    return normalized