from framework import Annotator, HITLAnnotator
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
import skimage
import numpy as np
import os
import joblib
from skimage.transform import resize
from ui import Ui
import pickle

width = 64
height = 64
annotators = []
a = 0.02
prob_high = 0.54
prob_low = 0.50
mode = 'multi'
min_annotators = 1
max_annotators = 3

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


def unpickle_cifar(file, width, height):
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    with open(file, 'rb') as fo:
        loaded = pickle.load(fo, encoding='bytes')
        pixels = loaded[b'data']
        
        images = []
        for img in pixels:
            rs = img[0:1024]
            gs = img[1024:2048]
            bs = img[2048:3072]
            rgbs = zip(rs,gs,bs)
            for rgb in rgbs:
                r = rgb[0]
                g = rgb[1]
                b = rgb[2]
                images.append(r)
                images.append(g)
                images.append(b)
        data_np = np.array(images)
        print('shape')
        print(np.shape(data_np))
        data_np_reshaped = np.reshape(data_np, (10000 ,32, 32, 3))
        print(data_np)
        data['label'] = loaded[b'labels']
        data['data'] = data_np_reshaped
        data['filename'] = loaded[b'filenames']
    pklname = f"images_{width}x{height}px_cifar.pkl" 
    joblib.dump(data, pklname)
    return data
        

# this is to bring the images into the right format
# it is assumed that images are placed in folders that are named by the labels of the ground truth in the src directory
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

unpickle_cifar('data_batch_1', width, height)

# load the previously preprocessed images
#data = joblib.load(f'images_{width}x{width}px.pkl')
data = joblib.load(f'images_{width}x{width}px_cifar.pkl')

 
print('')
print('---------------------------------')
print('number of samples: ', len(data['data']))
#print('keys: ', list(data.keys()))
#print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))


# ui = Ui()
# ui.start(3001)
# print('ui started')

# create annotators and connect ui instance
# for x in range(20):
#     ann = Annotator(str(x), 'primitive', limit=1000, accuracy_low=0.85, accuracy_high=0.98, random_state=None)
#     ann = Annotator(str(x), 'function', limit=1000)
#     ui.register(str(x))
#     annotators.append(ann)

# create primitive annotators for simulation
for x in range(20):
    ann = Annotator(str(x), 'primitive', limit=1000, accuracy_low=0.85, accuracy_high=0.99)
    annotators.append(ann)

X = np.array(data['data'])
y = np.array(data['label'])
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
)

grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)
X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3, loss='log_loss')

# the function that gets called when expoert annotation is requested
# requesting annotation from UI possible
# def return_label(id, sample_X, sample_y, sample_prob, possible_labels):
#     print(f'Consulting annotator {id}')
#     result = ui.request_annotation(id, sample_X, possible_labels, sample_y)
#     return result

hitl = HITLAnnotator(annotators, a, None, prob_high, prob_low, mode, sgd_clf, X_train_prepared, np.unique(data['label']), min_annotators, max_annotators, X_test_prepared, y_test, X)
hitl.train_classifier_with_human_in_the_loop(y_train)

def base_case():
    print('')
    print('---------------------------------')
    print('BASE CASE')
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)

    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_prepared)
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

base_case()

