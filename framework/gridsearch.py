from framework import Annotator, HITLAnnotator
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread, imsave
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

width = 64
height = 64
acc_low = 0.75
acc_high = 0.99
a = [0.05, 0.1, 0.15]
prob_high = [0.80, 0.75, 0.65, 0.99]
prob_low = [0.30, 0.4, 0.5, 0.65]
mode = 'multi'
min_annotators = [1,3]
max_annotators = [5,7]


data = joblib.load(f'images_{width}x{width}px.pkl')

print('')
print('---------------------------------')
print('number of samples: ', len(data['data']))
#print('keys: ', list(data.keys()))
#print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))

X = np.array(data['data'])
y = np.array(data['label'])
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
)

if not os.path.exists('tmp'): os.mkdir('tmp')
uint_img = skimage.util.img_as_ubyte(X[0])
imsave(f"tmp/tmp.png", uint_img)
exit(0)

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

def try_with_params(acc_l, acc_h, a, prob_h, prob_l, mina, maxa):
    annotators = []
    for x in range(20):
        ann = Annotator(str(x), 'primitive', limit=1000, accuracy_low=acc_l, accuracy_high=acc_h, random_state=None)
        annotators.append(ann)

    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3, loss='log_loss')

    hitl = HITLAnnotator(annotators, a, None, prob_h, prob_l, mode, sgd_clf, X_train_prepared, np.unique(data['label']), mina, maxa, X_test_prepared, y_test)
    classifier = hitl.train_classifier_with_human_in_the_loop(y_train)
    percentage = test_classifier(classifier)
    return percentage


def test_classifier(classifier):
    y_pred = classifier.predict(X_test_prepared)
    result = 100*np.sum(y_pred == y_test)/len(y_test)
    return result


def base_case():
    print('')
    print('---------------------------------')
    print('BASE CASE')

    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_prepared)
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

def try_params():
    best = None
    for iter_a in a:
        for iter_prob_high in prob_high:
            for iter_prob_low in prob_low:
                for iter_mina in min_annotators:
                    for iter_maxa in max_annotators:
                        print('params')
                        print([iter_a, iter_prob_low, iter_prob_high, iter_mina, iter_maxa])
                        result = try_with_params(acc_low, acc_high, iter_a, iter_prob_high, iter_prob_low, iter_mina, iter_maxa)
                        if best is None: best = [result, iter_a, iter_prob_low, iter_prob_high, iter_mina, iter_maxa]
                        elif best[0] < result: best = [result, iter_a, iter_prob_low, iter_prob_high, iter_mina, iter_maxa]
    print(best)

try_params()
base_case()
