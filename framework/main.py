from classification import resize_and_prepare_images
import numpy as np
import joblib
import math
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from classification import HogTransformer, RGB2GrayTransformer, PrimitveAnnotator, consult_primitive_classifiers_with_majority_vote, get_label_from_probabilities
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from tqdm import tqdm

width = 64
height = 64
sep_low = 0.3
sep_up = 0.8
m = 10 # amount of annotators
k = 10
a = 0.1 # fraction of samples to use for initial annotation
random_state = 42
iterations = 10  

include = ['River', 'SeaLake', 'Forest', 'Residential']

#resize_and_prepare_images('./images', 64, 64, include)

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
    random_state=random_state,
)

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

    sgd_clf = SGDClassifier(random_state=random_state, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_prepared)
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))


base_case()

def annotation_case(output):
    annotators = [PrimitveAnnotator(sep_low, sep_up, random_state * (x + 1)) for x in range(m)]
    intial_amount = math.floor(len(X_train) * a)
    X_train_initial = X_train[:intial_amount]
    y_train_initial = consult_primitive_classifiers_with_majority_vote(y_train[:intial_amount], np.unique(data['label']), annotators)
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()
    X_train_gray = grayify.fit_transform(X_train_initial)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
    sgd_clf = SGDClassifier(random_state=random_state, max_iter=1000, tol=1e-3, loss='log_loss')
    sgd_clf.fit(X_train_prepared, y_train_initial)
    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)
    y_pred = sgd_clf.predict_proba(X_test_prepared)
    #y_pred = sgd_clf.predict(X_test_prepared)
    y_pred_labels, y_pred_probabilities = get_label_from_probabilities(y_pred.tolist(), np.unique(data['label']))
    #print(y_pred_labels)
    percentage = 100*np.sum(y_pred_labels == y_test)/len(y_test)
    if output:
        print('')
        print('---------------------------------')
        print('ANNOTATION CASE')
        print('Percentage correct: ', percentage)
        print('Consultations used: ', np.sum([x.get_consultations() for x in annotators]))
    return percentage

percentages = [annotation_case(True) for _ in tqdm(range(iterations), leave=True)]
print('')
print('---------------------------------')
print('ANNOTATION CASE')
print('Average Percentage Correct: ', np.average(percentages))