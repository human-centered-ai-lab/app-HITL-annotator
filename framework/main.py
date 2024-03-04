from classification import resize_and_prepare_images
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from classification import HogTransformer, RGB2GrayTransformer, consult_primitive_classifier, consult_primitive_classifiers_with_majority_vote
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer

width = 64
height = 64
sep_low = 0.3
sep_up = 0.8
spp_low = 0.8
spp_up = 0.98

include = ['River', 'SeaLake', 'Forest', 'Residential']

#resize_and_prepare_images('./images', 64, 64, include)

data = joblib.load(f'images_{width}x{width}px.pkl')
 
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
    random_state=42,
)

y_train = consult_primitive_classifiers_with_majority_vote(y_train, np.unique(data['label']), sep_low, sep_up, 42, 5)

grayify = RGB2GrayTransformer()

hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)

scalify = StandardScaler()

# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_prepared, y_train)

X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

y_pred = sgd_clf.predict(X_test_prepared)
#print(np.array(y_pred == y_test)[:25])
print('')
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))