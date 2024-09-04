from ui import Ui
import time
from skimage.io import imread

ui = Ui()
ui.start(3001)
print('ui started')

ui.register('test')
time.sleep(10)
image = imread('tmp/tmp.png')
result = ui.request_annotation('test', image, ['label1', 'label2', 'label3'])
print('received result')
print(result)


