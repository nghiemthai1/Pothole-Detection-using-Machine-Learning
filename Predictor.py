import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage import exposure


model = Sequential()
model = load_model('./models/128x72x3x10000/90PercentModel3ep90/my_model3conlayer.h5')

X_test = np.load('./models/trainData/128x72x3x10000/X_test.npy')
y_test = np.load('./models/trainData/128x72x3x10000/y_test.npy')

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))