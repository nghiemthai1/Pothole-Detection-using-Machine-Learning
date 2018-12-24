import numpy as np
import cv2
import glob, time
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage import exposure

model = Sequential()
model = load_model('my_model3conlayer.h5')

size = 128

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (size, size, 3)

start_time = time.time()

early = EarlyStopping(monitor='val_loss', patience = 10, verbose = 0)
model.fit(X_train, y_train, nb_epoch=50, callbacks=[early],validation_split=0.1)
metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

print("Saving model weights and configuration file")

model.save('my_model3conlayer.h5')
# serialize model to JSON
model_json = model.to_json()
with open("model3conlayer.json", "w") as json_file:
    json_file.write(model_json)

elapsed_time = time.time() - start_time
print (elapsed_time)

model.save_weights("modelWeight3conlayer.h5")
print("Saved model to disk")
