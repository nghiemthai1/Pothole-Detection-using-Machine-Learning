import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt

model = Sequential()
model = load_model('./models/128x3x10000/93PercentModel4ep100/my_model3conlayer.h5')
image_size = [128, 128, 3]
#Get it from a smaller dataset
X_test = np.load('./models/trainData/128x3x10000/X_test.npy')
y_test = np.load('./models/trainData/128x3x10000/y_test.npy')

import numpy as np
import matplotlib.pyplot as plt
index = 0
misclassifiedIndexes = []
predict = model.predict(X_test)
predict[predict < 0.5] = 0
predict[predict >= 0.5] = 1

fig = plt.figure()
fig.suptitle("Good Road' misclassified as 'Bad Road'", fontsize=16)
for label, predict_ in zip(y_test, predict):
    #if (label != predict_).any():
    if (label[0] != predict_[0] and label[0] == 1): #only good road misclassify
        misclassifiedIndexes.append(index)
    index +=1
print("{}: {}".format("'Good Road' misclassified as 'Bad Road'", len(misclassifiedIndexes)))
for i in range(25):
    plt.subplot(5, 5, i + 1)  # dont take 0
    im = X_test[misclassifiedIndexes[i]]
    plt.imshow(np.reshape(im, image_size), interpolation="nearest")

index =0
misclassifiedIndexes = []
for label, predict_ in zip(y_test, predict):
    #if (label != predict_).any():
    if (label[1] != predict_[1] and label[1] == 1): #only pot hole road misclassify
        misclassifiedIndexes.append(index)
    index +=1

fig = plt.figure(2)
fig.suptitle("'Bad Road' misclassified 'Good Road'", fontsize=16)
for i in range(25):
    plt.subplot(5, 5, i + 1)  # dont take 0
    im = X_test[misclassifiedIndexes[i]]
    plt.imshow(np.reshape(im, image_size), interpolation="nearest")

print("{}: {}".format("'Bad Road' misclassified 'Good Road'", len(misclassifiedIndexes)))

plt.show()