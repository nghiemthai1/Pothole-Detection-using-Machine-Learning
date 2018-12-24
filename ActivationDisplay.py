import numpy as np
from keras.models import Sequential
from keras.models import load_model
from matplotlib import pyplot as plt

#I = ndimage.imread('C:/Users/nghiemthai1/Documents/MATLAB/MachineLearning1/Python/Pothole Detection/dataset1_04/image_1523381532.42.png')
def get_models_first_conv(fname):
    model = load_model(fname)

    model.summary()

    model.pop()  # Get rid of the classification layer
    model.pop()  # Get rid of dense_3
    model.pop()  # Get rid of act_5
    model.pop()  # Get rid of drop_out3
    model.pop()  # Get rid of dense_2
    model.pop()  # Get rid of activation_4
    model.pop()  # Get rid of drop_out_2
    model.pop()  # Get rid of dense_1
    model.pop()  # Get rid of activation_3
    model.pop()  # Get rid of drop_out1
    model.pop()  # Get rid of global
    model.pop()  # Get rid of conv2d_3
    # model.pop()  # Get rid of activation_2
    # model.pop()  # Get rid of conv2d_2
    # model.pop()  # Get rid of activation_1

    return model

def get_models_second_conv(fname):
    model = load_model(fname)
    model.pop()  # Get rid of the classification layer
    model.pop()  # Get rid of dense_3
    model.pop()  # Get rid of act_5
    model.pop()  # Get rid of drop_out3
    model.pop()  # Get rid of dense_2
    model.pop()  # Get rid of activation_4
    model.pop()  # Get rid of drop_out_2
    model.pop()  # Get rid of dense_1a
    model.pop()  # Get rid of activation_3
    model.pop()  # Get rid of drop_out1
    # model.pop()  # Get rid of global
    # model.pop()  # Get rid of conv2d_3
    # model.pop()  # Get rid of activation_2
    return model

def get_models_third_conv(fname):
    model = load_model(fname)
    model.pop()  # Get rid of the classification layer
    model.pop()  # Get rid of dense_3
    model.pop()  # Get rid of act_5
    model.pop()  # Get rid of drop_out3
    model.pop()  # Get rid of dense_2
    model.pop()  # Get rid of activation_4
    model.pop()  # Get rid of drop_out_2
    model.pop()  # Get rid of dense_1
    model.pop()  # Get rid of activation_3
    model.pop()  # Get rid of drop_out1
    model.pop()  # Get rid of global
    return model

fileName = './models/128x72x3x10000/90PercentModel3ep90/my_model3conlayer.h5'
model = Sequential()

X_train = np.load('./models/trainData/128x72x3x10000/X_train.npy')
hole = X_train[125] #100 is clean road, 125 is road with pot hole #125 good road, 123 pothole
print(hole.shape)
plt.figure(1)
#plt.imshow(np.reshape(hole,[128,128]), interpolation="nearest", cmap="gray")
plt.imshow(hole, interpolation="nearest")
model = load_model(fileName)
hole_batch = np.expand_dims(hole,axis=0)
print(model.predict(hole_batch))

model = get_models_first_conv(fileName)
model.summary()

model1 = Sequential()
model2 = get_models_second_conv(fileName)
model2.summary()

# model3 = Sequential()
# model3 = get_models_third_conv(fileName)


hole_conv = model.predict(hole_batch)
hole_conv = np.squeeze(hole_conv, axis=0)
print(hole_conv.shape)
# plt.imshow(hole_conv[:, :, 8], cmap='gray')
# plt.show()
plt.figure(2)
for i in range(16): #0 to 16
    plt.subplot(4, 4, i+1) #dont take 0
    plt.imshow(hole_conv[:, :, i], cmap="jet") # take 0 plt.imshow(hole_conv[:, :, :, i], cmap="jet")
hole_conv = model2.predict(hole_batch)
hole_conv = np.squeeze(hole_conv, axis=0)
print(hole_conv.shape)
plt.figure(3)
for i in range(32): #0 to 16
    plt.subplot(8, 4, i+1) #dont take 0
    plt.imshow(hole_conv[:, :, i], cmap="jet") # take 0

# hole_conv = model3.predict(hole_batch)
# hole_conv = np.squeeze(hole_conv, axis=0)
# print(hole_conv.shape)
# plt.figure(4)
# for i in range(64): #0 to 16
#     plt.subplot(8, 8, i+1) #dont take 0
#     plt.imshow(hole_conv[:, :, i], cmap='gray') # take 0

plt.show()