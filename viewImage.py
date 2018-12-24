import cv2, glob
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from skimage import exposure
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def rotateImage(img, angle):
    (rows, cols, ch) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def loadBlurImg(path, imgSize):
    img = cv2.imread(path)
    angle = np.random.randint(-180, 180)
    img = rotateImage(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, imgSize)
    return img


def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []

    for path in classPath:
        img = loadBlurImg(path, imgSize)
        x.append(img)
        y.append(classLable)

    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)
    return x, y


def loadData(img_sizeX, img_sizeY,  classSize):
    goodRoad = glob.glob('./dataset1_01/1/image_1523380110.35.png', recursive=True)
    pothole = glob.glob('./dataset1_04/image_1523381521.21.png', recursive=True)

    imgSize = (img_sizeX, img_sizeY)
    xHotdog, yHotdog = loadImgClass(goodRoad, 0, classSize, imgSize)
    xNotHotdog, yNotHotdog = loadImgClass(pothole, 1, classSize, imgSize)
    print("There are", len(xHotdog), "good road images")
    print("There are", len(xNotHotdog), "bad road images")

    X = np.array(xHotdog + xNotHotdog)
    y = np.array(yHotdog + yNotHotdog)
    # y = y.reshape(y.shape + (1,))
    return X, y

def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html

    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    #images = images.reshape(images.shape + (1,))
    return images


def preprocessData(images):
    # grayImages = toGray(images)
    # return normalizeImages(grayImages)

    return normalizeImages(images)

sizeX = 128
sizeY = 72
classSize = 1
scaled_X, y = loadData(sizeX, sizeY, classSize)

n_classes = len(np.unique(y))
print("Number of classes =", n_classes)

scaled_X = preprocessData(scaled_X)

hole = scaled_X[1] #100 is clean road, 125 is road with pot hole #125 good road, 123 pothole
print(hole.shape)
plt.figure(1)
#plt.imshow(np.reshape(hole,[128,128]), interpolation="nearest", cmap="gray")
plt.imshow(hole, interpolation="nearest")

good = scaled_X[0] #100 is clean road, 125 is road with pot hole #125 good road, 123 pothole
print(good.shape)
plt.figure(2)
#plt.imshow(np.reshape(hole,[128,128]), interpolation="nearest", cmap="gray")
plt.imshow(good, interpolation="nearest")

plt.show()