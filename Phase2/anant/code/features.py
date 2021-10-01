import numpy as np
import math

from skimage.feature import local_binary_pattern, hog
from skimage.transform import resize

# Divides image into 8x8 blocks and returns an array with 64x64 array containing segments
def divide_image(image, rows, cols):
    return (np.reshape(np.reshape(image, (rows, cols, -1, 8)).swapaxes(1, 2), (-1, 8, 8)))

# Function which fetches all features
def get_all_features(image):
    segmented_image = divide_image(image, 8, 8)
    mean = get_mean(segmented_image)
    sd = get_sd(segmented_image, mean)
    skew = get_skew(segmented_image, mean)
    elbp = get_elbp(image)
    hog = get_hog(image)
    return {"mean": mean, "sd": sd, "skew": skew, "elbp": elbp, "hog": hog}

# Gets the mean for color moments
def get_mean(image):
    xaxes, yaxes, zaxes = np.shape(image)
    mean = []
    for x in range(xaxes):
        sum = 0
        for y in range(yaxes):
            for z in range(zaxes):
                inner_val = image[x][y][z]
                sum = sum + inner_val

        outer_val = sum/(yaxes * zaxes)
        mean.append(outer_val)
    return mean

# Gets the standard deviation for color moments
def get_sd(image, mean):
    xaxes, yaxes, zaxes = np.shape(image)
    sd = []
    for x in range(xaxes):
        sum = 0
        for y in range(yaxes):
            for z in range(zaxes):
                inner_val = ((image[x][y][z] - mean[x]) ** 2)
                sum = sum + inner_val

        outer_val = math.sqrt(sum/(yaxes * zaxes))
        sd.append(outer_val)
    return sd

# Gets skewness for color moments
def get_skew(image, mean):
    xaxes, yaxes, zaxes = np.shape(image)
    skew = []
    for x in range(xaxes):
        sum = 0
        for y in range(yaxes):
            for z in range(zaxes):
                inner_val = ((image[x][y][z] - mean[x]) ** 3)
                sum = sum+inner_val

        outer_sum_val = sum/(yaxes * zaxes)
        multiplier = 1
        if(outer_sum_val < 0):
            multiplier = -1
        outer_val = math.pow((multiplier * outer_sum_val), (1/3))
        skew.append(multiplier * outer_val)
    return skew

# Gets ELBP features as a array
def get_elbp(image):
    p = 24
    r = 3
    e = 1e-7
    lbp = local_binary_pattern(image, p, r, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, p + 3), range=(0, p + 2))
    hist = hist.astype("float")
    hist = hist/(hist.sum()+e)

    return hist

# Gets the HOG features of image
def get_hog(image):
    resized_image = resize(image,(128,64))
    value = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(
        2, 2), visualize=False, multichannel=False, block_norm="L2-Hys")
    
    return value