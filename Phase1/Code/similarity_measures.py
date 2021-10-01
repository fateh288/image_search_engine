import numpy as np
import math
from feature_extractor import ColorMoment
from scipy.stats import chisquare

# https://en.wikipedia.org/wiki/Color_moments
def color_moment_similarity(color_moment_image1: ColorMoment, color_moment_image2: ColorMoment, w1=0.34, w2=0.33, w3=0.33):
    val1 = np.abs(color_moment_image1.first_moment() - color_moment_image2.first_moment())
    #print("val1=",val1,"mean val1=",np.mean(val1))
    val2 = np.abs(color_moment_image1.second_moment() - color_moment_image2.second_moment())
    #print("val2=",val2,"mean val2=",np.mean(val2))
    val3 = np.abs(color_moment_image1.third_moment() - color_moment_image2.third_moment())
    #print("val3=", val3, "mean val3=", np.mean(val3))
    return round(w1*np.mean(val1) + w2*np.mean(val2) + w3*np.mean(val3),3)

def color_moment_feature_similarity(color_moment1a, color_moment1b, color_moment2a,
                            color_moment2b, color_moment3a, color_moment3b,
                            w1 = 0.34, w2=0.33, w3=0.33):
    val1 = np.abs(color_moment1a - color_moment1b)
    val2 = np.abs(color_moment2a - color_moment2b)
    val3 = np.abs(color_moment3a - color_moment3b)
    return round(w1 * np.mean(val1) + w2 * np.mean(val2) + w3 * np.mean(val3), 3)

# li=1 ---> L1 Norm
# li=2 ---> L2 Norm
# li=p ---> Lp Norm
def l_norm_similarity(vector1:np.ndarray, vector2:np.ndarray, li=2):
    assert vector1.shape == vector2.shape and vector1.ndim == 1 and li >= 1
    sm = np.sum(np.power(np.abs(vector1-vector2),li))
    return math.pow(sm, 1/float(li))

def chi_square(vector1:np.ndarray, vector2:np.ndarray):
    assert vector1.shape == vector2.shape and vector1.ndim == 1
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(vector1, vector2) if not (a==0 and b==0)])
    return chi