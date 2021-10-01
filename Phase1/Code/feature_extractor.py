import numpy
import numpy as np
import scipy.stats

import display_utils
import utils
from skimage.feature import local_binary_pattern, hog

# https://en.wikipedia.org/wiki/Color_moments
class ColorMoment:
    moment1 = None
    moment2 = None
    moment3 = None

    def __init__(self, image, block_size):
        #print(image.shape)
        #print("manual mean=",np.mean(image[0:8,0:8]))
        #print("manual std=", np.std(image[0:8, 8:16]))
        self.blocks = utils.array2d_to_blocks(image, (block_size, block_size))
        #print("blocks shape = ",self.blocks.shape)
        self.flattened_blocks = self.blocks.reshape(self.blocks.shape[0],self.blocks.shape[1]*self.blocks.shape[2])

    def __str__(self):
        return "ColorMoment[ moment1= "+str(self.first_moment())+" \nmoment2= "+str(self.second_moment())+" \nmoment3= "+str(self.third_moment())+"]"

    #mean
    def first_moment(self):
        if self.moment1 is not None:
            return self.moment1

        mean = np.mean(self.flattened_blocks, axis=1)
        self.moment1 = mean
        # mean = np.mean(self.blocks,axis=)
        # print("mean=",mean)
        # print(mean.shape)
        return mean

    #standard deviation
    #The second color moment is the standard deviation, which is obtained by taking the square root of the variance of the color distribution. sqrt(sigma(pixel-mean)^2)
    def second_moment(self):
        if self.moment2 is not None:
            return self.moment2
        sd = np.std(self.flattened_blocks, axis=1)
        self.moment2 = sd
        # print("standard deviation=",sd)
        # print(sd.shape)
        return sd

    #skewness
    # The third color moment is the skewness. It measures how asymmetric the color distribution is, and thus it gives information about the shape of the color distribution.
    #Skewness can be computed with the following formula: cuberoot(sigma(pixel-mean)^3)
    def third_moment(self):
        if self.moment3 is not None:
            return self.moment3
        skew = scipy.stats.skew(self.flattened_blocks, axis=1)
        self.moment3 = skew
        # print("skewness=", skew)
        # print(skew.shape)
        return skew

    def combined_feature(self):
        return np.array([self.first_moment(),self.second_moment(),self.third_moment()]).flatten()
'''
When surrounding pixels are all black or all white, then that image region is flat (i.e. featureless). 
Groups of continuous black or white pixels are considered “uniform” patterns that can be interpreted as corners or edges. 
If pixels switch back-and-forth between black and white pixels, the pattern is considered “non-uniform”.
'''
# Library - https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern
'''
Documentation
skimage.feature.local_binary_pattern(image, P, R, method='default')

image(N, M) array
Graylevel image.

P int
Number of circularly symmetric neighbour set points (quantization of the angular space).

R float
Radius of circle (spatial resolution of the operator).

method{‘default’, ‘ror’, ‘uniform’, ‘var’}
Method to determine the pattern.

‘default’: original local binary pattern which is gray scale but not
rotation invariant.

‘ror’: extension of default implementation which is gray scale and
rotation invariant.

‘uniform’: improved rotation invariance with uniform patterns and
finer quantization of the angular space which is gray scale and rotation invariant.

‘nri_uniform’: non rotation-invariant uniform patterns variant
which is only gray scale invariant [2].

‘var’: rotation invariant variance measures of the contrast of local
image texture which is rotation but not gray scale invariant.
'''
#Gray Scale and Rotation Invariant Texture Classification with Local Binary Patterns
#Ojala, T. Pietikainen, M. Maenpaa, T. Lecture Notes in Computer Science (Springer) 2000, ISSU 1842, pages 404-420
# https://www.researchgate.net/publication/221303862_Gray_Scale_and_Rotation_Invariant_Texture_Classification_with_Local_Binary_Patterns


class ExtendedLocalBinaryPattern:
    lbp_feature = None
    METHOD = ['default', 'ror', 'uniform', 'var']

    def __init__(self, image, radius=3, n_points=9, method='uniform', window_size = 8):
        assert method in self.METHOD
        self.image = image
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.window_size = 8

    def lbp(self):
        if self.lbp_feature is not None:
            return self.lbp_feature
        self.lbp_feature = local_binary_pattern(self.image, self.n_points, self.radius, self.method)
        #print("lbp_feature_shape=",self.lbp_feature.shape)
        #print("lbp_feature=",self.lbp_feature)
        lbp_feature_blocks = utils.array2d_to_blocks(self.lbp_feature, (self.window_size, self.window_size))

        hists = []
        for i in range(lbp_feature_blocks.shape[0]):
            hist = np.histogram(lbp_feature_blocks[i, :].ravel(), bins=np.arange(0, self.n_points + 3),
                         range=(0, self.n_points + 2))[0]
            hist = np.asarray(hist, dtype=float)
            # hist = hist.astype("float")
            hist = hist/hist.sum()
            hists.append(hist)
        hists = np.array(hists)
        #print(hists.shape)
        self.lbp_feature = hists
        #print("lbp_feature=",self.lbp_feature)
        return self.lbp_feature

'''
Compute a Histogram of Oriented Gradients (HOG) by
(optional) global image normalization
computing the gradient image in row and col
computing gradient histograms
normalizing across blocks
flattening into a feature vector
'''

'''
Documentation - 
skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', 
visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None, *, channel_axis=None)

out(n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray
HOG descriptor for the image. If feature_vector is True, a 1D (flattened) array is returned.

hog_image(M, N) ndarray, optional
A visualisation of the HOG image. Only provided if visualize is True.
'''
'''
Dalal, N and Triggs, B, Histograms of Oriented Gradients for Human Detection, IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2005 San Diego, CA, USA, https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf, DOI:10.1109/CVPR.2005.177
'''

class HOG:

    feature = None
    plot_hog = None

    def __init__(self, image, cell_size=8, block_size=1, block_norm='L2-Hys'):
        self.image = image
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_norm = block_norm

    def hog_feature(self):
        if self.feature is not None:
            return self.feature
        self.feature, self.plot_hog = hog(self.image, pixels_per_cell=(self.cell_size,self.cell_size),
                                    visualize=True, feature_vector=False,cells_per_block=(self.block_size,self.block_size),
                                          block_norm=self.block_norm)
        #print("hog feature shape=",self.feature.shape)
        self.feature = np.squeeze(self.feature)
        new_shape = (self.feature.shape[0]*self.feature.shape[1], self.feature.shape[2])
        self.feature = self.feature.reshape(new_shape)
        #self.feature = self.feature.flatten(order='A')
        #print("hog_shape new=",self.feature.shape)
        #print("hog_feature = ",self.feature)
        #print("hog im=",im, "min=", np.min(im), "max=",np.max(im))
        #im_list = [image,im]
        #display_utils.display_images(im_list)
        #print("hog feature=",self.feature)
        #print("hog feature shape",self.feature.shape,"min=",np.min(self.feature),"max=",np.max(self.feature))
        return self.feature

    def hog_viz_image(self):
        if self.plot_hog is not None:
            return self.plot_hog
        self.hog_feature()
        return self.plot_hog


class FeatureList:

    color_moment_1 = []
    color_moment_2 = []
    color_moment_3 = []
    hog_feature = []
    elbp_feature = []

    def __init__(self, image, id):
        self.image = image
        self.id = id

    def __repr__(self):
        return "\n\nFeatureList[ id="+str(self.id) +" image="+str(self.image)+"\ncolor_moment_1= "+str(self.color_moment_1)+" \ncolor_moment_2= "+str(self.color_moment_2)+\
               " \ncolor_moment_3= "+str(self.color_moment_3)+"\nhog_feature="+str(self.hog_feature)+"\nelbp_feature="+str(self.elbp_feature)+"]\n\n"

    def __str__(self):
        return "\n\nFeatureList[ id="+str(self.id) +" image="+str(self.image)+"\ncolor_moment_1= "+str(self.color_moment_1)+" \ncolor_moment_2= "+str(self.color_moment_2)+\
               " \ncolor_moment_3= "+str(self.color_moment_3)+"\nhog_feature="+str(self.hog_feature)+"\nelbp_feature="+str(self.elbp_feature)+"]\n\n"

    def add_color_moment_feature(self):
        self.color_moment_1 = ColorMoment(self.image,8).first_moment()
        self.color_moment_2 = ColorMoment(self.image,8).second_moment()
        self.color_moment_3 = ColorMoment(self.image,8).third_moment()

    def add_hog_feature(self):
        self.hog_feature = HOG(self.image).hog_feature()

    def add_elbp(self):
        self.elbp_feature = ExtendedLocalBinaryPattern(self.image).lbp()

    def add_default_features(self):
        self.add_color_moment_feature()
        self.add_elbp()
        self.add_hog_feature()