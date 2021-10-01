import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
from skimage.draw import rectangle_perimeter
import constants

matplotlib.rc('font', size=16)

def display_images(images : list):
    rows = 2
    columns = math.ceil(len(images)/2)
    fig = plt.figure()
    for index, image in enumerate(images):
        fig.add_subplot(rows, columns, index+1)
        plt.imshow(image)
    plt.show()

def display_similar_images(query_image, similar_images:list, query_image_data, similar_images_data:list, save_file_name):
    assert len(similar_images)==len(similar_images_data)
    rows = 1 + math.ceil(len(similar_images)/3)
    columns = 3
    fig = plt.figure()
    fig.set_size_inches(24, 24)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(query_image)
    plt.title("Query Image | ID="+str(query_image_data))

    for index, image in enumerate(similar_images):
        fig.add_subplot(rows, columns, columns+1+index)
        fig.tight_layout(h_pad=3,w_pad=12)
        plt.imshow(image)
        plt.title("ID= "+str(similar_images_data[index][0])+" | Similarity score=" + str(similar_images_data[index][1]))
    #plt.show()
    plt.savefig(constants.output_dir+save_file_name,dpi=100)

def draw(image,start_coord, end_coord,val=0.1):

    rr,cc = rectangle_perimeter(start_coord, end=end_coord, shape=image.shape,clip=True)
    image[rr,cc] = val
    return image
    #display_images([image])
