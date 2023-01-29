import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from collections import namedtuple

Annotation = namedtuple('Annotation', ['idx', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax'])

def read_annotations():
    # specify the path to the XML file
    file_path = 'datasets/cars_labelimg/0.xml'

    # parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # extract the image filename
    filename = root.find('filename').text
    print('Filename:', filename)

    # extract the image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    print('Width:', width)
    print('Height:', height)

    # extract the object class and location
    for obj in root.iter('object'):
        name = obj.find('name').text
        print('Object class:', name)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        print('Location:', xmin, ymin, xmax, ymax)

def display():
    annotations = []
    # specify the path to the XML file
    file_path = 'datasets/cars_dataset/cars_labelimg/0.xml'

    # parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # extract the image filename
    filename = root.find('filename').text
    print('Filename:', filename)

    # extract the image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    print('Width:', width)
    print('Height:', height)

    # extract the object class and location
    for idx, obj in enumerate(root.iter('object')):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        print('idx', idx, 'Object class:', name, 'Location:', xmin, ymin, xmax, ymax)
        annotations.append(Annotation(idx, name, xmin, ymin, xmax, ymax))

    # display the image
    image = plt.imread('datasets/cars_dataset/cars/0.jpg')

    # reshape the image and annotations
    new_width = 300
    new_height = 300
    image, annotations = reshape_image(image, annotations, new_width, new_height)

    # display the bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for annotation in annotations:
        xmin, ymin, xmax, ymax = annotation.xmin, annotation.ymin, annotation.xmax, annotation.ymax
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.axis("off")
    plt.show()

# function to reshape the image and annotations
def reshape_image(image, annotations, new_width, new_height):
    new_annotations = []
    # get the image dimensions
    height, width, _ = image.shape

    print('new_width:', new_width)
    print('new_height:', new_height)
    # reshape the image
    image = cv2.resize(image, (new_width, new_height))

    # reshape the annotations
    ratio_width = new_width / width
    ratio_height = new_height / height

    for idx, annotation in enumerate(annotations):

        class_name = annotation.class_name
        xmin = int(annotation.xmin * ratio_width)
        ymin = int(annotation.ymin * ratio_height)
        xmax = int(annotation.xmax * ratio_width)
        ymax = int(annotation.ymax * ratio_height)
        print('New Location:', xmin, ymin, xmax, ymax)
        new_annotations.append(Annotation(idx, class_name, xmin, ymin, xmax, ymax))

    return image, new_annotations

if __name__ == '__main__':
    display()