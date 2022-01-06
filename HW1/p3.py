#!/usr/bin/env python3
import cv2
from p1n2 import *
import os


def get_attr(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_image = binarize(gray_image, 128)
    labeled_image = label(binary_image)
    attribute_list = get_attribute(labeled_image)

    return attribute_list

def best_match(object_database, test_object):
    '''

    Args:
        object_database: a list training images and each training image is stored as dictionary with keys name and image
        test_object: test image, a 2D unit8 array

    Returns:
        object_names: a list of filenames from object_database whose patterns match the test image
        You will need to use functions from p1n2.py
    '''

    test_attr = get_attr(test_object)
    match = []

    temp_features = []
   
    
    for obj in object_database:
        invert = 255 - (obj["image"])
        train_attr = get_attr(invert)
        temp_features.append({'name': obj['name'], 'roundedness': train_attr[0]['roundedness']})
    
    
    for o in test_attr:
        for obj in temp_features:
            if abs(o['roundedness'] - obj['roundedness']) <= 0.01:
                match.append(obj["name"])
    
    match = set(match)
    return list(match)


def main(argv):
  img_name = argv[0]
  test_img = cv2.imread('test/' + img_name + '.png', cv2.IMREAD_COLOR)

  train_im_names = os.listdir('train/')
  object_database = []
  for train_im_name in train_im_names:
      train_im = cv2.imread('train/' + train_im_name, cv2.IMREAD_COLOR)
      object_database.append({'name': train_im_name, 'image':train_im})
  
  cv2.imshow("test", test_img)
  object_names = best_match(object_database, test_img)
  print(object_names)


if __name__ == '__main__':
  main(sys.argv[1:])

# example usage: python p3.py many_objects_1.png
