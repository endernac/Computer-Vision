#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def normxcorr2(template, image):
  """Do normalized cross-correlation on grayscale images.

  When dealing with image boundaries, the "valid" style is used. Calculation
  is performed only at locations where the template is fully inside the search
  image.

  Args:
  - template (2D float array): Grayscale template image.
  - image (2D float array): Grayscale search image.

  Return:
  - scores (2D float array): Heat map of matching scores.
  """
  corr = np.zeros_like(image)

  rows, cols = image.shape         # image dimensions
  k1, k2 = template.shape          # length of kernel
  m1 = int((k1 - 1) / 2)
  m2 = int((k2 - 1) / 2)

  image_padded = np.zeros((rows + k1 - 1, cols + k2 - 1))
  image_padded[m1:-m1, m2:-m2] = image
  temp_var = np.sqrt((template**2).sum())

  for x in range(rows):
    for y in range(cols):
      arr_var = np.sqrt((image_padded[x: x+k1, y: y+k2]**2).sum())
      num = (template * image_padded[x: x+k1, y: y+k2]).sum()
      corr[x, y] = num / (arr_var * temp_var)

  return corr



def find_matches(template, image, thresh=None):
  """Find template in image using normalized cross-correlation.

  Args:
  - template (3D uint8 array): BGR template image.
  - image (3D uint8 array): BGR search image.

  Return:
  - coords (2-tuple or list of 2-tuples): When `thresh` is None, find the best
      match and return the (x, y) coordinates of the upper left corner of the
      matched region in the original image. When `thresh` is given (and valid),
      find all matches above the threshold and return a list of (x, y)
      coordinates.
  - match_image (3D uint8 array): A copy of the original search images where
      all matched regions are marked.
  """

  gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  temp = gray_temp / gray_temp.max()
  img = gray_img / gray_img.max()

  mats = normxcorr2(temp, img)
  mats = mats/mats.max()

  y, x = np.where(mats > thresh)
  k,_ = temp.shape
  half = int((k-1)/2)

  coords = [(x[i] - half, y[i] - half) for i in range(len(x))]
  cpy = np.copy(image)

  for c in coords:
    cv2.rectangle(cpy, c, (c[0] + k, c[1] + k), (255, 0, 0), 3)

  return coords, cpy




def main(argv):
  template_img_name = argv[0]
  search_img_name = argv[1]

  template_img = cv2.imread("data/" + template_img_name + ".png", cv2.IMREAD_COLOR)
  search_img = cv2.imread("data/" + search_img_name + ".png", cv2.IMREAD_COLOR)

  _, match_image = find_matches(template_img, search_img, 0.99)

  cv2.imwrite("output/" + search_img_name + ".png", match_image)


if __name__ == "__main__":
  main(sys.argv[1:])

# example usage: python p4.py face king
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
