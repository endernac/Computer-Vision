#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import scipy.stats as st
from scipy.signal import convolve2d


def binarize(gray_image, thresh_val):
    """ Function to threshold grayscale image to binary
        Sets all pixels lower than threshold to 0, else 255

        Args:
        - gray_image: grayscale image as an array
        - thresh_val: threshold value to compare brightness with

        Return:
        - binary_image: the thresholded image
    """
    rows, cols = gray_image.shape
    binary_image = gray_image.copy()

    for i in range(rows):
        for j in range(cols):
            if gray_image[i,j] > thresh_val:
                binary_image[i,j] = 255
            else:
                binary_image[i,j] = 0

    return binary_image



def label(binary_image):
    """ Function to labeled components in a binary image
        Uses a sequential labeling algorithm

        Args:
        - binary_image: binary image with multiple components to label

        Return:
        - lab_im: binary image with grayscale level as label of component
    """

    _, lab_im = cv2.connectedComponents(binary_image)
    return lab_im


def get_center(bin_img):
  y, x = np.where(bin_img > 0)
  return {'x': x.mean(), 'y': y.mean()}


def get_orientation(bin_img):
  y, x = np.where(bin_img > 0)

  x_r = x - x.mean()
  y_r = y - y.mean()

  a = (x_r**2).sum()
  b = (x_r * y_r).sum()
  c = (y_r**2).sum()

  return 0.5 * np.arctan2(2 * b, a - c)


def get_roundedness(bin_img):
  y, x = np.where(bin_img > 0)

  x_r = x - x.mean()
  y_r = y - y.mean()

  a = (x_r**2).sum()
  b = (x_r * y_r).sum()
  c = (y_r**2).sum()

  t_1 = 0.5 * np.arctan2(2 * b, a - c)
  t_2 = t_1 + np.pi/2

  E_min = a * np.sin(t_1) ** 2 - b * np.sin(t_1) * np.cos(t_1) + c * np.cos(t_1) ** 2
  E_max = a * np.sin(t_2) ** 2 - b * np.sin(t_2) * np.cos(t_2) + c * np.cos(t_2) ** 2

  return E_min / E_max



def get_attribute(labeled_image):
    """ Function to get the attributes of each component of the image
        Calculates the position, orientation, and roundedness

        Args:
        - labeled_image: image file with labeled components

        Return:
        - attribute_list: a list of the aforementioned attributes
    """

    attribute_list = []
    num_comps = labeled_image.max()
    
    for c in range(1, num_comps + 1):
      obj = ((labeled_image == c) * 255).astype(np.uint8)
      center = get_center(obj)
      orient = get_orientation(obj)
      rounded = get_roundedness(obj)

      attr = {'position': center, 'orientation': orient, 'roundedness': rounded}
      attribute_list.append(attr)

    return attribute_list



def draw_attributes(image, attribute_list):
    num_row = image.shape[0]
    attributed_image = image.copy()
    for attribute in attribute_list:
        center_x = (int)(attribute["position"]["x"])
        center_y = (int)(attribute["position"]["y"])
        slope = np.tan(attribute["orientation"])

        cv2.circle(attributed_image, (center_x, center_y), 2, (255, 0, 0), 2)
        cv2.line(
            attributed_image,
            (center_x, center_y),
            (center_x + 20, int(20 * (slope) + center_y)),
            (255, 0, 0),
            2)

        cv2.line(
            attributed_image,
            (center_x, center_y),
            (center_x - 20, int(-20 * (slope) + center_y)),
            (255, 0, 0),
            2)

    return attributed_image



def gkern(klen=7, ksig=2, ord_x=0, ord_y=0):
    """Takes kernel length, and sigma and returns one std 2D gaussian."""

    x1 = np.linspace(-ksig, ksig, klen+1)
    x2 = np.linspace(-ksig, ksig, klen+2)

    # get the precise values of the gaussian and its derivative
    g1 = np.diff(st.norm.cdf(x1))
    g2 = np.diff(np.diff(st.norm.cdf(x2)))

    dim1 = g1 if ord_x==0 else g2
    dim2 = g1 if ord_y==0 else g2

    kern = np.outer(dim1, dim2)
    return kern/kern.sum()


def suppress(mag, ang):
  max = np.zeros_like(mag)
  angle = ang * 180. / np.pi
  rows, cols = mag.shape

  for i in range(1,rows-1):
    for j in range(1,cols-1):
      if (0 <= angle[i,j] < 45) or (-180 <= angle[i,j] <= -135):
        a = mag[i, j+1]
        b = mag[i, j-1]
      elif (45 <= angle[i,j] < 90) or (-135 <= angle[i,j] < -90):
        a = mag[i+1, j-1]
        b = mag[i-1, j+1]
      elif (90 <= angle[i,j] < 135) or (-90 <= angle[i,j] < -45):
        a = mag[i+1, j]
        b = mag[i-1, j]
      elif (135 <= angle[i,j] < 180) or (-45 <= angle[i,j] < 0):
        a = mag[i-1, j-1]
        b = mag[i+1, j+1]

      max[i, j] = mag[i,j] if ((mag[i,j] >= a) and (mag[i,j] >= b)) else 0

  return max.astype(np.uint8)



def detect_edges(image, sigma, threshold):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
    - edge_image (2D binary image): each location indicates whether it belongs to an edge or not
  """

  dx = gkern(7, sigma, 1, 0)
  dy = gkern(7, sigma, 0, 1)
  
  I_x = convolve2d(image, dx, mode='same')
  I_y = convolve2d(image, dy, mode='same')

  mag = np.sqrt(I_x ** 2 + I_y ** 2)
  mag = (mag/mag.max() * 255)

  ang = np.arctan2(I_y, I_x).astype(np.float32)
  max = suppress(mag, ang)
  thresh = binarize(max / max.max(), threshold)

  return thresh


def get_len(edge_image, rho, theta, thresh):
  '''We multiply the point coordinates by the normal vector to find the offset
    we then count points with offsets close to the true offset for the line as
    belonging to the line. 
  '''

  points = np.argwhere(edge_image > 0)
  tran = np.array([np.sin(theta), np.cos(theta)])

  dist_points = np.sum(points * tran, axis=1)
  error = np.abs(dist_points - rho)
  close_points = np.where(error < thresh, 1, 0) 
  len_edge = close_points.sum()

  return len_edge


def get_obj_and_len(labeled_image, edge_image, rho, theta, thresh):
  '''We multiply the point coordinates by the normal vector to find the offset
    we then count points with offsets close to the true offset for the line as
    belonging to the line. Then we find the list of objects that the edge lies
    along.
  '''

  # get the edge points
  points = np.argwhere(edge_image > 0)
  tran = np.array([np.sin(theta), np.cos(theta)])

  # compute length of edge
  dist_points = np.sum(points * tran, axis=1)
  error = np.abs(dist_points - rho)
  close = np.where(error < thresh, 1, 0) 
  len_edge = close.sum()

  # get coordinates of close edge points
  close2 = np.expand_dims(close, -1)
  close_points = close2 * points

  # only consider the points that are close
  short_list = []
  for p in close_points:
    if p[0] != 0 and p[1] != 0:
      short_list.append([p[0], p[1]])

  short_list = np.array(short_list)

  # get object labels corresponding to close points
  close_objs = labeled_image[short_list[:, 0], short_list[:, 1]]
  close_objs = close_objs[close_objs != 0]

  # count how many point corresponding to each object lie along edge
  max = labeled_image.max()
  on_obj = np.zeros(max)
  for obj in close_objs:
    on_obj[obj - 1] = on_obj[obj - 1] + 1

  # object must have at least 5 pixel on edge for edge to belong to object
  for i in range(max):   
    if on_obj[i] > 5:
      on_obj[i] = True
    else:
      on_obj[i] = False

  return len_edge, on_obj



def get_edge_attribute(labeled_image, edge_image):
  '''
  Function to get the attributes of each edge of the image
        Calculates the angle, distance from the origin and length in pixels
  Args:
    labeled_image: binary image with grayscale level as label of component
    edge_image (2D binary image): each location indicates whether it belongs to an edge or not

  Returns:
     attribute_list: a list of list [[dict()]]. For example, [lines1, lines2,...],
     where lines1 is a list and it contains lines for the first object of attribute_list in part 1.
     Each item of lines 1 is a line, i.e., a dictionary containing keys with angle, distance, length.
     You should associate objects in part 1 and lines in part 2 by putting the attribute lists in same order.
     Note that votes in HoughLines opencv-python is not longer available since 2015. You will need to compute the length yourself.
  '''
  cpy = labeled_image.copy()
  lines = cv2.HoughLines(edge_image.astype(np.uint8),6,np.pi/8,40)
  temp = []

  for line in lines:
      rho,theta = line[0]
      len, on_obj = get_obj_and_len(labeled_image, edge_image, rho, theta, 4)
      temp.append({'angle': theta, 'distance': rho, 'length': len, 'obj': on_obj})

  max = labeled_image.max()
  attr = []

  for i in range(max):
    obj_lines = []
    for l in temp:
      if l['obj'][i] == True:
        theta = l['angle']
        rho = l['distance']
        len = l['length']
        obj_lines.append({'angle': theta, 'distance': rho, 'length': len})
    attr.append(obj_lines)

  return attr


def draw_edge_attributes(image, attribute_list):
    attributed_image = image.copy()
    for lines in attribute_list:
        for line in lines:
            angle = (float)(line["angle"])
            distance = (float)(line["distance"])

            a = np.cos(angle)
            b = np.sin(angle)
            x0 = a * distance
            y0 = b * distance
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(
                attributed_image,
                pt1,
                pt2,
                (0, 255, 0),
                2,
            )

    return attributed_image

def get_circle_attribute(labeled_image, edge_image):

    # extra credits
    raise NotImplementedError  #TODO



def detect_objects(attribute_list, edge_attribute_list, num_obj):
    '''
    feel free to edit the input format
    Args:
        attribute_list:
        edge_attribute_list:

    Returns: list of global and edge attributes for each object

    '''
    full_list = []
    
    for i in range(num_obj):
      full_attr = attribute_list[i].copy()
      full_attr['lines'] = edge_attribute_list[i]
      full_list.append(full_attr)
    
    return full_list



def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)

  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)

  # part 1
  binary_image = binarize(gray_image, thresh_val=thresh_val)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)

  labeled_image = label(binary_image)
  labeled_image_visible = 255 - 10 * labeled_image
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image_visible)

  attribute_list = get_attribute(labeled_image)
  print('attribute list:')
  print(attribute_list)

  attributed_image = draw_attributes(img, attribute_list)
  cv2.imwrite("output/" + img_name + "_attributes.png", attributed_image)


  # part 2
  # feel free to tune hyperparameters or use double-threshold
  edge_image = detect_edges(gray_image, sigma=1., threshold=0.3)
  cv2.imwrite("output/" + img_name + "_edges.png", edge_image)

  edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
  print('edge attribute list:')
  print(edge_attribute_list)

  attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
  cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)

  # extra credits for part 2: show your circle attributes and plot circles
  # circle_attribute_list = get_circle_attribute(labeled_image, edge_image)
  # attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
  # cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)

  # part 3
  objects = detect_objects(attribute_list, edge_attribute_list, labeled_image.max())
  print(objects)


if __name__ == '__main__':
  main(sys.argv[1:])
# example usage: python p1n2.py two_objects 128
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
