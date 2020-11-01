# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:15:54 2020

@author: panpy
"""

import cv2
import numpy as np
from PIL import Image


def cut_character(apath):
    img = cv2.imread(apath)

    # binarization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    gray = np.logical_xor(1, gray).astype('uint8')

    # sum by row
    sum_of_row = np.sum(gray, axis=1)

    # select maximum sum of row
    index_row_valid = []
    max_index_row_valid = []
    len_row_valid = 0
    max_len_row_valid = 0
    for index, value in enumerate(sum_of_row):
        if value > 2:
            index_row_valid.append(index)
            len_row_valid += 1
        else:
            if len_row_valid > max_len_row_valid:
                max_len_row_valid = len_row_valid
                max_index_row_valid = index_row_valid
                len_row_valid = 0
                index_row_valid = []
            else:
                len_row_valid = 0
                index_row_valid = []
        # if the slice end in the last row
        if len_row_valid > max_len_row_valid:
            max_len_row_valid = len_row_valid
            max_index_row_valid = index_row_valid

    # cut in row
    gray = gray[max_index_row_valid]
    img = img[max_index_row_valid]

    # cut off the blank at the right side
    sum_of_column = np.sum(gray, axis=0)
    for i in range(len(sum_of_column) - 1, -1, -1):
        if sum_of_column[i] >= 2:
            gray = gray[:, 0:i]
            img = img[:, 0:i]
            break

    # dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    thresh = cv2.dilate(gray, kernel)

    # get edges [with opencv 4.4 return 2 values, previous version return 3 values]
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cut the correct image
    character_bbox_list = []

    for i, cnt in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(cnt)

        if h > 6 and 0.5 < w / h < 2:
            character_bbox_list.append([x, y, w, h])

    character_bbox_list = np.array(character_bbox_list)

    # axis x need to be found in the range of [:,0] array
    # np.min[0] means that the min value of 0 column which stands for axis x
    left_x_index = np.where(character_bbox_list[:, 0] == np.min(character_bbox_list, axis=0)[0])[0][0]
    right_x_index = np.where(character_bbox_list[:, 0] == np.max(character_bbox_list, axis=0)[0])[0][0]
    left_x_position = character_bbox_list[left_x_index][0]
    right_x_position = character_bbox_list[right_x_index][0]
    right_bbox_width = character_bbox_list[right_x_index][2]
    length_character = right_x_position - left_x_position + right_bbox_width

    # axis y need to be found in the range of [:,1] array
    # np.min[1] means that the min value of 1 column which stands for axis y
    top_y_index = np.where(character_bbox_list[:, 1] == np.min(character_bbox_list, axis=0)[1])[0][0]
    top_y_position = character_bbox_list[top_y_index][1]
    max_height = 0

    for i, value in enumerate(character_bbox_list):
        if character_bbox_list[i][1] + character_bbox_list[i][3] > max_height:
            max_height = character_bbox_list[i][1] + character_bbox_list[i][3]

    height_character = max_height - top_y_position

    img = img[top_y_position:top_y_position + height_character, left_x_position:left_x_position + length_character]

    return img
