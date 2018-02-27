import pickle
import random

from PIL import Image

import numpy as np


def adjust_from_circle(circle_data, random_threshold=0.1, rand_range=None, num_samples=1000, dark_cut_off=30):
    """

    :param circle_data:
    :param random_threshold:
    :param rand_range:
    :param num_samples:
    :param dark_cut_off:
    :return:
    """
    if not rand_range:
        rand_range = (80, 256)
    rand_range = tuple(map(int, rand_range))
    if isinstance(circle_data, str):
        with open(circle_data, 'rb') as circle_file:
            try:
                circle_data = pickle.loads(circle_file.read())
            except Exception:
                circle_data = np.asarray(Image.open(circle_data).convert(u'L').getdata(), dtype=np.float64)
    for _ in range(num_samples):
        # sample = []
        # for item in circle_data:
        #     if item < dark_cut_off and random.random() <= random_threshold:
        #         print('rand')
        #         sample.append(random.randrange(*rand_range))
        #     else:
        #         print('same')
        #         sample.append(item)
        # yield sample

        yield [random.randrange(*rand_range)
               if item < dark_cut_off and random.random() <= random_threshold
               else item
               for item in circle_data]


def find_top(circle):
    top_black_index = min([idx for idx, item in enumerate(circle) if item < 200])
    return top_black_index

def find_bottom(circle):
    bottom_black_index = max([idx for idx, item in enumerate(circle) if item < 200])
    return bottom_black_index

def find_left(circle, top_idx, bottom_idx):
    i_s = set()
    dataset = circle.reshape(30, 30)
    for i in range(30):
        for j in range(30):
            if dataset[i][j] < 200:
                i_s.add((i, j))
    smallest_i = sorted(i_s, key=lambda item: item[0])[0]
    return smallest_i[1]*30 + smallest_i[0]


def find_right(circle, top_idx, bottom_idx):
    i_s = set()
    dataset = circle.reshape(30, 30)
    for i in range(30):
        for j in range(30):
            if dataset[i][j] < 200:
                i_s.add((i, j))
    smallest_i = sorted(i_s, key=lambda item: item[0])[-1]
    return smallest_i[1]*30 + smallest_i[0]



def set_fill_value_from_circle(circle_data, threshold=0.99, rand_range=None, num_samples=1000):
    if not rand_range:
        rand_range = (80, 256)

    rand_range = tuple(map(int, rand_range))
    if isinstance(circle_data, str):
        with open(circle_data, 'rb') as circle_file:
            try:
                circle_data = pickle.loads(circle_file.read())
            except Exception:
                circle_data = np.asarray(Image.open(circle_data).convert(u'L').getdata(), dtype=np.float64)
    top_black_index = find_top(circle_data)
    bottom_black_index = find_bottom(circle_data)
    left_index = find_left(circle_data, top_black_index, bottom_black_index)
    right_index = find_right(circle_data, top_black_index, bottom_black_index)
    for _ in range(num_samples):
        yield [random.randrange(*rand_range)
               if item < 200 and random.random() <= threshold
               else item
               for item in circle_data]