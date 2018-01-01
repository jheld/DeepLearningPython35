from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import io
import os
import numpy as np

from future.builtins import open, str

import network2


def get_formatted_input(input_data, classifier):
    """
    Transforms the data into numpy format
    :param input_data: file path or iterable
    :param classifier: 0 or 1 classifier for the set
    :return:
    """
    if isinstance(input_data, str):
        input_data = open(input_data, 'rb')
        circle_data = pickle.load(input_data)
    else:
        circle_data = input_data
    inner_arrays = []
    for l_circle in circle_data:
        label_data = []
        inner_array = [np.array([circle], np.float32) for circle in l_circle]
        label_data.append(np.array([classifier, ], np.float32))
        inner_arrays.append(tuple([np.array(inner_array, np.float32), np.array(label_data, np.float32)]))
    return inner_arrays


def get_formatted_input_not_training(input_data, classifier):
    """
    Transforms the data into numpy format
    :param input_data: file path or iterable
    :param classifier: 0 or 1 classifier for the set
    :return:
    """
    if isinstance(input_data, str):
        input_data = open(input_data, 'rb')
        circle_data = pickle.load(input_data)
    else:
        circle_data = input_data
    inner_arrays = []
    for l_circle in circle_data:
        label_data = []
        inner_array = [np.array([circle], np.float32) for circle in l_circle]
        label_data.append(np.array([classifier, ], np.float32))
        inner_arrays.append(tuple([np.array(inner_array, np.float32), classifier]))
    return inner_arrays


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument('pickle_file_name', type=str)
    a_p.add_argument('network_output_file', type=str)
    a_p.add_argument('classifier', type=int)
    a_p.add_argument('size', type=int)
    a_p.add_argument('epochs', type=int)
    a_p.add_argument('hidden_nodes', type=int)
    a_p.add_argument('--monitor_training', default=False, type=bool)
    args = a_p.parse_args()
    formatted_input = get_formatted_input(args.pickle_file_name, args.classifier)
    # consider using the len of the first formatted input's value as the size, instead of being CLI-based.
    net = network2.Network([args.size, args.hidden_nodes, 1])
    net.SGD(formatted_input, args.epochs, 10, 3.0,
            monitor_training_accuracy=args.monitor_training,
            monitor_training_cost=args.monitor_training)
    net.save(args.network_output_file)
