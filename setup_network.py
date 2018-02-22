from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import io
import os
import random
from collections import namedtuple
from copy import deepcopy

import numpy as np

from future.builtins import open, str

import network2
from get_circle import pixels_from_circle, training_evaluation_test_split, crop, generate_threshold_adjustments


def get_default_input():
    tr_good, ev_good, te_good = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/eval_2_under_40.pkl')
    tr_bad, ev_bad, te_bad = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/eval_2_under_40.pkl')
    tr_ipsum, ev_ipsum, te_ipsum = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/lorem_ipsum_generated.pkl')
    tr_bad += tr_ipsum
    ev_bad += ev_ipsum
    te_bad += te_ipsum
    training = []
    training.extend(get_formatted_input(tr_good, 1, multi_class=True, use_inner_array=True, convert_scale=True))
    training.extend(get_formatted_input(tr_bad, 0, multi_class=True, use_inner_array=True, convert_scale=True))
    evaluation = []
    evaluation.extend(get_formatted_input_not_training(ev_good, 1, use_inner_array=True, convert_scale=True))
    evaluation.extend(get_formatted_input_not_training(ev_bad, 0, use_inner_array=True, convert_scale=True))
    testing = []
    testing.extend(get_formatted_input_not_training(te_good, 1, use_inner_array=True, convert_scale=True))
    testing.extend(get_formatted_input_not_training(te_bad, 0, use_inner_array=True, convert_scale=True))
    return training, evaluation, testing


def get_formatted_input(input_data, classifier=None, convert_scale=False, multi_class=False, use_inner_array=False):
    """
    Transforms the data into numpy format
    :param input_data: file path or iterable
    :param convert_scale:
    :param classifier: 0 or 1 classifier for the set
    :param multi_class:
    :param use_inner_array:
    :return:
    """
    if isinstance(input_data, str):
        input_data = open(input_data, 'rb')
        circle_data = pickle.load(input_data)
    else:
        circle_data = input_data
    inner_arrays = []
    if classifier is not None:
        if multi_class:
            if classifier == 1:
                classifier = [0, 1]
            else:
                classifier = [1, 0]
        else:
            classifier = [classifier]
        if use_inner_array:
            classifier = [np.array([item]) for item in classifier]
    for l_circle in circle_data:
        if not convert_scale:
            inner_array = [np.array([circle], np.float32)
                           if use_inner_array
                           else circle
                           for circle in l_circle]
        else:
            inner_array = [np.array([1.0 - (circle/255)], np.float32)
                           if use_inner_array
                           else 1.0 - (circle/255)
                           for circle in l_circle]
        if classifier is not None:
            inner_arrays.append(tuple([np.array(inner_array, np.float32), np.array(classifier)]))
        else:
            inner_arrays.append(np.array(inner_array, np.float32))
    return inner_arrays


def get_formatted_input_not_training(input_data, classifier, convert_scale=False, use_inner_array=False):
    """
    Transforms the data into numpy format
    :param input_data: file path or iterable
    :param classifier: 0 or 1 classifier for the set
    :param convert_scale:
    :param use_inner_array:
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
        if not convert_scale:
            inner_array = [np.array([circle], np.float32)
                           if use_inner_array
                           else circle
                           for circle in l_circle]
        else:
            inner_array = [np.array([1.0 - (circle/255)], np.float32) if use_inner_array
                           else 1.0 - (circle/255)
                           for circle in l_circle]
        label_data.append(np.array([classifier, ], np.float32))
        inner_arrays.append(tuple([np.array(inner_array, np.float32), classifier]))
    return inner_arrays


ParseCropResult = namedtuple(u'ParseCropResult', [u'idx', u'cr', u'cr_scaled', u'dc_formatted_cr', u'result'])


def parse_crops(crops, sDA):
    for idx, cr in enumerate(crops):
        cr = cr.convert('L')
        y = np.asarray(cr.getdata(), dtype=np.float64)
        y = np.asarray(y, dtype=np.uint8)
        y = np.array([1 - i/255 for i in y])
        y = np.array([y])
        dc_formatted_cr = deepcopy(y)
        result = sDA.predict(dc_formatted_cr)
        yield ParseCropResult(idx, cr, y, dc_formatted_cr, result)


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument('network_output_file', type=str)
    a_p.add_argument('epochs', type=int)
    a_p.add_argument('hidden_nodes', type=int)
    a_p.add_argument('--good_input_file_name', type=str)
    a_p.add_argument('--bad_input_file_name', type=str)
    a_p.add_argument(u'--lmbda', default=0.0)
    a_p.add_argument(u'--eta', default=3)
    a_p.add_argument(u'--default_input', default=False)
    a_p.add_argument('--monitor_training', default=False, type=bool)
    a_p.add_argument('--shuffle_input', default=False, type=bool)
    args = a_p.parse_args()
    if not args.default_input:
        # note, there is no eval and test input support here, yet.
        good_input_file = args.good_input_file_name
        bad_input_file = args.bad_input_file_name
        formatted_input = get_formatted_input(good_input_file, 1, convert_scale=True, use_inner_array=True)
        formatted_input.extend(get_formatted_input(bad_input_file, 0, convert_scale=True, use_inner_array=True))
        formatted_ev = None
    else:
        formatted_input, formatted_ev, _ = get_default_input()
        if args.shuffle_input:
            print(u'Going to shuffle now.')
            random.shuffle(formatted_input)
            random.shuffle(formatted_ev)
    # consider using the len of the first formatted input's value as the size, instead of being CLI-based.
    net = network2.Network([900, args.hidden_nodes, 2])
    eta = float(args.eta)
    lmbda = float(args.lmbda)
    print(u'eta: {eta}, lmbda: {lmbda}'.format(eta=eta, lmbda=lmbda))
    net.SGD(formatted_input, args.epochs, 10, eta=eta, lmbda=lmbda,
            evaluation_data=formatted_ev,
            monitor_training_accuracy=args.monitor_training,
            monitor_training_cost=args.monitor_training,
            monitor_evaluation_accuracy=bool(formatted_ev),
            monitor_evaluation_cost=bool(formatted_ev),
    )
    net.save(args.network_output_file)
