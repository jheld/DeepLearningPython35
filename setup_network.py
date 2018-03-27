from __future__ import unicode_literals, print_function, division
import argparse
import pickle
import io
import os
import random
from collections import namedtuple
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from future.builtins import open, str

import network2
from generate_samples import pixels_from_circle, training_evaluation_test_split, crop_sliding_window, generate_threshold_adjustments, \
    crop_list, CropResult


def get_default_input(good_permutations=True, multi_class=True):
    tr_good, ev_good, te_good = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/{}eval_1_under_30.pkl'.format(u'permutations_' if good_permutations else u''))
    correct_size = len(tr_good[0])
    tr_bad, ev_bad, te_bad = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/eval_90_under_100.pkl')
    tr_ipsum, ev_ipsum, te_ipsum = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/lorem_ipsum_generated.pkl')
    tr_qr, ev_qr, te_qr = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/qr_codes_small.pkl')
    tr_numbered, ev_numbered, te_numbered = training_evaluation_test_split((0.7, 0.15, 0.15), u'sample_data/numbered_list_cropped.pkl')
    tr_bad.extend(tr_ipsum)
    ev_bad.extend(ev_ipsum)
    te_bad.extend(te_ipsum)
    tr_bad.extend(tr_qr)
    ev_bad.extend(ev_qr)
    te_bad.extend(te_qr)
    tr_bad.extend(tr_numbered)
    ev_bad.extend(ev_numbered)
    te_bad.extend(te_numbered)
    training = []
    # with open(u'sample_data/eval_half_right.pkl', 'rb') as circle_input:
    #     half_right_circles = pickle.load(circle_input)
    #     tr_bad.extend(half_right_circles)
    training.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), np.array([np.array([0]), np.array([1])])]) for item in tr_good])
    training.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), np.array([np.array([1]), np.array([0])])]) for item in tr_bad])
    evaluation = []
    evaluation.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), 1]) for item in ev_good])
    evaluation.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), 0]) for item in ev_bad])
    testing = []
    testing.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), 1]) for item in te_good])
    testing.extend([tuple([np.array([np.array([1 - i/255, ]) for i in item]).reshape(correct_size, 1), 0]) for item in te_bad])
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


ParseCropResultSDA = namedtuple(u'ParseCropResult', [u'idx', u'cr', u'cr_scaled', u'dc_formatted_cr', u'result'])
ParseCropResult = namedtuple(u'ParseCropResult', [u'idx', u'cr_res', u'cr_scaled', u'result'])


def parse_crops(crops, sDA):
    for idx, cr in enumerate(crops):
        cr = cr.convert('L')
        y = np.asarray(cr.getdata(), dtype=np.float64)
        y = np.asarray(y, dtype=np.uint8)
        y = np.array([1 - i/255 for i in y])
        y = np.array([y])
        dc_formatted_cr = deepcopy(y)
        result = sDA.predict(dc_formatted_cr)
        yield ParseCropResultSDA(idx, cr, y, dc_formatted_cr, result)


def get_better(s_fm: list, net_proc: network2.Network, threshold: float=0.9) -> list:
    better_fm = []
    for f_idx, fm in enumerate(s_fm):
        count = 0
        max_matches = len(list(range(45, 360, 45)))
        for idx in range(45, 360, 45):
            r = net_proc.feedforward(np.array([1 - i / 255
                                               for i in np.array(make_rotation(fm.cr_res.cr, idx))]
                                              ).reshape(fm.cr_res.cr.size[0] * fm.cr_res.cr.size[1], 1))
            if r[1][0] >= threshold:
                count += 1
        if count == max_matches:
            better_fm.append((f_idx, fm))
    return better_fm


def draw_hits(parsed_rows, source_image):
    source_image = source_image.convert(u'RGBA')
    draw = ImageDraw.Draw(source_image)
    for p_r in parsed_rows:
        for item in p_r:
            draw.rectangle(
                (item[1].cr_res.box[0], item[1].cr_res.box[2], item[1].cr_res.box[1], item[1].cr_res.box[3]),
                outline='red')
    source_image.show()
    return source_image


def make_rotation(cr_img, rotation):
    im2 = cr_img.convert(u'RGBA')
    rot = im2.rotate(rotation)
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    out = Image.composite(rot, fff, rot)
    return out.convert(u'L')


def get_rows(items, x_threshold=15, y_threshold=15):
    sorted_by_y_start = sorted(items, key=lambda x: x[1].cr_res.box[2])
    prev_start = 0
    for target_item in sorted_by_y_start:
        if prev_start == 0:
            prev_start = target_item[1].cr_res.box[2]
        if prev_start - y_threshold <= target_item[1].cr_res.box[2]:
            yield find_per_row(sorted_by_y_start, target_item[1].cr_res.box[2],
                               x_threshold=x_threshold, y_threshold=y_threshold)
            prev_start = target_item[1].cr_res.box[3] + 5


def find_per_row(items, y_start, x_threshold=15, y_threshold=15):
    prev_start = 0
    sorted_by_x_start = sorted(items, key=lambda x: x[1].cr_res.box[0])
    best_by_row = []
    for idx, target_item in enumerate(sorted_by_x_start):
        if y_start - y_threshold <= target_item[1].cr_res.box[2] <= y_start + y_threshold:
            if prev_start == 0:
                prev_start = target_item[1].cr_res.box[0]
            if prev_start <= target_item[1].cr_res.box[0]:
                best = find_similar_item_get_best(target_item, items,
                                                  x_threshold=x_threshold, y_threshold=y_threshold)
                # found the best match against this item (row's slot x), so we don't want to look at another
                # unless it is past the end of this one's slot by some appropriate margin (this one's end + 5)
                prev_start = best[1].cr_res.box[1] + 5
                best_by_row.append(best)
    return best_by_row


def find_similar_item_get_best(target_item, items, x_threshold=15, y_threshold=15):
    x_start = target_item[1].cr_res.box[0]
    y_start = target_item[1].cr_res.box[2]
    similar_item = [comp_item
                    for comp_item in items
                    if y_start - y_threshold <= comp_item[1].cr_res.box[2] <= y_start + y_threshold and
                    x_start - x_threshold <= comp_item[1].cr_res.box[0] <=
                    x_start + x_threshold]
    return sorted(similar_item, key=lambda x: x[1].result[1][0])[-1]


def net_parse_crops(crops: CropResult, net_proc: network2.Network):
    for idx, cr_res in enumerate(crops):
        cr = cr_res.cr
        y = np.asarray(cr, dtype=np.uint8)
        y = np.array([1 - i/255 for i in y])
        y = y.reshape(y.size, 1)
        result = net_proc.feedforward(y)
        yield ParseCropResult(idx, cr_res, y, result)


def net_crops_that_are_good(source_file_path, net_proc, height=30, width=30,
                            multi_class=True, printing=True, crops=None, threshold=0.9):
    if crops is None:
        crops = crop_sliding_window(source_file_path, height, width, printing=printing)
    pcs = net_parse_crops(crops, net_proc)
    full_size = height * width
    res_idx = 1 if multi_class else 0
    for pc in pcs:
        if pc.result[res_idx][0] >= threshold:
            if printing:
                print(pc.result, pc.result[res_idx][0])
            if len([1 for idx in range(1, 4)
                    if net_proc.feedforward(np.array([1 - i / 255
                                                      for i in np.array(pc.cr_res.cr.rotate(90 * idx))]
                                                     ).reshape(full_size, 1))[1][0] >= threshold]) >= 2:
                yield pc


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument('epochs', type=int)
    a_p.add_argument('hidden_nodes', type=str)
    a_p.add_argument('--good_input_file_name', type=str)
    a_p.add_argument('--bad_input_file_name', type=str)
    a_p.add_argument(u'--lmbda', default=0.0)
    a_p.add_argument(u'--eta', default=3)
    a_p.add_argument(u'--default_input', default=0)
    a_p.add_argument('--monitor_training', default=False, type=bool)
    a_p.add_argument('--shuffle_input', default=0, type=bool)
    a_p.add_argument('--early_stopping_n', default=0, type=int)
    a_p.add_argument(u'--default_good_permutations', default=False)
    a_p.add_argument(u'--binary_classifier', default=1)
    a_p.add_argument(u'--x_size', type=int, default=30)
    a_p.add_argument(u'--y_size', type=int, default=30)
    args = a_p.parse_args()
    x_size = int(args.x_size)
    y_size = int(args.y_size)
    is_multi_class = bool(int(args.binary_classifier))
    formatted_te = []
    hidden_nodes = list(map(int, args.hidden_nodes.split(u',')))
    default_good_permutations = int(args.default_good_permutations) \
        if isinstance(args.default_good_permutations, str) \
        else args.default_good_permutations
    if not args.default_input:
        # note, there is no eval and test input support here, yet.
        good_input_file = args.good_input_file_name
        bad_input_file = args.bad_input_file_name
        formatted_input = get_formatted_input(good_input_file, 1, convert_scale=True, use_inner_array=True, multi_class=is_multi_class)
        formatted_input.extend(get_formatted_input(bad_input_file, 0, convert_scale=True, use_inner_array=True))
        formatted_ev = None
    else:
        input_name = u'sample_data/default_input{}.pkl'.format(u'' if default_good_permutations else u'_regular')
        formatted_input, formatted_ev, formatted_te = pickle.load(open(input_name, 'rb'))
        # formatted_input, formatted_ev, formatted_te = get_default_input(good_permutations=default_good_permutations, multi_class=multi_class)
        if args.shuffle_input:
            print(u'Going to shuffle now.')
            random.shuffle(formatted_input)
            random.shuffle(formatted_ev)
    # consider using the len of the first formatted input's value as the size, instead of being CLI-based.
    net = network2.Network([x_size*y_size, *hidden_nodes, 2 if is_multi_class else 1])
    eta = float(args.eta)
    lmbda = float(args.lmbda)
    print(u'eta: {eta}, lmbda: {lmbda}'.format(eta=eta, lmbda=lmbda))
    net.SGD(formatted_input, args.epochs, 10, eta=eta, lmbda=lmbda,
            evaluation_data=formatted_ev,
            early_stopping_n=args.early_stopping_n or 0,
            monitor_training_accuracy=args.monitor_training,
            monitor_training_cost=args.monitor_training,
            monitor_evaluation_accuracy=bool(formatted_ev),
            monitor_evaluation_cost=bool(formatted_ev),
    )
    te_accurate_count = 0
    for item in formatted_te:
        te_result = net.feedforward(item[0])
        result_idx = 1 if is_multi_class else 0
        if item[1] == 0:
            if te_result[result_idx][0] < 0.5:
                te_accurate_count += 1
        else:
            if te_result[result_idx][0] > 0.5:
                te_accurate_count += 1
    if formatted_te:
        print(u'Test accuracy: {}, {} / {}'.format(te_accurate_count / len(formatted_te), te_accurate_count, len(formatted_te)))
    save_state = input(u'Save the network state?')
    save_state = save_state or 0
    if int(save_state):
        network_output_file = u'eval_network_epoch_{}_hidden_{}_eta_{}_lmbda_{}_n_{}{}'.format(args.epochs, args.hidden_nodes, eta, lmbda, args.early_stopping_n, u'' if is_multi_class else u'_unary')
        print(u'Saving the network state: {}'.format(network_output_file))
        net.save(network_output_file)
