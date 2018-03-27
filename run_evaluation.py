import os
from concurrent.futures import ProcessPoolExecutor

from setup_network import crop_sliding_window, net_parse_crops, net_crops_that_are_good
import network2
import pickle
import argparse
from PIL import Image
import numpy as np
from math import sqrt, ceil, floor

from setup_network import find_per_row, get_rows, find_similar_item_get_best, draw_hits, get_better
from PIL import ImageDraw


def run_it(network_name, eval_form_file_name, evaluation_form, eval_form_marked_file_name, cache_initial_matches, cache_parsed_rows, parse_threshold=0.9, save=True, x_size=30, y_size=30):
    """

    :param network_name:
    :param eval_form_file_name:
    :param evaluation_form:
    :param eval_form_marked_file_name:
    :param cache_initial_matches:
    :param cache_parsed_rows:
    :param parse_threshold:
    :param save:
    :param x_size:
    :param y_size:
    :return:
    """
    net = network2.load(network_name)
    if not eval_form_marked_file_name:
        eval_form_marked_file_name = eval_form_file_name + '_' + os.path.basename(network_name) + u'_marked.jpg'
    full_matches = None
    if cache_initial_matches:
        if os.path.exists(eval_form_file_name + '_' + os.path.basename(network_name) + u'_initial_matches'):
            full_matches = pickle.load(
                open(eval_form_file_name + '_' + os.path.basename(network_name) + u'_initial_matches', 'rb'))
    if full_matches is None:
        full_crops = net_crops_that_are_good(eval_form_file_name, net, y_size, x_size, printing=False)
        full_matches = list(full_crops)
        if cache_initial_matches and save:
            pickle.dump(full_matches,
                        open(eval_form_file_name + '_' + os.path.basename(network_name) + u'_initial_matches', 'wb'))
    parsed_rows = None
    parsed_rows_file_name = eval_form_file_name + '_' + os.path.basename(network_name) + u'_parsed_rows'
    if cache_parsed_rows:
        if os.path.exists(parsed_rows_file_name):
            parsed_rows = pickle.load(
                open(parsed_rows_file_name, 'rb'))
    if parsed_rows is None:
        s_fm = sorted(full_matches, key=lambda x: x.result[1][0])
        better_fm = get_better(s_fm, net, threshold=parse_threshold)
        rows = get_rows(better_fm)
        parsed_rows = list(rows)
        if cache_parsed_rows and save:
            pickle.dump(parsed_rows,
                        open(parsed_rows_file_name, 'wb'))
    eval_image = Image.open(evaluation_form)
    eval_image = draw_hits(parsed_rows, eval_image)
    if save:
        eval_image.convert(u'RGB').save(eval_form_marked_file_name)
    return eval_form_marked_file_name, parsed_rows, parsed_rows_file_name


def get_input_vars(input_args):
    """

    :param input_args:
    :return:
    """
    network_name = input_args.network_name
    eval_form_file_name = input_args.evaluation_form
    evaluation_form = input_args.evaluation_form
    eval_form_marked_file_name = input_args.evaluation_form_marked
    cache_initial_matches = int(input_args.cache_initial_matches if input_args.cache_initial_matches != '' else 1) \
        if isinstance(input_args.cache_initial_matches, str) else int(input_args.cache_initial_matches)
    cache_parsed_rows = int(input_args.cache_parsed_rows if input_args.cache_parsed_rows != '' else 1) \
        if isinstance(input_args.cache_parsed_rows, str) else int(input_args.cache_parsed_rows)
    if input_args.save == '':
        save = True
    else:
        save = bool(int(input_args.save))
    parse_threshold = input_args.parse_threshold
    if parse_threshold == '':
        parse_threshold = 0.9
    else:
        parse_threshold = float(parse_threshold)
    x_size = int(input_args.x_size)
    y_size = int(input_args.y_size)
    return network_name, eval_form_file_name, evaluation_form, eval_form_marked_file_name, \
           cache_initial_matches, cache_parsed_rows, parse_threshold, save, x_size, y_size


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument(u'network_name', type=str)
    a_p.add_argument(u'evaluation_form', type=str)
    a_p.add_argument(u'--evaluation_form_marked', type=str)
    a_p.add_argument(u'--cache_initial_matches', type=int, default=1)
    a_p.add_argument(u'--cache_parsed_rows', type=int, default=1)
    a_p.add_argument(u'--parse_threshold', type=float, default=0.9)
    a_p.add_argument(u'--save', type=int, default=1)
    a_p.add_argument(u'--x_size', type=int, default=30)
    a_p.add_argument(u'--y_size', type=int, default=30)

    args = a_p.parse_args()

    run_it(*get_input_vars(args))
