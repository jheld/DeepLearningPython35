import os

from setup_network import crop, net_parse_crops, net_crops_that_are_good
import network2
import pickle
import argparse
from PIL import Image
import numpy as np
from math import sqrt, ceil, floor

from setup_network import find_per_row, get_rows, find_similar_item_get_best, draw_hits, get_better
from PIL import ImageDraw


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument(u'network_name', type=str)
    a_p.add_argument(u'evaluation_form', type=str)
    a_p.add_argument(u'--evaluation_form_marked', type=str)
    a_p.add_argument(u'--cache_initial_matches', type=int, default=1)
    a_p.add_argument(u'--cache_parsed_rows', type=int, default=1)

    args = a_p.parse_args()
    net = network2.load(args.network_name)
    eval_form_file_name = args.evaluation_form
    eval_form_marked_file_name = args.evaluation_form_marked
    if not eval_form_marked_file_name:
        eval_form_marked_file_name = eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_marked.jpg'
    cache_initial_matches = int(args.cache_initial_matches if args.cache_initial_matches != '' else 1) \
        if isinstance(args.cache_initial_matches, str) else int(args.cache_initial_matches)
    cache_parsed_rows = int(args.cache_parsed_rows if args.cache_parsed_rows != '' else 1) \
        if isinstance(args.cache_parsed_rows, str) else int(args.cache_parsed_rows)
    full_matches = None
    if cache_initial_matches:
        if os.path.exists(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_initial_matches'):
            full_matches = pickle.load(open(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_initial_matches', 'rb'))
    if full_matches is None:
        full_crops = net_crops_that_are_good(eval_form_file_name, net, 30, 30, printing=False)
        full_matches = list(full_crops)
        if cache_initial_matches:
            pickle.dump(full_matches, open(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_initial_matches', 'wb'))
    parsed_rows = None
    if cache_parsed_rows:
        if os.path.exists(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_parsed_rows'):
            parsed_rows = pickle.load(open(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_parsed_rows', 'rb'))
    if parsed_rows is None:
        s_fm = sorted(full_matches, key=lambda x: x.result[1][0])
        better_fm = get_better(s_fm, net)
        rows = get_rows(better_fm)
        parsed_rows = list(rows)
        if cache_parsed_rows:
            pickle.dump(parsed_rows, open(eval_form_file_name + '_' + os.path.basename(args.network_name) + u'_parsed_rows', 'wb'))
    eval_image = Image.open(args.evaluation_form)
    eval_image = draw_hits(parsed_rows, eval_image)
    eval_image.convert(u'RGB').save(eval_form_marked_file_name)
