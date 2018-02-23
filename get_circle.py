from __future__ import unicode_literals, division, print_function

import argparse
import pickle
import random

import numpy as np
from PIL import Image

from future.builtins import str, open, range


def pixels_from_circle(circle_data, regular_array=True):
    """

    :param circle_data:
    :param regular_array:
    :return:
    """
    if isinstance(circle_data, str):
        x = Image.open(circle_data, 'r')
    else:
        x = circle_data
    x = x.convert('L')
    y = np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1] * x.size[0], -1))
    y = np.asarray(y, dtype=np.uint8)
    if regular_array:
        return [item[0] for item in y]
    else:
        return y


def crop(file_path, height, width):
    im = Image.open(file_path)
    imgwidth, imgheight = im.size
    # slide a window across the image
    for y in range(0, imgheight, 1):
        for x in range(0, imgwidth, 1):
            # yield the current window
            if y + height < imgheight and x + width < imgwidth:
                print(x, x + width, y, y + height)
                yield im.crop((x, y, x + width, y + height))


def save_image_as_jpg(circle_data, output_file_name):
    """

    :param circle_data:
    :param output_file_name:
    :return:
    """
    Image.fromarray(circle_data).save(output_file_name)


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


def training_evaluation_test_split(percentages, data):
    """

    :param percentages: 3-record tuple of data splits
    :type percentages: tuple | list
    :param data:
    :return:
    """
    if isinstance(data, str):
        with open(data, 'rb') as sample_file:
            data = pickle.loads(sample_file.read())
    original_length = len(data)
    training_length = int(original_length * percentages[0])
    evaluation_length = int(original_length * percentages[1])
    test_length = int(original_length * percentages[2])
    training_idxs = sorted(random.sample(range(len(data)), training_length), reverse=True)
    training = []
    for idx in training_idxs:
        training.append(data.pop(idx))
    training = training[::-1]
    evaluation_idxs = sorted(random.sample(range(len(data)), evaluation_length), reverse=True)
    evaluation = []
    for idx in evaluation_idxs:
        evaluation.append(data.pop(idx))
    evaluation = evaluation[::-1]
    test_idxs = sorted(random.sample(range(len(data)), test_length), reverse=True)
    test = []
    for idx in test_idxs:
        test.append(data.pop(idx))
    test = test[::-1]
    return training, evaluation, test


def generate_threshold_adjustments(circle_data, threshold_start, threshold_end, threshold_step, **kwargs):
    """
    Wrapper around adjust_from_circle to generate several data thresholds
    :param circle_data:
    :param threshold_start:
    :param threshold_end:
    :param threshold_step:
    :param kwargs:
    :return:
    """
    for threshold in range(threshold_start, threshold_end, threshold_step):
        yield adjust_from_circle(circle_data, random_threshold=threshold / 100, **kwargs)


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument(u'circle_file_path', type=str)
    a_p.add_argument(u'output_file_path', type=str)
    a_p.add_argument(u'--random_threshold', type=float, default=0.1, required=False)
    a_p.add_argument(u'--rand_range', nargs=2, required=False)
    a_p.add_argument(u'--num_samples', type=int, default=1000, required=False)
    a_p.add_argument(u'--dark_cut_off', type=int,  default=30, required=False)
    a_p.add_argument(u'--default_good', default=False)
    a_p.add_argument(u'--default_bad', default=False)
    a_p.add_argument(u'--default_ipsum', default=False)
    a_p.add_argument(u'--shuffle_samples', default=False)
    args = a_p.parse_args()
    if args.default_good:
        adjustments = generate_threshold_adjustments(u'images/eval_circle.jpg', 1, 30, 3, rand_range=[255, 256], dark_cut_off=200)
        adjustments = [list(a)
                       for adj in adjustments
                       for a in adj]
        if args.shuffle_samples:
            random.shuffle(adjustments)
        with open(u'sample_data/eval_2_under_40.pkl', 'wb') as circle_output:
            pickle.dump(adjustments, circle_output)
    if args.default_bad:
        adjustments = generate_threshold_adjustments(u'images/eval_circle.jpg', 90, 100, 3, rand_range=[255, 256], dark_cut_off=200)
        adjustments = [list(a)
                       for adj in adjustments
                       for a in adj]
        if args.shuffle_samples:
            random.shuffle(adjustments)
        with open(u'sample_data/eval_90_under_100.pkl', 'wb') as circle_output:
            pickle.dump(adjustments, circle_output)
    if args.default_ipsum:
        crops = crop(u'sample_data/thesis_lorem_ipsum.jpg', 30, 30)
        ipsums = []
        for _ in range(3000):
            c = next(crops)
            ipsums.append(np.asarray(c.convert(u'L')).reshape(c.size))
        if args.shuffle_samples:
            random.shuffle(ipsums)
        with open(u'sample_data/lorem_upsem_generated.pkl', 'wb') as circle_output:
            pickle.dump(ipsums, circle_output)
    if not args.default_good and not args.default_bad and not args.default_ipsum:
        output_samples = adjust_from_circle(args.circle_file_path, args.random_threshold, args.rand_range, args.num_samples,
                                     args.dark_cut_off)
        with open(args.output_file_path, 'wb') as circle_output:
            pickle.dump(list(output_samples), circle_output)
