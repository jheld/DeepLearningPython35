from __future__ import unicode_literals, division, print_function

import argparse
import pickle
import random
from collections import namedtuple

import numpy as np
from PIL import Image, ImageFilter

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


CropResult = namedtuple(u'CropResult', [u'box', u'cr'])


def crop_sliding_window(file_path, height, width, printing=True):
    """

    :param file_path:
    :param height:
    :param width:
    :param printing:
    :return:
    """
    im = Image.open(file_path)
    im = im.convert(u'L')
    img_width, img_height = im.size
    # slide a window across the image
    for y in range(0, img_height, 1):
        for x in range(0, img_width, 1):
            # yield the current window
            if y + height < img_height and x + width < img_width:
                if printing:
                    print(x, x + width, y, y + height)
                yield CropResult((x, x + width, y, y + height), im.crop((x, y, x + width, y + height)))


def crop_list(file_path, height, width, printing=True):
    """
    Very expensive, but can be used (on its own)
    :param file_path:
    :param height:
    :param width:
    :param printing:
    :return:
    """
    im = Image.open(file_path)
    imgwidth, imgheight = im.size
    all_crops = []
    # slide a window across the image
    for y in range(0, imgheight, 1):
        for x in range(0, imgwidth, 1):
            # yield the current window
            if y + height < imgheight and x + width < imgwidth:
                if printing:
                    print(x, x + width, y, y + height)
                all_crops.append(im.crop((x, y, x + width, y + height)))
    return all_crops


def save_image_as_jpg(circle_data, output_file_name):
    """

    :param circle_data:
    :param output_file_name:
    :return:
    """
    Image.fromarray(circle_data).save(output_file_name)


def adjust_from_circle(circle_data, random_threshold=0.1, rand_range=None, num_samples=1000, dark_cut_off=30, rotations=None, x_size=30, y_size=30):
    """

    :param circle_data:
    :param random_threshold:
    :param rand_range:
    :param num_samples:
    :param dark_cut_off:
    :param rotations:
    :param x_size:
    :param y_size:
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

        new_adjustment = [random.randrange(*rand_range)
                          if item < dark_cut_off and random.random() <= random_threshold
                          else item
                          for item in circle_data]
        yield new_adjustment
        if rotations:
            adj_img = Image.fromarray(np.array(new_adjustment).reshape(x_size, y_size))
            for rotation in rotations:
                im2 = adj_img.convert(u'RGBA')
                rot = im2.rotate(rotation)
                fff = Image.new('RGBA', rot.size, (255,) * 4)
                out = Image.composite(rot, fff, rot)
                as_array = np.asarray(out.convert(u'L'))
                yield as_array


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
    from math import ceil, floor
    training_length = int(ceil(original_length * percentages[0]))
    evaluation_length = int(floor(original_length * percentages[1]))
    test_length = int(floor(original_length * percentages[2]))
    training_idxs = sorted(random.sample(range(len(data)), training_length), reverse=True)
    training = [data.pop(idx) for idx in training_idxs]
    training = training[::-1]
    evaluation_idxs = sorted(random.sample(range(len(data)), evaluation_length), reverse=True)
    evaluation = [data.pop(idx) for idx in evaluation_idxs]
    evaluation = evaluation[::-1]
    test_idxs = sorted(random.sample(range(len(data)), test_length), reverse=True)
    test = [data.pop(idx) for idx in test_idxs]
    test.extend([data.pop(idx) for idx in range(len(data))])
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


def bubble_permutation(file_path=u'images/thesis_circles.jpg', start_x=130, end_x=141, start_y=113, end_y=126, x_size=30, y_size=30):
    """

    :param file_path:
    :param start_x:
    :param end_x:
    :param start_y:
    :param end_y:
    :param x_size:
    :param y_size:
    :return:
    """
    im = Image.open(file_path)
    im = im.convert(u'L')
    for i in range(start_x, end_x, 1):
        for j in range(start_y, end_y, 1):
            # im.crop((i, j, i + x_size, j + y_size)).show()
            yield im.crop((i, j, i+x_size, j+y_size)).filter(ImageFilter.UnsharpMask(percent=1700, threshold=0))
            yield im.crop((i, j, i+x_size, j+y_size))


def scale_image(cr_img, scale_size, x_size=30, y_size=30):
    """
    Given an image, scale it to the given scale size.
    :param cr_img:
    :param scale_size:
    :param x_size:
    :param y_size:
    :return:
    """
    im_rgba = cr_img.convert(u'RGBA')
    im_rgba.thumbnail(scale_size)
    light_output = Image.new('RGBA', (x_size, y_size), (255,) * 4)
    scaled = Image.composite(im2, light_output, im2)
    return scaled.convert(u'L')


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument(u'--default_good', default=0)
    a_p.add_argument(u'--default_bad', default=0)
    a_p.add_argument(u'--default_ipsum', default=0)
    a_p.add_argument(u'--shuffle_samples', default=0)
    a_p.add_argument(u'--default_good_permutations', default=0)
    a_p.add_argument(u'--enable_scaling', default=0)
    a_p.add_argument(u'--x_size', type=int, default=30)
    a_p.add_argument(u'--y_size', type=int, default=30)
    args = a_p.parse_args()
    x_size = int(args.x_size)
    y_size = int(args.y_size)
    default_good_permutations = int(args.default_good_permutations) if isinstance(args.default_good_permutations, str) else args.default_good_permutations
    enable_scaling = int(args.enable_scaling) if isinstance(args.enable_scaling, str) else args.enable_scaling
    default_good = int(args.default_good)
    default_bad = int(args.default_bad)
    default_ipsum = int(args.default_ipsum)
    if default_good:
        if default_good_permutations:
            print(u'using default good permutations')
            all_adjustments = []
            num_samples = 35
            if enable_scaling:
                num_samples = 62
            # input_file = '/home/jason/box/from copy/Downloads/swami/eval/s18-cs411-1_only.jpg'
            input_file = '/home/jason/box/From_BrotherDevice/20160102123552_001.jpg'
            # input_file = u'images/thesis_circles.jpg'
            # permutation_bounds = dict(start_x = 79, end_x = 93, start_y = 57, end_y = 70)
            permutation_bounds = dict(start_x = 117, end_x = 124, start_y = 80, end_y = 88)
            for idx, bubble in enumerate(bubble_permutation(input_file, x_size=x_size, y_size=y_size, **permutation_bounds)):
                # bubble.show()
                adjustments = generate_threshold_adjustments(np.array(bubble).reshape(x_size*y_size), 1, 10, 2, rand_range=[230, 256],
                                                             dark_cut_off=200, num_samples=num_samples)
                print(idx)
                adjustments = [np.array(a)
                               for adj in adjustments
                               for a in adj]
                all_adjustments.extend(adjustments)
                if enable_scaling:
                    copy_of_bubble = Image.new(u'L', (x_size, y_size))
                    copy_of_bubble.paste(bubble)
                    scale_bubble = scale_image(copy_of_bubble, (25, 25))
                    scale_adjustments = generate_threshold_adjustments(np.array(scale_bubble).reshape(x_size*y_size), 1, 10, 2, rand_range=[255, 256],
                                                                       dark_cut_off=200, num_samples=num_samples, x_size=x_size, y_size=y_size)
                    scale_adjustments = [np.array(a)
                                         for adj in scale_adjustments
                                         for a in adj]
                    all_adjustments.extend(scale_adjustments)

            adjustments = all_adjustments
        else:
            # one_circle_path = 'images/eval_circle.jpg'
            one_circle_path = '/home/jason/box/From_BrotherDevice/20160102123552_001_single.jpg'
            adjustments = generate_threshold_adjustments(one_circle_path, 1, 20, 1, rand_range=[230, 256], dark_cut_off=200, rotations=[], x_size=x_size, y_size=y_size)
            adjustments = [np.array(a)
                           for adj in adjustments
                           for a in adj]
            adjustments.append(np.array(Image.open(one_circle_path).convert(u'L')).reshape(x_size * y_size))
        # im = Image.open(u'images/thesis_circles.jpg')
        # im = im.convert(u'L')
        # for i in range(130, 141, 1):
        #     for j in range(113, 126, 1):
        #         adjustments.append(np.asarray(im.crop((i, j, i+30, j+30))).reshape(900))
        # im.crop((135, 120, 165, 150))
        if args.shuffle_samples:
            random.shuffle(adjustments)
        print(u'will write out')
        if default_good_permutations:
            prefix_good = u'permutations_'
        else:
            prefix_good = u''
        with open(u'sample_data/{}eval_1_under_30.pkl'.format(prefix_good), 'wb') as circle_output:
            pickle.dump(adjustments, circle_output)
        print(u'finished writing, number: {}'.format(len(adjustments)))
        del adjustments
    if default_bad:
        one_circle_path = 'images/eval_circle.jpg'
        one_circle_path = '/home/jason/box/From_BrotherDevice/20160102123552_001_single.jpg'
        adjustments = generate_threshold_adjustments(one_circle_path, 80, 100, 3, rand_range=[255, 256], dark_cut_off=200, x_size=x_size, y_size=y_size)
        adjustments = [np.array(a)
                       for adj in adjustments
                       for a in adj]
        adjustments.append(np.array([255 for _ in range(x_size*y_size)]))
        if args.shuffle_samples:
            random.shuffle(adjustments)
        with open(u'sample_data/eval_90_under_100.pkl', 'wb') as circle_output:
            pickle.dump(adjustments, circle_output)

        # half_right = np.array(Image.open('images/eval_circle_right_half.jpg').convert(u'L')).reshape(x_size * y_size)
        # with open(u'sample_data/eval_half_right.pkl', 'wb') as circle_output:
        #     pickle.dump([half_right], circle_output)

        crops = crop_sliding_window(u'images/qr_codes_small.jpg', y_size, x_size, printing=False)
        qr_codes = []
        for c in crops:
            as_array = np.asarray(c.cr.convert(u'L'))
            qr_codes.append(as_array)
        with open(u'sample_data/qr_codes_small.pkl', 'wb') as circle_output:
            pickle.dump(qr_codes, circle_output)

        crops = crop_sliding_window(u'images/numbered_list_cropped_2.jpg', y_size, x_size, printing=False)
        qr_codes = []
        for c in crops:
            as_array = np.asarray(c.cr.convert(u'L'))
            qr_codes.append(as_array)
        with open(u'sample_data/numbered_list_cropped.pkl', 'wb') as circle_output:
            pickle.dump(qr_codes, circle_output)
    if default_ipsum:
        # crops = crop(u'sample_data/thesis_lorem_ipsum.jpg', 30, 30, printing=False)
        ipsums = []
        # for _ in range(12000):
        #     c = next(crops)
        #     as_array = np.asarray(c.convert(u'L'))
        #     ipsums.append(as_array.reshape(as_array.size))
        crops = crop_sliding_window(u'sample_data/thesis_lorem_ipsum_9.jpg', y_size, x_size, printing=False)
        for _ in range(1000):
            try:
                c = next(crops)
                for degrees in range(0, 180, 90):
                    im2 = c.cr.convert(u'RGBA')
                    rot = im2.rotate(degrees)
                    fff = Image.new('RGBA', rot.size, (255,) * 4)
                    out = Image.composite(rot, fff, rot)
                    as_array = np.asarray(out.convert(u'L'))
                    ipsums.append(as_array)
            except StopIteration:
                break
        crops = crop_sliding_window(u'sample_data/thesis_lorem_ipsum_9_1_5_line.jpg', y_size, x_size, printing=False)
        for _ in range(1000):
            try:
                c = next(crops)
                for degrees in range(0, 180, 90):
                    im2 = c.cr.convert(u'RGBA')
                    rot = im2.rotate(degrees)
                    fff = Image.new('RGBA', rot.size, (255,) * 4)
                    out = Image.composite(rot, fff, rot)
                    as_array = np.asarray(out.convert(u'L'))
                    ipsums.append(as_array)
            except StopIteration:
                break
        crops = crop_sliding_window(u'images/thesis_lorem_ipsum_9_2_line_cropped.jpg', y_size, x_size, printing=False)
        for _ in range(1000):
            try:
                c = next(crops)
                for degrees in range(0, 180, 90):
                    im2 = c.cr.convert(u'RGBA')
                    rot = im2.rotate(degrees)
                    fff = Image.new('RGBA', rot.size, (255,) * 4)
                    out = Image.composite(rot, fff, rot)
                    as_array = np.asarray(out.convert(u'L'))
                    ipsums.append(as_array)
            except StopIteration:
                break
        crops = crop_sliding_window(u'images/thesis_lorem_ipsum_7_2_line.jpg', y_size, x_size, printing=False)
        for _ in range(1000):
            try:
                c = next(crops)
                for degrees in range(0, 180, 90):
                    im2 = c.cr.convert(u'RGBA')
                    rot = im2.rotate(degrees)
                    fff = Image.new('RGBA', rot.size, (255,) * 4)
                    out = Image.composite(rot, fff, rot)
                    as_array = np.asarray(out.convert(u'L'))
                    ipsums.append(as_array)
            except StopIteration:
                break
        crops = crop_sliding_window(u'images/thesis_lorem_ipsum_7.5_2_line.jpg', y_size, x_size, printing=False)
        for _ in range(1000):
            try:
                c = next(crops)
                for degrees in range(0, 180, 90):
                    im2 = c.cr.convert(u'RGBA')
                    rot = im2.rotate(degrees)
                    fff = Image.new('RGBA', rot.size, (255,) * 4)
                    out = Image.composite(rot, fff, rot)
                    as_array = np.asarray(out.convert(u'L'))
                    ipsums.append(as_array)
            except StopIteration:
                break
        # crops = crop(u'images/eval-0-header_bad.jpg', 30, 30, printing=False)
        # len_before_header_bad = len(ipsums)
        # for c in crops:
        #     as_array = np.asarray(c.convert(u'L'))
        #     ipsums.append(as_array.reshape(as_array.size))
        if args.shuffle_samples:
            random.shuffle(ipsums)
        print(u'Number ipsums: {}'.format(len(ipsums)))
        with open(u'sample_data/lorem_ipsum_generated.pkl', 'wb') as circle_output:
            pickle.dump(ipsums, circle_output)
