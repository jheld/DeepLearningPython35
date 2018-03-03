import pickle
import random
from itertools import chain
import numpy

import network2
from DenoisingAutoEncoder.DenoisingAutoEncoder import utils
from DenoisingAutoEncoder.DenoisingAutoEncoder.SDA_layers import StackedDA

from get_circle import training_evaluation_test_split, crop

from setup_network import get_formatted_input, get_formatted_input_not_training, parse_crops


def back_to_greyscale(data):
    data = numpy.array([255 - i * 255 for i in data.reshape(data.size)]).reshape(*data.shape)
    return data


def another_attempt():
    tr_good, ev_good, te_good = training_evaluation_test_split((0.7, 0.15, 0.15), 'sample_data/eval_2_under_40.pkl')
    tr_bad, ev_bad, te_bad = training_evaluation_test_split((0.7, 0.15, 0.15), 'sample_data/eval_90_under_100.pkl')
    formatted_data = []
    for data_set in zip([tr_good, tr_bad, ], [1, 0, ]):
        formatted_data.extend(
            get_formatted_input(*data_set, convert_scale=True, multi_class=True, use_inner_array=True))
    formatted_rest_data = []
    for data_set in zip([ev_good, te_good, ev_bad, te_bad, ], [1, 1, 0, 0, ]):
        formatted_rest_data.extend(
            get_formatted_input_not_training(*data_set, convert_scale=True, use_inner_array=True))
    random.shuffle(formatted_data)
    random.shuffle(formatted_rest_data)
    net = network2.Network([900, 30, 2])
    result = net.SGD(training_data=formatted_data, epochs=15, mini_batch_size=10, eta=3,
                     evaluation_data=formatted_rest_data,
                     monitor_training_cost=True,
                     monitor_training_accuracy=True,
                     monitor_evaluation_cost=True,
                     monitor_evaluation_accuracy=True)
    print(result)
    return result, net


def sda_attempt(noise_rate=0.3, pre_train_epoch=1, final_epoch=2, fine_tune_epoch=2):

    if True:
        tr_good, ev_good, te_good = training_evaluation_test_split((0.7, 0.15, 0.15),
                                                                   u'sample_data/permutations_eval_1_under_30.pkl')
        tr_bad, ev_bad, te_bad = training_evaluation_test_split((0.7, 0.15, 0.15), 'sample_data/eval_90_under_100.pkl')
        tr_bad_lorem_gen, ev_bad_lorem_gen, te_bad_lorem_gen = training_evaluation_test_split((0.7, 0.15, 0.15), 'sample_data/lorem_ipsum_generated.pkl')
        tr_bad += tr_bad_lorem_gen
        ev_bad += ev_bad_lorem_gen
        te_bad += te_bad_lorem_gen
        formatted_data = []
        for data_set in zip([tr_good, tr_bad, ], [1, 0, ]):
            formatted_data.extend(get_formatted_input(*data_set, convert_scale=True, multi_class=True, use_inner_array=False))
        formatted_rest_data = []
        for data_set in zip([ev_good, ev_bad, te_good, te_bad, ], [1, 0, 1, 0, ]):
            formatted_rest_data.extend(get_formatted_input(*data_set, convert_scale=True, multi_class=True, use_inner_array=False))
    else:
        tr_ten, ev_ten, te_ten = training_evaluation_test_split((0.70, 0.15, 0.15), 'sample_data/eval_ten_samples.pkl')
        tr_twenty, ev_twenty, te_twenty = training_evaluation_test_split((0.70, 0.15, 0.15), 'sample_data/eval_twenty_samples.pkl')
        tr_thirty, ev_thirty, te_thirty = training_evaluation_test_split((0.70, 0.15, 0.15), 'sample_data/eval_thirty_samples.pkl')
        tr_ninety, ev_ninety, te_ninety = training_evaluation_test_split((0.70, 0.15, 0.15), 'sample_data/eval_ninety_samples.pkl')
        formatted_data = []
        for data_set in zip([tr_ten, tr_twenty, tr_thirty, tr_ninety], [1, 1, 1, 0]):
            formatted_data.extend(get_formatted_input(*data_set, convert_scale=True, multi_class=True, use_inner_array=False))
        formatted_rest_data = []
        for data_set in zip([ev_ten, ev_twenty, ev_thirty, ev_ninety, te_ten, te_twenty, te_thirty, te_ninety], [1, 1, 1, 0, 1, 1, 1, 0]):
            formatted_rest_data.extend(get_formatted_input(*data_set, convert_scale=True, multi_class=True, use_inner_array=False))
    random.shuffle(formatted_data)
    random.shuffle(formatted_rest_data)
    formatted_data, formatted_labels = numpy.array([i[0] for i in formatted_data]), numpy.array([i[1] for i in formatted_data])

    # building the SDA
    sDA = StackedDA([100])

    # pre-trainning the SDA
    sDA.pre_train(formatted_data[:100], noise_rate=noise_rate, epochs=pre_train_epoch)

    # saving a PNG representation of the first layer
    W = sDA.Layers[0].W.T[:, 1:]
    utils.saveTiles(W, img_shape=(30, 30), tile_shape=(10, 10), filename="results/res_dA.png")

    # adding the final layer
    sDA.finalLayer(formatted_data, formatted_labels, epochs=final_epoch)

    # trainning the whole network
    sDA.fine_tune(formatted_data, formatted_labels, epochs=fine_tune_epoch)
    formatted_rest_data, formatted_rest_labels = numpy.array([i[0] for i in formatted_rest_data]), numpy.array([i[1] for i in formatted_rest_data])

    # predicting using the SDA
    pred = sDA.predict(formatted_rest_data).argmax(1)

    # let's see how the network did
    formatted_rest_labels = formatted_rest_labels.argmax(1)
    e = 0.0
    for label_value, pred_value in zip(formatted_rest_labels, pred):
        e += label_value == pred_value
        if label_value != pred_value:
            print(label_value, pred_value)

    # printing the result, this structure should result in 80% accuracy
    print("accuracy: %2.2f%%" % (100 * e / len(formatted_rest_labels)))

    return sDA


def crops_that_are_good(source_file_path, sDA, multi_class=True):
    crops = crop(source_file_path, 30, 30)
    pcs = parse_crops(crops, sDA)
    m_index = 1 if multi_class else 0
    for pc in pcs:
        if pc[-1][0][m_index] > 0.5:
            yield pc
