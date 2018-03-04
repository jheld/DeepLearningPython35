import argparse
import pickle

from . setup_network import get_default_input


if __name__ == '__main__':
    a_p = argparse.ArgumentParser()
    a_p.add_argument(u'--default_good_permutations', default=False)
    a_p.add_argument(u'--binary_classifier', default=1)
    args = a_p.parse_args()
    is_multi_class = bool(int(args.binary_classifier))
    default_good_permutations = int(args.default_good_permutations) \
        if isinstance(args.default_good_permutations, str) else args.default_good_permutations

    tr, ev, te = get_default_input(good_permutations=default_good_permutations, multi_class=is_multi_class)
    pickle.dump(tuple([tr, ev, te]), open(u'sample_data/default_input_regular.pkl', 'wb'))
