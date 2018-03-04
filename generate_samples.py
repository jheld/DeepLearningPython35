import pickle

from . setup_network import get_default_input


if __name__ == '__main__':
    tr, ev, te = get_default_input()
    pickle.dump(tuple([tr, ev, te]), open(u'sample_data/default_input_regular.pkl', 'wb'))
