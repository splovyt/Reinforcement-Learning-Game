import pickle

def save_dict(filepath, dictionary):
    with open(filepath, 'wb') as handle:
        pickle.dump(dictionary, handle)

def load_dict(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)