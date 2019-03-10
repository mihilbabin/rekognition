import os
import json

import numpy as np

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')


def read_training_data(x_filename, y_filename):
    x_train_file = os.path.join(INPUT_DIR, x_filename)
    y_train_file = os.path.join(INPUT_DIR, y_filename)
    x_train = np.load(x_train_file).astype('float32')
    y_train = np.load(y_train_file)
    return x_train, y_train


def save_model(model, filename='binary_cnn'):
    out = os.path.join(INPUT_DIR, 'results', f"{filename}.json")
    with open(out, 'w') as f:
        json.dump(model.to_json(), f)


def save_config(config, filename='binary_cnn_conf'):
    out = os.path.join(INPUT_DIR, 'results', f"{filename}.txt")
    with open(out, 'w') as f:
        f.write(f"{str(config)}\n")


def save_result(train_val_accuracy, filename='binary_cnn_res'):
    out = os.path.join(INPUT_DIR, 'results', f"{filename}.txt")
    with open(out, 'w') as f:
        f.write(f"{str(train_val_accuracy)}\n")
