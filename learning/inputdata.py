import os
import numpy as np

from keras.utils.np_utils import to_categorical
import pandas as pd

import decorators

FER_EMOTIONS_LIST = {
    'Angry': 0,
    'Disgust': 0,
    'Fear': 0,
    'Happy': 0,
    'Sad': 0,
    'Surprise': 0,
    'Neutral': 0,
}

EMOTIONS_LIST = [
    'Angry', 'Fear', 'Happy', 'Disgust',
    'Sad', 'Surprise', 'Neutral'
]


def reconstruct(pix):
    pix_arr = np.fromiter(map(int, pix.split()), dtype=np.int)
    return pix_arr.reshape((48, 48))


@decorators.timelog('Data preparation')
def load_data(filepath, usage, sample_rate=0.3):
    df = pd.read_csv(filepath)
    df = df[df.Usage == usage]
    frames = [df[df['emotion'] == FER_EMOTIONS_LIST[e]] for e in EMOTIONS_LIST]
    data = pd.concat(frames)
    rows = np.random.choice(data.index.values, int(len(data)*sample_rate))
    data = data.loc[rows]
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.pixels])
    x_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    y_train = data.emotion.copy()
    y_train.loc[y_train == 1] = 0
    for new, e in enumerate(filter(lambda x: x != 'Disgust', EMOTIONS_LIST)):
        y_train.loc[y_train == FER_EMOTIONS_LIST[e]] = new
    y_train = to_categorical(y_train.values)
    return x_train, y_train


@decorators.timelog('Saving data')
def save_data(x_train, y_train, filepath):
    np.save(f"{os.path.join(filepath, 'x_train')}", x_train)
    np.save(f"{os.path.join(filepath, 'y_train')}", y_train)


if __name__ == "__main__":
    input_filepath = os.path.join(os.path.dirname(__file__), 'data', 'fer2013.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    usage = 'PrivateTest'
    x_train, y_train = load_data(input_filepath, usage, sample_rate=1)
    save_data(x_train, y_train, output_dir)