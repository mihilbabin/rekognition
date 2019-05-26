import argparse
import os

import pickle
from keras.models import load_model, model_from_json
from keras.metrics import categorical_accuracy

import capturer
import analyzer

def loading():
    with open('learning/model.json') as f:
        loaded = f.read()
    model = model_from_json(loaded)
    model.load_weights('learning/model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


def main():
    flags = parse_flags()
    if flags.analysis:
        with open(flags.analysis, 'rb') as f:
            data = pickle.load(f)
        a = analyzer.Analyzer(data)
        a.plot()
    elif flags.file and os.path.exists(flags.file):
        model = load_model('learning/model.1.h5')
        c = capturer.Capturer(flags.file, model=model)
        c.mainloop()
    else:
        model = load_model('learning/model.1.h5')
        # model = loading()
        c = capturer.Capturer(0, model=model)
        c.mainloop()


def parse_flags():
    parser = argparse.ArgumentParser(
        description='Emotion recognizer',
        epilog='Interactively recognizes emotions'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--file', help='Video file for recognition')
    group.add_argument('-a', '--analysis', help='Analyze passed training history')
    return parser.parse_args()


if __name__ == "__main__":
    main()
