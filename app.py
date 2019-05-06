import argparse
import os

from keras.models import load_model

import capturer

def main():
    flags = parse_flags()
    if flags.file and not os.path.exists(flags.file):
        raise FileNotFoundError('Source input not found')
    model = load_model('model.h5')
    c = capturer.Capturer(flags.file, model=model)
    c.mainloop()


def parse_flags():
    parser = argparse.ArgumentParser(
        description='Emotion recognizer',
        epilog='Interactively recognizes emotions'
    )
    parser.add_argument('-f', '--file', help='Video file for recognition')
    return parser.parse_args()


if __name__ == "__main__":
    main()
