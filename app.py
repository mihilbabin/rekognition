import argparse
import os

import capturer


def main():
    flags = parse_flags()
    if flags.file and not os.path.exists(flags.file):
        raise FileNotFoundError('Source input not found')
    c = capturer.Capturer(flags.file)
    c.mainloop()


def parse_flags():
    parser = argparse.ArgumentParser(
        description='Emotion recognizer',
        epilog='Interactively recognizes emotions'
    )
    parser.add_argument('-f', '--file', help='Video file for recognition')
    return parata(x_train, y_train, output_dir)ser.parse_args()


if __name__ == "__main__":
    main()
