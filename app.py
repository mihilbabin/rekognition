import argparse
import cv2

def main():
    flags = parse_flags()
    if flags.file is None:
        print('Capture WebCam')
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('Capture file')

def parse_flags():
    parser = argparse.ArgumentParser(
        description='Emotion recognizer',
        epilog='Interactively recognizes emotions'
    )
    parser.add_argument('-f', '--file', help='Video file for recognition')
    return parser.parse_args()

if __name__ == "__main__":
    main()