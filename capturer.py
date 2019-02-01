import cv2

class Capturer:
    def __init__(self, filename=None):
        if filename:
            self.input_stream = filename
            self.delay = 25
        else:
            self.input_stream = 0
            self.delay = 1

    def mainloop(self):
        cap = cv2.VideoCapture(self.input_stream)
        while True:
            _, frame = cap.read()

            cv2.imshow('frame', frame)
            if cv2.waitKey(self.delay) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()