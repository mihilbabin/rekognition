import cv2
import numpy as np



class Capturer:
    def __init__(self, filename=None, model=None):
        self.model = model
        if filename:
            self.input_stream = filename
            self.delay = 25
        else:
            self.input_stream = 0
            self.delay = 1

    def mainloop(self):
        # cv2.ocl.setUseOpenCL(False)
        cap = cv2.VideoCapture(self.input_stream)
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h+10), (255, 0, 0), 2)
                # roi_gray = gray[y:y + h, x:x + w]
                # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.imshow('Video', cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(self.delay) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
