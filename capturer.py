import cv2
import numpy as np

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(64, 64))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w+20, y+h+30), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                if self.model:
                    prediction = self.model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Video', cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(self.delay) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
