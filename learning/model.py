from keras.models import Sequential


class RecognitionModel(Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
