import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from cli import CustomParser
from cnn import CNN

BATCH_SIZE = 64
IMG_SHAPE = (48, 48, 1)
CLASSES_NUM = 7
EPOCHS = 50


def save_history(filename, history):
    with open(filename, 'wb') as f:
        pickle.dump(history, f)


def main():
    parser = CustomParser()
    args = parser.parse_args()

    train_gen = ImageDataGenerator(rescale=1.0/args.rescale_rate)
    validation_gen = ImageDataGenerator(rescale=1.0/args.rescale_rate)

    train_data = train_gen.flow_from_directory(
        args.train_dir,
        target_size=(48, 48),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    validation_data = validation_gen.flow_from_directory(
        args.validation_dir,
        target_size=(48, 48),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    cnn = CNN(classes_num=CLASSES_NUM, input_shape=IMG_SHAPE)
    cnn.model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=.0001, decay=1e-6)
    )

    model_data = cnn.model.fit_generator(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_steps=validation_data.samples // BATCH_SIZE,
        workers=8
    )

    save_history('history.pickle', model_data.history)

    cnn.model.save(args.outfile)


if __name__ == "__main__":
    main()
