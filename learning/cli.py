import argparse


class CustomParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description='Handle Dataset into Keras Model',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._fill_arguments()

    def _fill_arguments(self):
        self.add_argument(
            'train_dir', help='Absolute Path of the Training Dataset Images'
        )
        self.add_argument(
            'validation_dir', help='Absolute Path of the Validation Dataset Images'
        )
        self.add_argument(
            '-o', '--outfile',
            dest='outfile',
            default='model.h5',
            help='Name of the output model weights'
        )
        self.add_argument(
            '-t', '--training',
            dest='training_size',
            type=int,
            default=80,
            help='Percent of dataset to use for Training'
        )
        self.add_argument(
            '-r', '--rescale',
            dest='rescale_rate',
            type=int,
            default=255,
            help='Each image rescale rate'
        )

