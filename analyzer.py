import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

class Analyzer:
    def __init__(self, history):
        self.history = history
    
    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(15,5))
        axs[0].plot(self.history['acc'])
        axs[0].plot(self.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        ticks = np.arange(1, len(self.history['acc']) + 1)
        axs[0].set_xticks(ticks, len(self.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        axs[1].plot(self.history['loss'])
        axs[1].plot(self.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        ticks = np.arange(1, len(self.history['loss'])+1)
        axs[1].set_xticks(ticks, len(self.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        fig.canvas.set_window_title('History plot')
        fig.savefig('plot.png')
        plt.show()