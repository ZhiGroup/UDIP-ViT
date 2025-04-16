import matplotlib.pyplot as plt
import torch

class PlotLossesTrain():

    def __init__(self, losses):
        self.losses = losses

    def plot_losses(self):
        epochs = range(1, len(self.losses) + 1)
        losses_cpu = [loss.item() for loss in self.losses]
        plt.plot(epochs, losses_cpu, label="Training Loss")
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
