import numpy as np
import torch
import torch.nn as nn
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.pyplot as plt

class draw:
    def __init__(self, img_flag):
        if not img_flag:
            self.plots = list()
        else:
            self.imgs = list()
        self.flag = (img_flag >= 1)
    
    def fit(self, Input):
        if self.flag:
            self.imgs.append(Input)
        else:
            self.plots.append(Input)

    def show(self):
        fig = plt.figure(figsize=(8,8))
        if self.flag:
            plt.axis("off")
            ani = animation.ArtistAnimation(fig, self.imgs, interval=1000, repeat_delay=1000, blit=True)
        else:
            ani = animation.ArtistAnimation(fig, self.plots, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())

    def help(self):
        if not self.flag:
            temp = 'self.plots.append(plt.plot(pred.reshape(-1,).cpu().detach(), color = '
            temp += '\'red\', label = \'prediction\', animated = True)[0])'
            print(temp)
        else:
            temp = '[plt.imshow(np.transpose(img,(1,2,0)), animated=True)]'
            print(temp)

            