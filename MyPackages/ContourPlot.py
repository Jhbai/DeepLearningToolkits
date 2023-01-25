import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Graph:
    def __init__(self, model, X, y):
        """
        'Model': sklearn_model
        'Max': max(X array)
        'Min': min(X array)
        'Input': X array
        'Label': y array
        """
        self.parameters = {'Model': model,
                           'Max': [np.max(X[:, 0]), np.max(X[:, 1])],
                           'Min': [np.min(X[:, 0]), np.min(X[:, 1])],
                           'Input': X,
                           'Label': y}

    def plot(self, revolution=100):
        para = self.parameters
        Model, Max, Min, Input, Label = para['Model'], para['Max'], para['Min'], para['Input'], para['Label']
        dx = np.linspace(Min[0], Max[0], revolution)
        dy = np.linspace(Min[1], Max[1], revolution)
        dx, dy = np.meshgrid(dx, dy)
        z = Model.predict(np.c_[dx.flatten(), dy.flatten()]).reshape(dx.shape)
        plt.figure(figsize = (6, 4))
        plt.contourf(dx, dy, z, alpha = 1)
        plt.scatter(Input[np.where(Label == 0), 0],\
                    Input[np.where(Label == 0), 1], color = 'blue', label = 'Negative')
        plt.scatter(Input[np.where(Label == 1), 0],
                    Input[np.where(Label == 1), 1], color = 'red', label = 'Positive')
        plt.legend()
        plt.grid()
        plt.show()