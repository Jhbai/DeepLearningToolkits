import numpy as np
import pandas as pd
class TrainValidation:
    def __init__(self, x, y, ratio = 0.5):
        CLASSES = list(set(y))
        self.ratio = ratio
        self.x, self.y = x, y
        labels_idx = self.class_seperate_and_store(y, CLASSES)
        self.idx_tr, self.idx_val = self.label_split_after_seperation(labels_idx)
        
    def Data(self):
        idx_tr, idx_val = self.idx_tr, self.idx_val
        x, y = self.x, self.y
        return x[idx_tr], y[idx_tr], x[idx_val], y[idx_val]

    def class_seperate_and_store(self, y, labels):
        result = list()
        for label in labels:
            result.append([i for i in range(y.shape[0]) if y[i] == label])
        return result

    def label_split_after_seperation(self, labels_idx):
        ratio, Labels_tr, Labels_val = self.ratio, list(), list()
        for idxs in labels_idx:
            I1 = [int(i) for i in np.random.choice(idxs, size = (int(ratio*len(idxs)), 1))]
            I2 = [int(idx) for idx in idxs if idx not in I1]
            Labels_tr += I1
            Labels_val += I2
        return Labels_tr, Labels_val
