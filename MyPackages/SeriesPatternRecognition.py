import numpy as np
from fastdtw import fastdtw as fd
import matplotlib.pyplot as plt
from tqdm import tqdm

class PatternRecognition:
    def __init__(self, templates, start, end):
        self.pattern = templates[start: end]
        self.templates, self.start, self.end = templates, start, end
    
    def recognize(self, seq, StepSize):
        # Initialization
        paras = {'Pattern': self.pattern,
                 'Windows': self.WindowSetUp(self.pattern.shape[0], StepSize),
                 'StepSize':StepSize,
                 'Result': list(),
                 'Scores': list()
                 }
        # DTW Computing
        for WinSize in paras['Windows']:
            paras['Result'], paras['Scores'] = self.DTWResult(seq, paras['Pattern'], 
                                                              WinSize, paras['Result'],
                                                              paras['Scores'])
        self.Result, self.Scores = self.Sort(paras['Result'], paras['Scores'])
        # Plot Result
        plt.figure(figsize = (6, 4))
        ## First Part
        plt.subplot(2, 1, 1)
        plt.plot(self.templates, label = 'Time Series', alpha = 0.4, color = 'blue')
        IDX = [i for i in range(self.start, self.end)]
        plt.plot(IDX, self.templates[self.start: self.end], label = 'PatternGroundTruth',
                 color = 'blue')
        plt.legend()
        plt.grid()
        ## Second Part
        plt.subplot(2, 1, 2)
        plt.plot(seq, label = 'Time Series', alpha = 0.4, color = 'blue')
        IDX = [i for i in range(self.Result[0][0], self.Result[0][1])]
        plt.plot(IDX, seq[self.Result[0][0]: self.Result[0][1]], label = 'PatternPrediction',
                 color = 'blue')
        plt.xlabel('time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()
        
    def WindowSetUp(self, PatternSize, StepSize):
        return [size for size in range(PatternSize - StepSize, PatternSize + StepSize)]

    def DTWResult(self, seq, Pattern, WinSize, Result, Scores):
        with tqdm(total = (seq.shape[0]-WinSize), leave = False, desc = 'DTW Processing') as pbar:
            for start in range(seq.shape[0]-WinSize):
                Result.append([start, start+WinSize])
                Scores.append(fd(seq[start: start+WinSize], Pattern)[0])
                pbar.update()
        return Result, Scores
    
    def Sort(self, Result, Scores):
        I = np.argsort(Scores)
        return np.array(Result)[I], np.array(Scores)[I]

    def update(self, pattern):
        self.pattern = pattern