import numpy as np
from fastdtw import fastdtw as fd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import time

class ParallelPatternRecognition:
    def __init__(self, templates, start, end):
        self.pattern = templates[start: end]
        self.templates, self.start, self.end = templates, start, end
        ## Parallelism Tools
        self.TEMP_RESULT, self.TEMP_SCORES = multiprocessing.Manager().list(), multiprocessing.Manager().list()
        self.lock = multiprocessing.Lock()
    
    def recognize(self, seq, StepSize):
        # Initialization
        paras = {'Pattern': self.pattern,
                 'Windows': self.WindowSetUp(self.pattern.shape[0], StepSize),
                 'StepSize':StepSize,
                 'Result': list(),
                 'Scores': list()
                 }
        # DTW ParallelComputing
        processes = list()
        for i, WinSize in enumerate(paras['Windows']):
            processes.append(multiprocessing.Process(target=self.DTWResult, args = (seq, paras['Pattern'], WinSize)))
            processes[i].start()
        for i in range(len(processes)):
            processes[i].join()
        
        for alist in self.TEMP_RESULT:
            paras['Result'] += alist
        for alist in self.TEMP_SCORES:
            paras['Scores'] += alist
        
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

    def DTWResult(self, seq, Pattern, WinSize):
        #with tqdm(total = (seq.shape[0]-WinSize), leave = False, desc = 'DTW Processing') as pbar:
        Result = list()
        Scores = list()
        for start in range(seq.shape[0]-WinSize):
            Result.append([start, start+WinSize])
            Scores.append(fd(seq[start: start+WinSize], Pattern)[0])
        self.lock.acquire()
        self.TEMP_RESULT.append(Result)
        self.TEMP_SCORES.append(Scores)
        self.lock.release()
            
        return Result, Scores
    
    def Sort(self, Result, Scores):
        I = np.argsort(Scores)
        return np.array(Result)[I], np.array(Scores)[I]

    def update(self, pattern):
        self.pattern = pattern

if __name__ == '__main__':
    start = time.time()
    #from SeriesPatternRecognition import PatternRecognition
    dX = (0.02*5.0*1/365 + 0.4*5.0*np.random.normal(0, 1, size = (365, 1)))
    Series1 = 5.0 + np.cumsum(dX)
    dX = (0.02*5.0*1/365 + 0.4*5.0*np.random.normal(0, 1, size = (365, 1)))
    Series2 = 5.0 + np.cumsum(dX)

    plt.figure(figsize = (4, 3))
    plt.plot(Series1, color = 'blue', label = 'Price 2')
    plt.plot(Series2, color = 'red', label = 'Price 2')
    plt.grid()
    plt.legend()
    plt.show()
    model = ParallelPatternRecognition(Series1, Series1.shape[0]-100, Series1.shape[0])
    model.recognize(Series2, 10)
    end = time.time()
    print(end - start)