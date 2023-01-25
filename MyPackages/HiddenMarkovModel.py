from hmmlearn import hmm
import numpy as np

class HiddenMarkovModel:
    def __init__(self, TimeSeries, n_states):
        self.model = hmm.GaussianHMM(n_components = n_states,
                                     covariance_type = "full",
                                     n_iter = 1000,
                                     tol = 0.001)
        self.model.fit(TimeSeries)

    def state(self, InputTimeSeries):
        return self.model.decode(InputTimeSeries, algorithm = 'viterbi')[1]
    
    def Performance(self, InputTimeSeries):
        return self.model.score(InputTimeSeries)
    
    def info(self):
        prob1, prob2, prob3 = self.model.startprob_, self.model.transmat_, self.model.emissionprob_
        print("[0] is Start Probability\n[1] is Transfer Probability\n[2] is Series Probability Conditional on State")
        return prob1, prob2, prob3