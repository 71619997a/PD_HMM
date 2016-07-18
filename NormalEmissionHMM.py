import ghmm
class NormalEmissionHMM:
    def __init__(self, transition_probabilities, emission_distributions, initial_state_probabilities):
        self.sigma = ghmm.Float()
        self.hmm = ghmm.HMMFromMatrices(self.sigma, ghmm.GaussianDistribution(self.sigma), transition_probabilities, emission_distributions, initial_state_probabilities)
    
    def train(self, sequence):
        self.hmm.baumWelch(self.sigma, sequence)
    
    def viterbi(self, sequence):
        return self.hmm.viterbi(sequence)
