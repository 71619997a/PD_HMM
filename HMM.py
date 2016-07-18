class HMM {
    """Hidden Markov Model with infinite emissions.
Variables:
num_states - defines num of hidden states
transition_probabilities - defines probability of transitioning from states to other states, dictionary format
emission_probabilities - defines distribution of emissions based on state, dictionary format
"""
    def __init__(self, num_states = None, transition_probabilities = None, emission_probabilities = None):
        #For unknown model, no parameters. For known model, all parameters.
        self.num_states = num_states
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        
    def baum_welch(self,emissions, max_states):
        """Performs Baum-Welch algorithm, using the emission history to determine most likely probabilities of transition and emission.
Parameters:
emissions - list of emissions
max_states - upper limit on the number of states
Returns nothing
"""
    
    def viterbi(self,emissions):
        """Performs Viterbi algorithm, using the emission history along with the defined model to determine most likely state history.
Parameters:
emissions - list of emissions
Returns list of states, same length as list of emissions."""
    