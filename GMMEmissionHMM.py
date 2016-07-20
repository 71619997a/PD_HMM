"""This module defines the class GMMEmissionHMM."""
from hmmlearn import hmm


class GMMEmissionHMM:
    """This class defines a hidden Markov model with GMM emissions."""
    def __init__(
            self,
            num_states,
            num_gaussians,
            transition_probabilities=None,
            initial_state_probabilities=None):
        """Creates an HMM.

        Arguments:
        num_states -- the number of states.
        num_gaussians -- the number of gaussians in each GMM.
        transition_probabilities -- the transition probabilities for
            each state, in a 2D array.
        initial_state_probabilities -- initial probability for each
            state, in an array.
        """
        init_params = 'mcw'
        if transition_probabilities == None: init_params += 't'
        if initial_state_probabilities == None: init_params += 's'
        self.model = hmm.GMMHMM(n_components=num_states, n_mix=num_gaussians, 
            init_params=init_params)
        if not 's' in init_params: 
            self.model.startprob_ = initial_state_probabilities
        if not 't' in init_params: 
            self.model.transmat_ = transition_probabilities

    def train(self, train_data):
        """Trains the model with train_data, using the Baum-Welch algorithm.

        Arguments:
        train_data -- list of emissions, in a 2D array (length of data x
            num of emissions per state)
        """
        self.model.fit(train_data)

    def viterbi(self, sample_data):
        """Gets most likely state sequence from sample_data, given the
            trained model, using Viterbi algorithm.

        Arguments:
        sample_data -- list of emissions, in a 2D array
        Returns the most likely state sequence given the emissions as well
            as the logprob of the sequence.
        """
        return self.model.decode(sample_data)

    def score(self, real_data):
        """Scores the data, given the trained model.

        Arguments:
        real_data -- list of emissions, in a 2D array
        Returns the logprob of the data.
        """
        return self.model.score(real_data)

    def __str__(self):
        """Stringifies the model."""
        string = 'transmat_ = ' + str(self.model.transmat_)
        string += '\ngmms_ = ' + str(self.model.gmms_)
        return string
