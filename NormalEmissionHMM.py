"""This module defines the class NormalEmissionHMM."""
from hmmlearn import hmm


class NormalEmissionHMM:
    """This class defines a hidden Markov model with Gaussian emissions."""

    def __init__(
            self,
            transition_probabilities,
            emission_means,
            emission_covars,
            initial_state_probabilities):
        """Creates an HMM.

        Arguments:
        transition_probabilities -- the transition probabilities for
            each state, in a 2D array.
        emission_means -- means of the Gaussian distributions of
            emissions for each state, in a 2D array.
        emission_covars -- covariance matrices for each state x
            emissions, in a 3D array (array of matrices).
        initial_state_probabilities -- initial probability for each
            state, in an array.
        """
        self.model = hmm.GaussianHMM(n_components=len(transition_probabilities),
                                     # startprob_prior=initial_state_probabilities,
                                     # transmat_prior=transition_probabilities,
                                     # init_params='stmc'
                                    )
        self.model.startprob_ = initial_state_probabilities
        self.model.transmat_ = transition_probabilities
        self.model.means_ = emission_means
        self.model.covars_ = emission_covars
        self.model.n_features = len(emission_means[0])

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
        string += '\nmeans_ = ' + str(self.model.means_)
        string += '\ncovars_ = ' + str(self.model.covars_)
        return string
