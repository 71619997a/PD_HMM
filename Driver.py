from random import random
from NormalEmissionHMM import NormalEmissionHMM
transition = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.15, 0.1, 0.75]]
emission = [[1, 0.7], [3, 0.4], [5, 0.9]]
initial = [1, 0, 0]
hmm = NormalEmissionHMM(transition, emission, initial)
train_seq = [random() * 6 for i in range(3000)]
hmm.train(train_seq)
prac_seq = [random() * 6 for i in range(20)]
print(hmm.viterbi(prac_seq))
