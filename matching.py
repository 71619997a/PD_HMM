import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment as lin


def replace(seq, old, new):
    newseq = []
    for i in seq:
        if i == old:
            newseq.append(new)
        else:
            newseq.append(i)
    return newseq


def build_matching_index(seq_constant, seq_guess, num_states):
    match_index = np.zeros((num_states, num_states))
    for i in range(num_states):
        unsorted = np.zeros(num_states)
        for j in range(num_states):  # n^3
            new_guess = replace(seq_guess, i, j)
            diffs = diff_list(seq_constant, new_guess)
            unsorted[j] = diffs
        match_index[i] = unsorted
    return match_index


def diff_list(l1, l2):
    diffs = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            diffs += 1
    return diffs


def best_pairings(match_index):
    pairings = lin(match_index)
    ret = {}
    for pair in pairings:
        ret[pair[0]] = pair[1]
    return ret

def best_seq(seq_constant, seq_guess, num_states):
    pairings = best_pairings(build_matching_index(seq_constant, seq_guess, num_states))
    return [pairings[i] for i in seq_guess]

if __name__ == "__main__":
    const = [0, 3, 2, 4, 2, 1]
    guess = [1, 4, 3, 1, 2, 3]
    print best_seq(const, guess, 5)
