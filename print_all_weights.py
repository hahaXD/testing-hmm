import numpy as np
import matplotlib
matplotlib.use('agg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
#Options
params = {'text.usetex' : True,
          'font.size' : 20,
          'text.latex.unicode': True,}
plt.rcParams.update(params) 
import json
import subprocess, shlex

def get_normalized_pr (input_layer):
    sum_value = 1 / np.sum(input_layer, axis=1)
    return np.einsum('i, ij->ij',sum_value, input_layer)

def get_normalized_pr_with_batch (input_layer):
    sum_value = 1 / np.sum(np.exp(input_layer), axis=2)
    return np.einsum("ij, ijk -> ijk", sum_value, np.exp(input_layer))

def np_softmax(x):
    ps = np.exp(x)
    ps /= np.sum(ps)
    return ps

class Weight:
    def __init__(self):
        prior_raw = np.array([-0.55127573,  0.04222124])
        self.prior = np_softmax(prior_raw)
        self.transition_coeff_param = np.array([[ 3.0428107, -1.3315939],[-0.6559177,  2.2320385]])
        self.transition_bias_param = np.array([[-0.9065993 ,  0.15516631], [0.24486056,  0.9013595 ]])
        emission = np.array( [[-0.27115414,  0.5332743 ],[ 1.6537137 , -0.53402674]])
        self.emission = np.stack([np_softmax(emission[i,:]) for i in range(0, 2)], axis=0)

def gets_next_q(x, prev_q, w):
    x_aux = np.stack([1-x, x], axis = 1)
    transition_raw = np.einsum('ij, jk ->ijk', prev_q, w.transition_coeff_param)
    transition_raw += np.einsum("ij, jk ->ijk", np.ones_like(prev_q), w.transition_bias_param)
    transition = get_normalized_pr_with_batch(transition_raw)
    post_before_evid = np.einsum("ij, ijk ->ik", prev_q, transition)
    next_q = get_normalized_pr(post_before_evid * np.matmul(x_aux, np.transpose(w.emission)))
    return next_q, transition

def apply_evid(x, cur_q, w):
    x_aux = np.stack([1-x, x], axis = 1)
    next_q = get_normalized_pr(cur_q * np.matmul(x_aux, np.transpose(w.emission)))
    return next_q


def get_transition_weights (x, w):
    prior = np_softmax(w.prior_raw)
    emission = np.stack([np_softmax(w.emission[i,:]) for i in range(0, 2)], axis=0)
    #q = get_normalized_pr(prior * np.matmul(x[]))

def generate_all_input():
    x = np.zeros(shape=(1<<7,7))
    y = np.zeros(shape=(1<<7))
    for idx, i in enumerate(np.ndindex(tuple([2 for i in range(7)]))):
        x[idx] = np.array(i)
    return np.concatenate((x, np.ones((1<<7, 1)) * 0.5), axis=1), y

w = Weight()
x, _ = generate_all_input()
next_q  = apply_evid(x[:,0], w.prior, w)
transitions = []
for i in range(1, 8):
    next_q, transition_matrix = gets_next_q(x[:,i], next_q, w)
    transitions.append(transition_matrix[:,0,1])

for i in range(1, 8):
    plt.scatter(np.ones(1 << 7) * i, transitions[i-1], c=(0,0,0))
plt.xlabel("Location of the hidden variable")
plt.ylabel(r"Selected $\theta_{H_i=1 \mid H_{i-1}=0}$", fontsize=20)
plt.tight_layout()
plt.savefig("param.pdf")
plt.clf()


