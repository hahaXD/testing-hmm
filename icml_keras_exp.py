from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import sys
import math
from tensorflow.python import debug as tf_debug
"""
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 2
"""

def get_normalized_pr (input_layer):
    sum_value = 1 / tf.reduce_sum(input_layer, axis=1)
    return tf.einsum('i, ij->ij',sum_value, input_layer)

def get_normalized_pr_exp (input_layer, scope=None):
    sum_value = 1 / tf.reduce_sum(tf.exp(input_layer), axis=1)
    if scope != None:
        return tf.einsum('i, ij->ij',sum_value, tf.exp(input_layer), name=scope)
    else:
        return tf.einsum('i, ij->ij',sum_value, tf.exp(input_layer))

def get_normalized_pr_with_batch (input_layer):
    sum_value = 1 / tf.reduce_sum(tf.exp(input_layer), axis=2)
    return tf.einsum("ij, ijk -> ijk", sum_value, tf.exp(input_layer))

class EvidenceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EvidenceLayer, self).__init__()

    def build(self, input_shape):
        self.cardinality = int(input_shape[0][-1])
        self.emission_weight_raw = self.add_variable("emission_weight_raw", shape=[self.cardinality, 2], initializer=tf.keras.initializers.get("glorot_uniform"))

    def call(self, inputs):
        h = inputs[0]
        x = inputs[1]
        aug_x = tf.stack([1-x, x], axis=1)
        with tf.name_scope("evidence_param") as scope:
            emission  = tf.stack([tf.nn.softmax(self.emission_weight_raw[i,:]) for i in range(0, self.cardinality)], axis=0, name=scope)
        return get_normalized_pr(h * tf.matmul(aug_x, tf.transpose(emission)))

class TestingTransitionLayer(tf.keras.layers.Layer):
    def __init__ (self):
        super(TestingTransitionLayer, self).__init__()

    def build(self, input_shape):
        self.cardinality = int(input_shape[-1])
        self.transition_param_coeff = self.add_variable("transition_param_coeff", shape=[self.cardinality, self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform"))
        self.transition_param_bias = self.add_variable("transition_param_bias", shape=[self.cardinality, self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform"))
    def call(self, last_h):
        transition_raw = tf.einsum('ij, jk ->ijk', last_h, self.transition_param_coeff)
        transition_raw += tf.einsum("ij, jk ->ijk", tf.ones_like(last_h), self.transition_param_bias)
        transition = get_normalized_pr_with_batch(transition_raw)
        post_before_evid = tf.einsum("ij, ijk ->ik", last_h, transition)
        return post_before_evid

class TestingSigmoidalTransitionLayer(tf.keras.layers.Layer):
    def __init__ (self, clip = False, fixed_gamma = False):
        super(TestingSigmoidalTransitionLayer, self).__init__()
        self.clip = clip
        self.fixed_gamma = fixed_gamma

    def build(self, input_shape):
        self.cardinality = int(input_shape[-1])
        self.positive_transition_param_raw = self.add_variable("positive_transition_param_raw", shape=[self.cardinality, self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform"))
        self.negative_transition_param_raw = self.add_variable("negative_transition_param_raw", shape=[self.cardinality, self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform"))
        self.positive_transition_param = get_normalized_pr_exp(self.positive_transition_param_raw)
        self.negative_transition_param = get_normalized_pr_exp(self.negative_transition_param_raw)
        if self.clip:
            self.threshold = self.add_variable("threshold", shape=[self.cardinality], constraint=tf.keras.constraints.MaxNorm(max_value=0.5))+0.5
        else:
            self.threshold = tf.math.sigmoid(self.add_variable("threshold_raw", shape=[self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform")))
        if self.fixed_gamma:
            self.gamma = 8
        else:
            self.gamma = self.add_variable("gamma",shape=[1], constraint=tf.keras.constraints.MinMaxNorm(0, 8))

    def call(self, last_h):
        frac = tf.math.sigmoid((last_h - self.threshold) * self.gamma)
        transition = tf.einsum("ij, jk ->ijk", frac,  self.positive_transition_param) +tf.einsum("ij,jk->ijk", (1-frac), self.negative_transition_param)
        post_before_evid = tf.einsum("ij, ijk ->ik", last_h, transition)
        return post_before_evid


class RegularTransitionLayer(tf.keras.layers.Layer):
    def __init__ (self):
        super(RegularTransitionLayer, self).__init__()

    def build(self, input_shape):
        self.cardinality = int(input_shape[-1])
        self.transition_param_raw = self.add_variable("transition_param_raw", shape=[self.cardinality, self.cardinality], initializer=tf.keras.initializers.get("glorot_uniform"))
    def call(self, last_h):
        with tf.name_scope("regular_transition_param") as scope:
            transition = get_normalized_pr_exp(self.transition_param_raw, scope)
        post_before_evid = tf.einsum("ij, jk ->ik", last_h, transition)
        return post_before_evid

class PriorLayer(tf.keras.layers.Layer):
    def __init__ (self, cardinality):
        super(PriorLayer, self).__init__()
        self.cardinality = cardinality
        self.prior_weights_raw = self.add_variable("prior_weight_raw", shape=[self.cardinality],initializer=tf.keras.initializers.get("glorot_uniform"))
    def call(self, x):
        return tf.nn.softmax(self.prior_weights_raw)

class TestingHMM(tf.keras.Model):
    def __init__(self, cardinality, chain_length):
        super(TestingHMM, self).__init__()
        self.chain_length = chain_length
        self.cardinality = cardinality
        self.prior_layer = PriorLayer(self.cardinality)
        self.transition_layer = TestingTransitionLayer()
        self.evidence_layer = EvidenceLayer()

    def call(self, inputs):
        self.transition_prs = []
        post = self.evidence_layer([self.prior_layer(None), inputs[:,0]])
        for i in range(1, self.chain_length):
            post_before_evid = self.transition_layer(post)
            post = self.evidence_layer([post_before_evid, inputs[:,i]])
        return tf.debugging.check_numerics(post, "testing hmm bad")

class TestingSigmoidalHMM(tf.keras.Model):
    def __init__(self, cardinality, chain_length, clip, fixed_gamma):
        super(TestingSigmoidalHMM, self).__init__()
        self.chain_length = chain_length
        self.cardinality = cardinality
        self.prior_layer = PriorLayer(self.cardinality)
        self.transition_layer = TestingSigmoidalTransitionLayer(clip, fixed_gamma)
        self.evidence_layer = EvidenceLayer()

    def call(self, inputs):
        self.transition_prs = []
        post = self.evidence_layer([self.prior_layer(None), inputs[:,0]])
        for i in range(1, self.chain_length):
            post_before_evid = self.transition_layer(post)
            post = self.evidence_layer([post_before_evid, inputs[:,i]])
        return tf.debugging.check_numerics(post, "testing hmm bad")

class RegularHMM(tf.keras.Model):
    def __init__(self, cardinality, chain_length):
        super(RegularHMM, self).__init__()
        self.chain_length = chain_length
        self.cardinality = cardinality
        self.prior_layer = PriorLayer(self.cardinality)
        self.transition_layer = RegularTransitionLayer()
        self.evidence_layer = EvidenceLayer()

    def call(self, inputs):
        prior_layer_value = tf.debugging.check_numerics(self.prior_layer(None), "prior")
        post = tf.debugging.check_numerics(self.evidence_layer([prior_layer_value, inputs[:,0]]), "after evidence");
        for i in range(1, self.chain_length):
            post = tf.debugging.check_numerics(self.transition_layer(post), "after %s, transition" % i)
            post = tf.debugging.check_numerics(self.evidence_layer([post, inputs[:,i]]), "after %s, evidence" %i)
        return tf.debugging.check_numerics(post, "regular hmm bad")

def generate_records (transition_matrix, emission, record_size, chain_size, evidence_size = None):
    transition_matrix = np.array(transition_matrix)
    hmm_order = len(transition_matrix.shape) - 1
    records = []
    labels = []
    for i in range(0, record_size):
        cur_record = []
        cur_obs = []
        for j in range(0, hmm_order):
            cur_record.append(int(random.random() > 0.5))
        for j in range(hmm_order, chain_size):
            transition_pr = transition_matrix[tuple(cur_record[-hmm_order:])]
            assert transition_pr.shape[0] == 2
            if random.random() > transition_pr[0]:
                cur_record.append(1)
            else:
                cur_record.append(0)
        evidence_size = chain_size - 1 if evidence_size is None else evidence_size
        evidence_size = min(chain_size - 1, evidence_size)
        for j in range(0, chain_size):
            cur_emission = emission[cur_record[j]]
            if random.random() > cur_emission[0]:
                cur_obs.append(1.0)
            else:
                cur_obs.append(0.0)
        for j in range(evidence_size, chain_size):
            cur_obs[j] = 0.5
        records.append(cur_obs)
        labels.append([1, 0] if cur_record[-1] == 0 else [0, 1])
    return records, labels

def icml_exp_with_single_configuration (transition_matrix_index, log_fname):
    chain_length = 8
    evidence_length = 7
    transition_state = np.unravel_index(transition_matrix_index, [2 for i in range(0, 8)])
    transition_matrix = np.zeros((2,2,2,2));
    log_dirs = "logs/%s/"%transition_matrix_index
    for k, cur_value in enumerate(transition_state):
        cur_index = np.unravel_index(k, (2, 2, 2))
        transition_matrix[cur_index][0] = 0.01 if cur_value == 0 else 0.99
        transition_matrix[cur_index][1] = 0.99 if cur_value == 0 else 0.01
    with open(log_fname, "a") as fp:
        fp.write("Transition Matrix index :\n %s\n Transition Matrix :\n %s\n" % (transition_matrix_index, transition_matrix) )
    for idx, emission_err in enumerate([0, 0.01, 0.05, 0.1, 0.2]):
        emission = [[1-emission_err, emission_err],[emission_err, 1-emission_err]]
        records, labels = generate_records(transition_matrix, emission, 1 << 14, chain_length, evidence_length)
        records = np.array(records)
        labels = np.array(labels)
        regular_accs = []
        testing_accs = []
        sig_testing_accs = []
        fold_size = int(len(records)/5);
        for i in range(5):
            test_records = records[i*fold_size:(i+1)*fold_size]
            test_labels = labels[i*fold_size:(i+1)*fold_size]
            train_records = np.concatenate((records[: i*fold_size], records[(i+1)*fold_size:]), axis=0)
            train_labels = np.concatenate((labels[: i*fold_size], labels[(i+1) * fold_size:]), axis=0)
            regular_model = RegularHMM(2, chain_length)
            regular_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1), tf.keras.callbacks.TensorBoard(log_dir="%s/%s/%s/regular"%(log_dirs, idx, i))]
            #regular_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
            regular_acc = regular_model.evaluate(test_records, test_labels)
            print ("Regular acc %s" % regular_acc)
            with open(log_fname, "a") as fp:
                fp.write("Regular ACC %s\n" % regular_acc[1])
            regular_accs.append(regular_acc[1])
            testing_model = TestingHMM(2, chain_length)
            testing_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1), tf.keras.callbacks.TensorBoard(log_dir="%s/%s/%s/testing"%(log_dirs, idx, i))]
            #testing_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
            testing_acc = testing_model.evaluate(test_records, test_labels)
            print ("test acc %s" % testing_acc)
            with open(log_fname, "a") as fp:
                fp.write("Testing ACC %s\n" % testing_acc[1])
            testing_accs.append(testing_acc[1])
            testing_sig_model = TestingSigmoidalHMM(2, chain_length)
            testing_sig_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1), tf.keras.callbacks.TensorBoard(log_dir="%s/%s/%s/testing_sig"%(log_dirs, idx, i))]
            testing_sig_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
            testing_sig_acc = testing_sig_model.evaluate(test_records, test_labels)
            print ("sig test acc %s" % testing_sig_acc)
            with open(log_fname, "a") as fp:
                fp.write("Testing sig ACC %s\n" % testing_sig_acc[1])
            sig_testing_accs.append(testing_sig_acc[1])

            with open(log_fname, "a") as fp:
                fp.write("Iteration %s: regular acc %s, testing acc %s , sig testing acc %s \n" % (i, regular_acc, testing_acc, testing_sig_acc))
        with open(log_fname, "a") as fp:
            fp.write("Average for emission error %s: regular acc %s, testing acc %s , testing sig acc %s\n" % (emission_err, np.mean(regular_accs), np.mean(testing_accs), np.mean(sig_testing_accs)))
        with open(log_fname, "a") as fp:
            fp.write("STD for emission error %s: regular acc %s, testing acc %s , testing sig acc %s\n" % (emission_err, np.std(regular_accs), np.std(testing_accs), np.std(sig_testing_accs)))



def icml_plot_single_configuration (transition_matrix_index, log_fname):
    chain_length = 8
    evidence_length = 7
    transition_state = np.unravel_index(transition_matrix_index, [2 for i in range(0, 8)])
    transition_matrix = np.zeros((2,2,2,2));
    log_dirs = "logs/%s/"%transition_matrix_index
    for k, cur_value in enumerate(transition_state):
        cur_index = np.unravel_index(k, (2, 2, 2))
        transition_matrix[cur_index][0] = 0.01 if cur_value == 0 else 0.99
        transition_matrix[cur_index][1] = 0.99 if cur_value == 0 else 0.01
    with open(log_fname, "a") as fp:
        fp.write("Transition Matrix index :\n %s\n Transition Matrix :\n %s\n" % (transition_matrix_index, transition_matrix) )
    for idx, emission_err in enumerate([0.05]):
        emission = [[1-emission_err, emission_err],[emission_err, 1-emission_err]]
        records, labels = generate_records(transition_matrix, emission, 1 << 14, chain_length, evidence_length)
        records = np.array(records)
        labels = np.array(labels)
        regular_accs = []
        testing_accs = []
        fold_size = int(len(records)/5);
        for i in range(1):
            test_records = records[i*fold_size:(i+1)*fold_size]
            test_labels = labels[i*fold_size:(i+1)*fold_size]
            train_records = np.concatenate((records[: i*fold_size], records[(i+1)*fold_size:]), axis=0)
            train_labels = np.concatenate((labels[: i*fold_size], labels[(i+1) * fold_size:]), axis=0)
            regular_model = RegularHMM(2, chain_length)
            regular_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1)]
            regular_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
            regular_acc = regular_model.evaluate(test_records, test_labels)
            print ("Regular acc %s" % regular_acc)
            with open(log_fname, "a") as fp:
                fp.write("Regular ACC %s\n" % regular_acc[1])
            regular_accs.append(regular_acc[1])
            testing_model = TestingHMM(2, chain_length)
            testing_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1), tf.keras.callbacks.TensorBoard(log_dir="%s/%s/%s/testing"%(log_dirs, idx, i))]
            testing_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
            testing_acc = testing_model.evaluate(test_records, test_labels)
            print ("test acc %s" % testing_acc)
            with open(log_fname, "a") as fp:
                fp.write("Testing ACC %s\n" % testing_acc[1])
            testing_accs.append(testing_acc[1])
            with open(log_fname, "a") as fp:
                fp.write("Iteration %s: regular acc %s, testing acc %s \n" % (i, regular_acc, testing_acc))
            print (testing_model.weights)
        with open(log_fname, "a") as fp:
            fp.write("Average for emission error %s: regular acc %s, testing acc %s\n" % (emission_err, np.mean(regular_accs), np.mean(testing_accs)))
        with open(log_fname, "a") as fp:
            fp.write("STD for emission error %s: regular acc %s, testing acc %s\n" % (emission_err, np.std(regular_accs), np.std(testing_accs)))

import logging
import pickle
def generate_records_for_bm():
    transition_matrix_index = 16
    chain_length = 8
    evidence_length = 7
    emission_err = 0.01
    emission = [[1-emission_err, emission_err],[emission_err, 1-emission_err]]
    transition_state = np.unravel_index(transition_matrix_index, [2 for i in range(0, 8)])
    transition_matrix = np.zeros((2,2,2,2));
    for k, cur_value in enumerate(transition_state):
        cur_index = np.unravel_index(k, (2, 2, 2))
        transition_matrix[cur_index][0] = 0.01 if cur_value == 0 else 0.99
        transition_matrix[cur_index][1] = 0.99 if cur_value == 0 else 0.01
    records, labels = generate_records(transition_matrix, emission, 1 << 14, chain_length, evidence_length)
    return np.array(records), np.array(labels)

def bm(model_lists):
    model_accs = {}
    for i in range(0, 5):
        records, labels = generate_records_for_bm()
        train_size = int(len(records) / 5 * 4)
        test_records = records[train_size:]
        test_labels = labels[train_size:]
        train_records = records[:train_size]
        train_labels = labels[:train_size]
        for j in range(0, 5):
            for model in model_lists:
                cur_model = model_lists[model]
                cur_model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.005,patience=50,verbose=1)]
                cur_model.fit(train_records, train_labels, epochs=100, batch_size=128, callbacks=callbacks, validation_split=0.2)
                acc = cur_model.evaluate(test_records, test_labels)
                model_accs.setdefault(model, []).append(acc[1])
            print (model_accs)
    with open("model_acc.p", "wb") as fp:
        pickle.dump(model_accs, fp)
    for model_name in model_accs:
        print ("Model name :%s mean %s std %s" % (model_name, np.mean(model_accs[model_name]), np.std(model_accs[model_name])))




if __name__ == "__main__":
    #tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
    import sys
    interation_index = int(sys.argv[1])
    log_fname = sys.argv[2]
    #icml_plot_single_configuration(interation_index, log_fname)
    #icml_exp_with_single_configuration(interation_index, log_fname)
    #generate_records_for_bm()
    chain_length = 8
    print ("here")
    bm({"testing_model":RegularHMM(2, chain_length), "testing_model":TestingHMM(2, chain_length), "testing_sig_non_clip_no_fixed_gamma": TestingSigmoidalHMM(2, chain_length, False, False), "testing_sig_non_clip_fixed_gamma": TestingSigmoidalHMM(2, chain_length, False, True), "testing_sig_clip_no_fixed_gamma": TestingSigmoidalHMM(2, chain_length, True, False), "testing_sig_clip_fixed_gamma": TestingSigmoidalHMM(2, chain_length, True, True)})
