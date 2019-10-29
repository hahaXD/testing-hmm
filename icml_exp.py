import random
import tensorflow as tf
import numpy as np
import sys
import math

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8


def get_normalized_pr (input_layer):
    sum_value = 1 / tf.reduce_sum(input_layer, axis=1)
    return tf.einsum('i, ij->ij',sum_value, input_layer)

def get_normalized_pr_exp (input_layer):
    sum_value = 1 / tf.reduce_sum(tf.exp(input_layer), axis=1)
    return tf.einsum('i, ij->ij',sum_value, tf.exp(input_layer))

def get_normalized_pr_with_batch (input_layer):
    sum_value = 1 / tf.reduce_sum(tf.exp(input_layer), axis=2)
    return tf.einsum("ij, ijk -> ijk", sum_value, tf.exp(input_layer))

def testing_hmm (chain_length, cardinality, testing_emission=False, learning_rate = 0.005):
    x = tf.placeholder(tf.float32, shape=[None, chain_length])
    aug_x = tf.stack([1-x, x], axis = 2)
    y = tf.placeholder(tf.float32, shape=[None, cardinality])
    prior = tf.nn.softmax(tf.Variable(tf.random.uniform([cardinality])))
    transition_param_coeff = tf.Variable(tf.random.uniform([cardinality, cardinality]))
    transition_param_bias = tf.Variable(tf.random.uniform([cardinality, cardinality]))
    emission = None
    if not testing_emission:
        print ("Emission does not contain testing unit")
        emission_raw = tf.Variable(tf.random.uniform([cardinality, 2]))
        emission  = tf.stack([tf.nn.softmax(emission_raw[i,:]) for i in range(0, cardinality)], axis=0)
        post = get_normalized_pr(prior * (tf.matmul(aug_x[:,0,:], tf.transpose(emission))))
    else:
        print ("Emission contains testing unit")
        emission_param_coeff = tf.Variable(tf.random.uniform([cardinality, 2]))
        emission_param_bias = tf.Variable(tf.random.uniform([cardinality, 2]))
        cur_emission_raw = tf.einsum('j, jk -> jk', prior, emission_param_coeff)
        cur_emission_raw += emission_param_bias
        cur_emission = get_normalized_pr_exp(cur_emission_raw)
        post = get_normalized_pr(prior * (tf.matmul(aug_x[:,0,:], tf.transpose(cur_emission))))
    posts = [post]
    for i in range(1, chain_length):
        transition_raw = tf.einsum('ij, jk ->ijk', post, transition_param_coeff)
        transition_raw += tf.einsum("ij, jk ->ijk", tf.fill(tf.shape(post), 1.0), transition_param_bias)
        transition = get_normalized_pr_with_batch(transition_raw)
        post_before_evid = tf.einsum("ij, ijk ->ik", post, transition)
        if not testing_emission:
            post = get_normalized_pr(post_before_evid * (tf.matmul(aug_x[:,i,:], tf.transpose(emission))))
        else:
            cur_emission_raw = tf.einsum('ij, jk -> ijk', post_before_evid, emission_param_coeff)
            cur_emission_raw += tf.einsum("ij, jk ->ijk", tf.fill(tf.shape(post_before_evid), 1.0), emission_param_bias)
            cur_emission = get_normalized_pr_with_batch(cur_emission_raw)
            post = get_normalized_pr(post_before_evid * tf.einsum("ij, ikj ->ik", aug_x[:,i,:], cur_emission))
        posts.append(post)
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.maximum(post, 0.000001)), reduction_indices=[1]))
    prediction = tf.math.argmax(post, axis=1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return x, y, loss, prediction, optimizer

def regular_hmm (chain_length, cardinality, learning_rate = 0.05):
    x = tf.placeholder(tf.float32, shape=[None, chain_length])
    aug_x = tf.stack([1-x, x], axis = 2)
    y = tf.placeholder(tf.float32, shape=[None, cardinality])
    prior = tf.nn.softmax(tf.Variable(tf.random.uniform([cardinality])))
    transition_raw = tf.Variable(tf.random.uniform([cardinality, cardinality]))
    transition = tf.stack([tf.nn.softmax(transition_raw[i,:]) for i in range(0, cardinality)], axis=0)
    emission_raw = tf.Variable(tf.random.uniform([cardinality, 2]))
    emission  = tf.stack([tf.nn.softmax(emission_raw[i,:]) for i in range(0, cardinality)], axis=0)
    post = get_normalized_pr(prior * (tf.matmul(aug_x[:,0,:], tf.transpose(emission))))
    posts = [post]
    for i in range(1, chain_length):
        post_before_evid = tf.matmul(post, transition)
        post = get_normalized_pr(post_before_evid * (tf.matmul(aug_x[:,i,:], tf.transpose(emission))))
        posts.append(post)
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.maximum(post, 0.000001)), reduction_indices=[1]))
    prediction = tf.math.argmax(post, axis=1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return x, y, loss, prediction, optimizer, prior, transition, emission, posts

def learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations = 100000, batch_size = 2048):
    train_size = train_records.shape[0]
    feature_size = train_records.shape[1]
    label_size = train_labels.shape[1]
    training_dataset = tf.data.Dataset.from_tensor_slices((train_records, train_labels)).repeat().batch(batch_size)
    training_iterator = training_dataset.make_initializable_iterator()
    next_training_element = training_iterator.get_next()
    total_training_feed_dict = {x: train_records, y: train_labels}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(training_iterator.initializer)
        for i in range(iterations):
            training_features, training_labels = sess.run(next_training_element)
            sess.run(optimizer, feed_dict={x:training_features, y: training_labels})
            if i % 100 == 0:
                err = loss.eval(feed_dict = total_training_feed_dict)
                print ("== iteration %d ======================" % (i))
                print ("err: %.8g" % err)
        err = loss.eval(total_training_feed_dict)
        print( "final-err: %.8g" %  err)
        predictions = prediction.eval(feed_dict={x:test_records})
        acc = np.mean(1.0 * (np.argmax(test_labels, axis=1) == predictions))
    return acc, err

def learn_with_start_up_bound(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, err_bound, iterations = 5000, batch_size = 2048):
    train_size = train_records.shape[0]
    feature_size = train_records.shape[1]
    label_size = train_labels.shape[1]
    training_dataset = tf.data.Dataset.from_tensor_slices((train_records, train_labels)).repeat().batch(batch_size)
    training_iterator = training_dataset.make_initializable_iterator()
    next_training_element = training_iterator.get_next()
    total_training_feed_dict = {x: train_records, y: train_labels}
    smart_training_iterations = 10
    for k in range(0, smart_training_iterations):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(training_iterator.initializer)
            for i in range(500):
                training_features, training_labels = sess.run(next_training_element)
                sess.run(optimizer, feed_dict={x:training_features, y: training_labels})
            err = loss.eval(feed_dict = total_training_feed_dict)
            print ("Smart learning iteration %s: with initial error %s, and error bound : %s" % (k, err, err_bound))
            if (err > err_bound and k != smart_training_iterations-1):
                continue
            for i in range(500, iterations):
                training_features, training_labels = sess.run(next_training_element)
                sess.run(optimizer, feed_dict={x:training_features, y: training_labels})
                if i % 1000 == 0:
                    err = loss.eval(feed_dict = total_training_feed_dict)
                    print ("== iteration %d ======================" % (i))
                    print ("err: %.8g" % err)
            err = loss.eval(total_training_feed_dict)
            print( "final-err: %.8g" %  err)
            predictions = prediction.eval(feed_dict={x:test_records})
            acc = np.mean(1.0 * (np.argmax(test_labels, axis=1) == predictions))
            return acc, err

def learn_smart(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, batch_size = 2048):
    errs = []
    for i in range (0, 5):
        with tf.Session() as sess:
            acc, err = learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations = 500, batch_size = batch_size)
            errs.append(err)
    acc, err = learn_with_start_up_bound(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, np.mean(errs), batch_size = batch_size)
    return acc, err

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
    for k, cur_value in enumerate(transition_state):
        cur_index = np.unravel_index(k, (2, 2, 2))
        transition_matrix[cur_index][0] = cur_value
        transition_matrix[cur_index][1] = 1 - cur_value
    with open(log_fname, "a") as fp:
        fp.write("Transition Matrix index :\n %s\n Transition Matrix :\n %s\n" % (transition_matrix_index, transition_matrix) )
    for i in range(1, 6):
        if i == 5:
            emission_err = 0
        else:
            emission_err = 1.0/(1.0 + math.exp(i))
        emission = [[1-emission_err, emission_err],[emission_err, 1-emission_err]]
        records, labels = generate_records(transition_matrix, emission, 1 << 14, chain_length, evidence_length)
        records = np.array(records)
        labels = np.array(labels)
        regular_accs = []
        testing_accs = []
        for i in range(5):
            training_size = int(len(records) * 4 / 5);
            perm = np.random.permutation(len(records))
            train_records = records[perm[:training_size]]
            train_labels = labels[perm[:training_size]]
            test_records = records[perm[training_size:]]
            test_labels = labels[perm[training_size:]]
            x, y, loss, prediction, optimizer, prior, transition, learned_emission, posts = regular_hmm (chain_length, 2, learning_rate = 0.01)
            regular_acc = learn_smart(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
            with open(log_fname, "a") as fp:
                fp.write("Regular ACC %s\n" % regular_acc[0])
            regular_accs.append(regular_acc[0])
            x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, testing_emission=False, learning_rate = 0.01)
            testing_acc = learn_smart(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
            with open(log_fname, "a") as fp:
                fp.write("Testing ACC %s\n" % testing_acc[0])
            testing_accs.append(testing_acc[0])
            with open(log_fname, "a") as fp:
                fp.write("Iteration %s: regular acc %s, testing acc %s \n" % (i, regular_acc, testing_acc))
        with open(log_fname, "a") as fp:
            fp.write("Average for emission error %s: regular acc %s, testing acc %s\n" % (emission_err, np.mean(regular_accs), np.mean(testing_accs)))
        with open(log_fname, "a") as fp:
            fp.write("STD for emission error %s: regular acc %s, testing acc %s\n" % (emission_err, np.std(regular_accs), np.std(testing_accs)))


if __name__ == "__main__":
    import sys
    interation_index = int(sys.argv[1])
    log_fname = sys.argv[2]
    icml_exp_with_single_configuration(interation_index, log_fname)
