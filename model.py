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
        emission  = tf.stack([tf.nn.softmax(self.emission_weight_raw[i,:]) for i in range(0, self.cardinality)], axis=0)
        return get_normalized_pr(h * tf.matmul(aug_x, tf.transpose(emission)))

class TransitionLayer(tf.keras.layers.Layer):
    def __init__ (self):
        super(TransitionLayer, self).__init__()

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
        self.transition_layer = TransitionLayer()
        self.evidence_layer = EvidenceLayer()

    def call(self, inputs):
        print (inputs[:,0].shape)
        post = self.evidence_layer([self.prior_layer(None), inputs[:,0]])
        for i in range(1, self.chain_length):
            post = self.transition_layer(post)
            post = self.evidence_layer([post, inputs[:,i]])
        return post

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

def learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, prior = None, transition = None, emission = None, posts = None, iterations = 100000):
    train_size = train_records.shape[0]
    feature_size = train_records.shape[1]
    label_size = train_labels.shape[1]
    training_dataset = tf.data.Dataset.from_tensor_slices((train_records, train_labels)).repeat().batch(2048)
    training_iterator = training_dataset.make_initializable_iterator()
    next_training_element = training_iterator.get_next()
    total_training_feed_dict = {x: train_records, y: train_labels}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(training_iterator.initializer)
        try:
            for i in range(iterations):
                training_features, training_labels = sess.run(next_training_element)
                sess.run(optimizer, feed_dict={x:training_features, y: training_labels})
                if i % 100 == 0:
                    err = loss.eval(feed_dict = total_training_feed_dict)
                    print ("== iteration %d ======================" % (i))
                    print ("err: %.8g" % err)

                """
            feed_dict = {x: train_records, y: train_labels}
            print( "== initialization ==================")
            print( "err: %.8g" % loss.eval(feed_dict) )
            for iteration in range(iterations):
                perm = np.random.permutation(train_size)
                batch_size = 64
                num_batch = int(train_size/ batch_size)
                for j in range(0, num_batch):
                    cur_x_train = train_records[perm[j*batch_size: (j+1)*batch_size]]
                    cur_y_train = train_labels[perm[j*batch_size: (j+1)*batch_size]]
                    optimizer.run(feed_dict={x: cur_x_train, y: cur_y_train})
                if iteration % 100 == 0:
                    err = loss.eval(feed_dict)
                    print( "== iteration %d ==================" % (iteration+1) )
                    print( "err: %.8g" % err )
                    if err <= 0.00001:
                        break;

                """
        except KeyboardInterrupt:
            pass
        err = loss.eval(total_training_feed_dict)
        print( "final-err: %.8g" %  err)
        predictions = prediction.eval(feed_dict={x:test_records})
        acc = np.mean(1.0 * (np.argmax(test_labels, axis=1) == predictions))
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

def true_model_prediction(cur_transition_matrix, emission, x):
    pass

def run_with_2_trial(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000):
    result = [learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000) for i in range(0, 2)]
    print (result)
    err_result = [a[1] for a in result]
    min_index = err_result.index(min(err_result))
    return result[min_index][0]


def chain_3_emission_experiments():
    cur_transition_matrix = [[[[0, 1], [0,1]], [[0,1], [0,1]]],[[[0,1],[1,0]], [[1,0], [1,0]]]]
    chain_length = 8
    evidence_length = 7
    for i in range(0, 7):
        emission_err = 1-1.0/(1.0 + math.exp(-i))
        emission = [[1-emission_err, emission_err],[emission_err, 1-emission_err]]
        records, labels = generate_records(cur_transition_matrix, emission, 1 << 14, chain_length, evidence_length)
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
            regular_acc = run_with_2_trial(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=5000)
            print ("Regular ACC %s " % regular_acc)
            regular_accs.append(regular_acc)
            x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, testing_emission=False, learning_rate = 0.01)
            testing_acc = run_with_2_trial(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=5000)
            print ("Testing ACC %s" % testing_acc)
            testing_accs.append(testing_acc)
            """
            x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, testing_emission=True, learning_rate = 0.01)
            emission_testing_acc = run_with_5_trial(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000)
            print ("Emission testing ACC %s" % emission_testing_acc)
            emission_testing_accs.append(emission_testing_acc)
            """
            print ("Iteration %s: regular acc %s, testing acc %s" % (i, regular_acc, testing_acc))
        print ("Average for emission error %s: regular acc %s, testing acc %s" % (emission_err, np.mean(regular_accs), np.mean(testing_accs)))


def chain_3_particular_experiments():
    cur_transition_matrix = [[[[0, 1], [0,1]], [[0,1], [0,1]]],[[[0,1],[1,0]], [[1,0], [1,0]]]]
    emission = [[1, 0], [0, 1]]
    chain_length = 8
    evidence_length = 7
    records, labels = generate_records(cur_transition_matrix, emission, 1 << 14, chain_length, evidence_length)
    records = np.array(records)
    labels = np.array(labels)
    regular_accs = []
    testing_accs = []
    emission_testing_accs = []
    for i in range(5):
        training_size = int(len(records) * 4 / 5);
        perm = np.random.permutation(len(records))
        train_records = records[perm[:training_size]]
        train_labels = labels[perm[:training_size]]
        test_records = records[perm[training_size:]]
        test_labels = labels[perm[training_size:]]
        x, y, loss, prediction, optimizer, prior, transition, learned_emission, posts = regular_hmm (chain_length, 2, learning_rate = 0.05)
        regular_acc = max([learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000)[0] for i in range(0, 5)])
        print ("Regular ACC %s " % regular_acc)
        regular_accs.append(regular_acc)
        x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, testing_emission=False, learning_rate = 0.01)
        testing_acc = max([learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000)[0] for i in range(0, 5)])
        print ("Testing ACC %s" % testing_acc)
        testing_accs.append(testing_acc)
        x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, testing_emission=True, learning_rate = 0.01)
        emission_testing_acc = max([learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer, iterations=500000) for i in range(0, 5)])
        print ("Emission testing ACC %s" % emission_testing_acc)
        emission_testing_accs.append(emission_testing_acc)
        print ("Iteration %s: regular acc %s, testing acc %s, emission testing acc %s" % (i, regular_acc, testing_acc, emission_testing_acc))
    print ("Cur Transition Matrix %s " % cur_transition_matrix)
    print ("Best: regular acc %s, testing acc %s, emission testing acc" % (np.amax(regular_accs), np.amax(testing_accs), np.amax(emission_testing_accs)))
    print ("Average: regular acc %s, testing acc %s, emission testing acc" % (np.mean(regular_accs), np.mean(testing_accs), np.mean(emission_testing_accs)))

def chain_3_experiments ():
    emission = [[1, 0], [0, 1]]
    chain_length = 8
    evidence_length = 7
    for entry in np.ndindex(*[2 for i in range(0, 8)]):
        cur_transition_matrix = np.zeros((2,2,2,2));
        for k, cur_value in enumerate(entry):
            cur_index = np.unravel_index(k, (2, 2, 2))
            cur_transition_matrix[cur_index][0] = cur_value
            cur_transition_matrix[cur_index][1] = 1 - cur_value
        records, labels = generate_records(cur_transition_matrix, emission, 1 << 12, chain_length, evidence_length)
        records = np.array(records)
        labels = np.array(labels)
        regular_accs = []
        testing_accs = []
        for i in range(10):
            training_size = int(len(records) * 4 / 5);
            perm = np.random.permutation(len(records))
            train_records = records[perm[:training_size]]
            train_labels = labels[perm[:training_size]]
            test_records = records[perm[training_size:]]
            test_labels = labels[perm[training_size:]]
            x, y, loss, prediction, optimizer, prior, transition, learned_emission, posts = regular_hmm (chain_length, 2, learning_rate = 0.05)
            regular_acc = learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
            regular_accs.append(regular_acc)
            x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, learning_rate = 0.05)
            testing_acc = learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
            testing_accs.append(testing_acc)
            print ("Iteration %s: regular acc %s, testing acc %s" % (i, regular_acc, testing_acc))
        print ("Cur Transition Matrix %s " % cur_transition_matrix)
        print ("Best: regular acc %s, testing acc %s" % (np.maximum(regular_accs), np.maximum(testing_accs)))
        print ("Average: regular acc %s, testing acc %s" % (np.mean(regular_accs), np.mean(testing_accs)))
chain_3_emission_experiments()
#chain_3_particular_experiments()


"""

transition_matrix = [[1, 1], [1,0]]
chain_length = 11
records = []
labels = []
for i in range(0, 1<<10):
    cur_record = [random.random() > 0.5, random.random() > 0.5]
    for j in range (2, chain_length):
        cur_record.append(transition_matrix[cur_record[-2]][cur_record[-1]])
    labels.append([0,1] if cur_record[-1] == 1 else [1,0])
    cur_record[-1] = 0.5
    records.append(cur_record)
records = np.array(records)
labels = np.array(labels)
regular_accs = []
testing_accs = []
for i in range(1):
    training_size = int(len(records) * 4 / 5);
    perm = np.random.permutation(len(records))
    train_records = records[perm[:training_size]]
    train_labels = labels[perm[:training_size]]
    test_records = records[perm[training_size:]]
    test_labels = labels[perm[training_size:]]
    x, y, loss, prediction, optimizer, prior, transition, emission, posts = regular_hmm (chain_length, 2, learning_rate = 0.05)
    regular_acc = learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
    regular_accs.append(regular_acc)
    x, y, loss, prediction, optimizer = testing_hmm (chain_length, 2, learning_rate = 0.05)
    testing_acc = learn(train_records, train_labels, test_records, test_labels, x, y, loss, prediction, optimizer)
    testing_accs.append(testing_acc)
    print ("Iteration %s: regular acc %s, testing acc %s" % (i, regular_acc, testing_acc))
print ("Average: regular acc %s, testing acc %s" % (np.mean(regular_accs), np.mean(testing_accs)))

if __name__ == "__main__":
    cur_transition_matrix = [[[[0, 1], [0,1]], [[0,1], [0,1]]],[[[0,1],[1,0]], [[1,0], [1,0]]]]
    emission = [[1, 0], [0, 1]]
    chain_length = 8
    evidence_length = 7
    records, labels = generate_records(cur_transition_matrix, emission, 1 << 8, chain_length, evidence_length)
    records = np.array(records)
    labels = np.array(labels)
    model = TestingHMM(2, chain_length)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001,patience=6,verbose=1), tf.keras.callbacks.TensorBoard()]
    callbacks = [tf.keras.callbacks.TensorBoard()]
    model.fit(records, labels, epochs=1000, batch_size=16, callbacks=callbacks, validation_split=0.2)
    print(model.evaluate(records, labels))
"""
