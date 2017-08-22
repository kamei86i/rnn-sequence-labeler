import tensorflow as tf
from tensorflow.contrib import rnn
import utils.data as du

import numpy as np
import random
np.random.seed(1337)
random.seed = 1337


class SequenceLabeler:
    def __init__(self, sequence_length, embedding, cell_size, num_classes, hls, gcap=5, lam=0.1, verbose=False):
        self.max_sequence_length = sequence_length
        self.embedding = embedding
        self.cell_size = cell_size
        self.num_classes = num_classes
        self.hls = hls
        self.verbose=verbose
        self.gcap = gcap
        self.lam = lam

    def build_network(self):
        with tf.variable_scope('input'):
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_sequence_length], name="input_x")
            if self.verbose:
                print(self.input_x)

            self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.max_sequence_length, self.num_classes], name="input_y")
            if self.verbose:
                print(self.input_y)

            self.dkp = tf.placeholder(dtype=tf.float32, name="dropout")
            if self.verbose:
                print(self.dkp)

            self.mask = tf.not_equal(self.input_x, tf.constant(0, dtype=tf.int32), name='mask')
            if self.verbose:
                print(self.mask)

            self.correct_labels = tf.argmax(self.input_y, axis=2, name='argmax_labels')
            if self.verbose:
                print(self.correct_labels)

            self.sequence_lengths = tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1, name='sequence_lengths')
            if self.verbose:
                print(self.sequence_lengths)

        with tf.variable_scope('embedding_layer'):
            emb_w = tf.Variable(trainable=self.embedding['trainable'], initial_value=self.embedding['weights'], dtype=tf.float32, name="embedding")
            token_embedded = tf.nn.embedding_lookup(emb_w, self.input_x, name="embedding_token")
            token_embedded = tf.nn.dropout(token_embedded, self.dkp, name="dropout_embedding")
            self.emb_loss = tf.nn.l2_loss(emb_w, name="embedding_loss")
            if self.verbose:
                print(token_embedded)

        with tf.variable_scope('bi_rnn'):
            with tf.variable_scope('forward'):
                fw_cell = rnn.GRUCell(num_units=self.cell_size)

            with tf.variable_scope('backward'):
                bw_cell = rnn.GRUCell(num_units=self.cell_size)

            with tf.variable_scope('rnn'):
                self.output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, token_embedded,
                                                                self.sequence_lengths, dtype=tf.float32)
                if self.verbose:
                    print(self.output)

                self.output = tf.concat(self.output, axis=2, name="concat")
                self.output = tf.nn.dropout(self.output, self.dkp, name="rnn_dropout")
                if self.verbose:
                    print(self.output)

        with tf.variable_scope('sequence_classifier'):
            in_size = self.cell_size * 2
            self.classify = tf.unstack(self.output, axis=1, name="unstack")
            if self.verbose:
                print(self.classify)

            self.sequence_cl_loss = 0.0

            if self.hls > 0:
                wh = tf.get_variable(name="W_hidden", dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[in_size, self.hls])
                bh = tf.get_variable(name="b_hidden", dtype=tf.float32, initializer=tf.zeros_initializer(),
                                 shape=[self.hls])

                self.sequence_cl_loss += tf.nn.l2_loss(wh) + tf.nn.l2_loss(bh)
                self.classify = [tf.nn.relu(tf.nn.xw_plus_b(x, wh, bh), name="hidden") for x in self.classify]
                in_size = self.hls

            wc = tf.get_variable(name="W_classify", dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                shape=[in_size, self.num_classes])
            bc = tf.get_variable(name="b_classify", dtype=tf.float32, initializer=tf.zeros_initializer(),
                                shape=[self.num_classes])

            self.sequence_cl_loss += tf.nn.l2_loss(wc) + tf.nn.l2_loss(bc)

            self.scores = [tf.nn.xw_plus_b(x, wc, bc, name="scores") for x in self.classify]
            if self.verbose:
                print(self.scores)
            self.scores = tf.stack(self.scores, axis=1, name="stack")
            if self.verbose:
                print(self.scores)

        with tf.variable_scope('loss'):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.correct_labels, logits=self.scores)
            if self.verbose:
                print(self.losses)
            self.losses = tf.multiply(self.losses, tf.cast(self.mask, dtype=tf.float32), name='masked_losses')
            if self.verbose:
                print(self.losses)
            self.loss1 = tf.divide(tf.reduce_sum(self.losses, axis=1), tf.cast(self.sequence_lengths, dtype=tf.float32), name="masked_mean_losses")
            self.loss = tf.reduce_mean(self.loss1, name="mean_loss") + self.lam * self.emb_loss + self.lam * self.sequence_cl_loss
            if self.verbose:
                print(self.loss)
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('measures'):
            self.predictions = tf.argmax(self.scores, axis=2, name='argmax_predictions')
            if self.verbose:
                print(self.predictions)

            self.correct_predictions = tf.cast(tf.equal(self.predictions, self.correct_labels), dtype=tf.float32, name='correct_predictions')
            if self.verbose:
                print(self.correct_predictions)

            self.correct_predictions = tf.boolean_mask(self.correct_predictions, self.mask, name="masked_correct_predictions")
            if self.verbose:
                print(self.correct_predictions)

            self.accuracy = tf.reduce_mean(self.correct_predictions, name="accuracy")
            if self.verbose:
                print(self.accuracy)
            tf.summary.scalar('accuracy', self.accuracy)

    def build_train_ops(self, lr):
        with tf.name_scope('train_op'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(lr)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, self.gcap), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step,
                                                           name="train_op")

    def summary(self):
        self.merged = tf.summary.merge_all()

    def train(self, session, batch_x, batch_y, dropout):
        feed_dict = {
            self.input_x: batch_x,
            self.input_y: batch_y,
            self.dkp: dropout
        }

        _, step, loss, summary, accuracy, mask, correct_labels, lengths, losses, loss1, scores = session.run(
            [self.train_op, self.global_step, self.loss, self.merged, self.accuracy, self.mask, self.correct_labels, self.sequence_lengths, self.losses, self.loss1, self.scores], feed_dict)
        return step, loss, summary, accuracy, mask, correct_labels, lengths, losses, loss1, scores

    def predict(self, session, x, y):
        feed_dict = {
            self.input_x: x,
            self.input_y: y,
            self.dkp: 1.0
        }
        loss, accuracy, predictions = session.run(
            [self.loss, self.accuracy, self.predictions], feed_dict)
        return loss, accuracy, predictions


if __name__ == '__main__':
    emb = du.initialize_random_embeddings(100, 20)
    sl = SequenceLabeler(20, emb, 16, 5, 12, True)
    sl.build_network()