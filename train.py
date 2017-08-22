import utils.data as du
import tensorflow as tf
from sequence_labeler import SequenceLabeler
import configparser
import os

import numpy as np
import random
np.random.seed(1337)
random.seed = 1337
np.set_printoptions(threshold=np.nan)


def train_and_test(train_path, data_save_path, embedding_path, conf_path, model_save_path, model_name, summaries_dir):
    with tf.Graph().as_default():
        np.random.seed(1337)
        tf.set_random_seed(1337)

        config = configparser.ConfigParser()
        config.read(conf_path)

        sentence_f = config.get("Data", "sentence_field")
        token_f = config.get("Data", "token_field")
        label_f = config.get("Data", "categories_field")

        processors_num = int(config.get("Training", "processors"))
        batch_size = int(config.get("Training", "batch_size"))
        epochs = int(config.get("Training", "epochs"))
        patience = int(config.get("Training", "patience"))
        validation_split = float(config.get("Training", "val_split"))
        learning_rate = float(config.get("Training", "learning_rate"))

        embedding_size = int(config.get("Network", "embedding_size"))
        cell_rnn_size = int(config.get("Network", "cell_rnn_size"))
        dropout = float(config.get("Network", "dropout"))
        hidden_layer_size = int(config.get("Network", "hidden_layer_size"))

        data, vocab, max_length, nc, cl, cl_inv = du.load_data(train_path,
                                                                text_field=sentence_f,
                                                                category_field=label_f,
                                                                feature_field=token_f)
        params = [vocab, nc, max_length, cl, cl_inv, sentence_f, label_f,
                  token_f]
        du.save_params(params, data_save_path)
        train_data, valid_data = du.split(data, validation_split)

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False, inter_op_parallelism_threads=processors_num,
            intra_op_parallelism_threads=processors_num)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("Initializing Embedding")
            embedding = du.load_embeddings(embedding_path, vocab) if embedding_path != 'random' else du.initialize_random_embeddings(len(vocab), embedding_size)

            print("Building nn_model")
            sequence_labeler = SequenceLabeler(sequence_length=max_length, embedding=embedding, cell_size=cell_rnn_size, num_classes=len(cl), hls=hidden_layer_size)
            sequence_labeler.build_network()

            print("Building training operations")
            sequence_labeler.build_train_ops(learning_rate)
            sequence_labeler.summary()

            tf.global_variables_initializer().run()

            valid_x, valid_y = du.get_training_data(valid_data, len(cl))

            saver = tf.train.Saver(max_to_keep=1)

            if os.path.exists(model_save_path):
                saver.restore(sess=sess, save_path=model_save_path + "/" + model_name)
            else:
                os.mkdir(model_save_path)

            best_vd_accuracy = 0.0
            best_vd_loss = 1000.0
            num_without_improvement = 0

            writer = tf.summary.FileWriter(summaries_dir + "/train",
                                           sess.graph)

            print("Start training")
            for epoch in range(epochs):
                if num_without_improvement > patience:
                    break

                np.random.shuffle(train_data)
                batches = du.get_training_batches(train_data, batch_size)
                # Training on batches
                for batch in batches:
                    train_x, train_y = du.get_training_data(batch, len(cl))

                    # print(batch[0].tokens)
                    # print(batch[0].labels)
                    # print(train_x[0])
                    # print(train_y[0])

                    step, loss, summary, accuracy, mask, correct_labels, seq_lengths, losses, loss1,scores = sequence_labeler.train(sess, train_x, train_y, dropout)
                    writer.add_summary(summary, step)
                    print("Training: epoch\t{:g}\tstep\t{:g}\tloss\t{:g}\taccuracy\t{:g}".format(epoch, step, loss, accuracy))

                    # print(mask[0])
                    # print(correct_labels[0])
                    # print(seq_lengths[0])
                    # print(losses[0])
                    # print(scores[0])
                    # print(loss1[0])
                    # exit(0)

                # Evaluate on validation and test set
                vd_loss, vd_accuracy, vd_pre = sequence_labeler.predict(sess, valid_x, valid_y)
                print("Validation: loss\t{:g}\taccuracy\t{:g}".format(vd_loss, vd_accuracy))

                if vd_accuracy > best_vd_accuracy:
                    best_vd_accuracy = vd_accuracy
                    best_vd_loss = vd_loss
                    print("Saving nn_model")
                    saver.save(sess, model_save_path + "/" + model_name)
                    num_without_improvement = 0
                else:
                    num_without_improvement += 1

            print("Best Validation: loss\t{:g}\taccuracy\t{:g}".format(best_vd_loss, best_vd_accuracy))

if __name__ == "__main__":
    train_path = "ner_data/english/ner_conll_03_train.json"
    data_save_path = "data_params.pkl"
    conf_path = "conf/test.properties"
    model_save_path="ner_model"
    model_name="model"
    summaries_dir="ner_summaries"
    embedding_path="random"

    train_and_test(train_path, data_save_path, embedding_path, conf_path, model_save_path, model_name, summaries_dir)