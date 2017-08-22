import utils.data as du
import tensorflow as tf
from sequence_labeler import SequenceLabeler
import configparser

import numpy as np
import random
np.random.seed(1337)
random.seed = 1337


def test(test_path, data_save_path, conf_path, model_save_path, model_name, embedding_path, out_file_path):
    ow = open(out_file_path, "w")
    with tf.Graph().as_default():
        np.random.seed(1337)
        tf.set_random_seed(1337)

        config = configparser.ConfigParser()
        config.read(conf_path)

        processors_num = int(config.get("Training", "processors"))
        embedding_size = int(config.get("Network", "embedding_size"))
        cell_rnn_size = int(config.get("Network", "cell_rnn_size"))
        hidden_layer_size = int(config.get("Network", "hidden_layer_size"))

        vocab, num_classes, max_length, cl, cl_inv, text_field, category_field, feature_field = du.load_params(
            data_save_path)

        test_data = du.load_data_with_maps(test_path, vocab=vocab, max_length=max_length,
                                           text_field=text_field, category_field=category_field,
                                           feature_field=feature_field,
                                           cl_map=cl)

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False, inter_op_parallelism_threads=processors_num,
            intra_op_parallelism_threads=processors_num)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("Initializing Embedding")
            embedding = du.load_embeddings(embedding_path, vocab) if embedding_path != 'random' else du.initialize_random_embeddings(
                len(vocab), embedding_size)

            print("Building nn_model")
            sequence_labeler = SequenceLabeler(sequence_length=max_length, embedding=embedding, cell_size=cell_rnn_size, num_classes=len(cl), hls=hidden_layer_size, verbose=False)
            sequence_labeler.build_network()

            tf.global_variables_initializer().run()

            test_x, test_y = du.get_training_data(test_data, len(cl))

            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess=sess, save_path=model_save_path+"/"+model_name)

            loss, accuracy, predictions = sequence_labeler.predict(sess, test_x, test_y)

            for i in range(len(test_data)):
                l = test_data[i].original_length
                tokens = test_data[i].original_tokens[0:l]
                golds = test_data[i].labels[0:l]
                predic = predictions[i]
                pred_nums = predic[0:l]
                pred_labels = [cl_inv[x] for x in pred_nums]

                for a in zip(tokens, golds, pred_labels):
                    ow.write(" ".join(a)+"\n")
                ow.write("\n")
    ow.close()


if __name__ == "__main__":
    test_path = "ner_data/english/ner_conll_03_testb.json"
    data_save_path = "data_params.pkl"
    conf_path = "conf/test.properties"
    model_save_path = "ner_model"
    model_name = "model"
    summaries_dir = "ner_summaries"
    embedding_path = "random"
    out = "testb.tsv"

    test(test_path, data_save_path, conf_path, model_save_path, model_name, embedding_path, out)