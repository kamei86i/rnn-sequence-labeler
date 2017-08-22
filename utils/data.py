import json
import numpy as np

np.random.seed(1337)

import _pickle as cPickle
from gensim.models import Word2Vec
from keras.utils.np_utils import to_categorical

PAD = 'PAD'
UNK = 'UNK'
PAD_LABEL = "_PL_"


class NNSequenceData:
    def __init__(self, id, original_sentence, tokens, pos, labels):
        self.id = id
        self.original_sentence = original_sentence
        self.original_length = len(tokens)
        self.prediction = []
        self.original_tokens = [x for x in tokens if x.strip() != ""]
        self.tokens = [x.lower() for x in tokens if x.strip() != ""]
        self.pos = [x.lower() for x in pos]
        self.labels = [y for y in labels]
        self.labels_ids = []
        self.tokens_ids = []
        self.pos_ids = []

    def pad(self, max_l, pad_token, pad_label):
        diff = max_l - len(self.tokens)
        if diff > 0:
            self.tokens = self.tokens + ([pad_token] * diff)
            self.original_tokens = self.original_tokens + ([pad_token] * diff)
            self.labels = self.labels + ([pad_label] * diff)
            self.pos = self.pos + ([pad_token] * diff)

        self.tokens = self.tokens[0:max_l]
        self.original_tokens = self.original_tokens[0:max_l]
        self.labels = self.labels[0:max_l]
        self.pos = self.pos[0:max_l]

    def apply_maps(self, vocab, unk_token, cl):
        for token in self.tokens:
            self.tokens_ids.append(vocab[token] if token in vocab else vocab[unk_token])

        for label in self.labels:
            label_num = cl[label]
            self.labels_ids.append(label_num)


def initialize_vocab(nn_d):
    vocab = dict()
    VOCAB_COUNTER = 0
    vocab[PAD] = VOCAB_COUNTER
    VOCAB_COUNTER += 1
    vocab[UNK] = VOCAB_COUNTER
    VOCAB_COUNTER += 1

    for tr in nn_d:
        for token in tr.tokens:
            if token not in vocab:
                vocab[token] = VOCAB_COUNTER
                VOCAB_COUNTER += 1
    return vocab


def load_json(path, text_field, categories_field, feature_field):
    f = open(path, 'r', encoding='utf8')
    ret = []
    for line in f:
        line = line.strip()
        js = json.loads(line)
        nn = NNSequenceData(id=js['id'], original_sentence=js[text_field],
                            tokens=js[feature_field], labels=js[categories_field],
                            pos=js['pos'])
        ret.append(nn)
    return ret


def get_labels_maps(data):
    cl = dict()
    cl_inv = dict()
    c = 0
    cl[PAD_LABEL] = c
    cl_inv[c] = PAD_LABEL
    c += 1
    for d in data:
        for label in d.labels:
            if label not in cl:
                cl[label] = c
                cl_inv[c] = label
                c += 1
    return cl, cl_inv


def load_data(train_path, text_field, category_field, feature_field="lemmas", max_length=None):
    raw_data = load_json(train_path, text_field, category_field, feature_field)
    if max_length is None:
        max_length = max([len(x.tokens) for x in raw_data])

    for tr in raw_data:
        tr.pad(max_length, PAD, PAD_LABEL)

    vocab = initialize_vocab(raw_data)
    cl, cl_inv = get_labels_maps(raw_data)

    for tr in raw_data:
        tr.apply_maps(vocab, UNK, cl)

    return raw_data, vocab, max_length, len(cl), cl, cl_inv


def load_data_with_maps(path, vocab, max_length, text_field, category_field, cl_map, feature_field="lemmas"):
    data = load_json(path, text_field, category_field, feature_field)

    for d in data:
        d.pad(max_length, PAD, PAD_LABEL)
        d.apply_maps(vocab, UNK, cl_map)
    return data


def split(dataset, split=0.8):
    np.random.shuffle(dataset)
    size = int(split*len(dataset))
    return dataset[0:size], dataset[size:]


def get_training_data(dataset, nb_classes):
    ret_x_w = []
    ret_y = []
    for d in dataset:
        ret_x_w.append(d.tokens_ids)
        ret_y.append(to_categorical(d.labels_ids, num_classes=nb_classes))
    return np.asarray(ret_x_w), np.asarray(ret_y)


def save_params(params, path):
    cPickle.dump(params, open(path, 'wb'))


def save_model(model, file_name):
    model.summary()
    model.save(file_name)


def load_params(path):
    return cPickle.load(open(path, 'rb'))


def initialize_random_embeddings(vocab_size, embedding_size):
    embedding = dict()
    embedding['size'] = vocab_size
    embedding['dims'] = embedding_size
    embedding['trainable'] = True

    weights = np.random.uniform(-1.0, 1.0, (vocab_size, embedding_size))
    embedding['weights'] = weights

    return embedding


def load_embeddings(embeddings_path, vocab):
    w2v = Word2Vec.load_word2vec_format(embeddings_path, binary=True, unicode_errors='ignore')
    embedding = dict()
    embedding['size'] = len(vocab)
    embedding['dims'] = w2v.vector_size
    embedding['trainable'] = True

    weights = np.random.uniform(-1.0, 1.0, (embedding['size'], embedding['dims']))

    for word in vocab:
        if word in w2v:
            weights[vocab[word]] = np.asarray(w2v[word])

    embedding['weights'] = weights
    return embedding


def get_training_batches(data, batch_size):
    num_batches = int(len(data) / batch_size)
    for batch_i in range(num_batches):
        start = batch_i * batch_size
        end = min((batch_i + 1) * batch_size, len(data))

        yield data[start: end]