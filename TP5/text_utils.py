
import struct
import numpy as np
import torch
import torch.nn as nn


RANDOMIZED = False  # Mettre à True si vous n'avez pas accès au fichier Word2Vec


print("Computing vocabulary...")
vocabulary = set()

with open("/mnt/ssd/flickr30k/results_20130124.token", encoding="utf-8") as f:
    for line in f:
        _, words = line.strip().split("\t")
        for word in words.split(" "):
            vocabulary.add(word)

print("Number of words:", len(vocabulary))


def read_until(open_file):
    while True:
        char = open_file.read(1)
        if char == b' ':
            break
        yield char


def read_word2vec():
    with open("/mnt/ssd/word2vec/GoogleNews-vectors-negative300.bin", "rb") as f:
        first_line = f.readline()
        n_words, n_dim = map(int, first_line.split())

        for n_word in range(n_words):
            word = b''.join(read_until(f)).decode('utf-8', errors='ignore')
            vector = struct.unpack('f' * n_dim, f.read(n_dim * 4))
            yield word, vector


def create_embedding_layer(vocab, randomized=False):
    # Indice 0 réservé au padding
    idx_to_word = [""]
    word_to_idx = {"": 0}
    weights = [np.zeros((1, 300), dtype=np.float32)]

    if randomized:
        for word in sorted(vocab):
            idx_to_word.append(word)
            word_to_idx[word] = len(idx_to_word) - 1
            weights.append(np.random.rand(1, 300).astype(np.float32))
    else:
        for word, vector in read_word2vec():
            if word in vocab and word not in word_to_idx:
                idx_to_word.append(word)
                word_to_idx[word] = len(idx_to_word) - 1
                weights.append(np.array([vector], dtype=np.float32))

    n_words = len(weights)
    n_dim = len(weights[0][0])

    weights = np.concatenate(weights, axis=0)

    emb_layer = nn.Embedding(n_words, n_dim, padding_idx=0)
    emb_layer.load_state_dict({"weight": torch.Tensor(weights)})

    if not randomized:
        emb_layer.weight.requires_grad = False

    return emb_layer, idx_to_word, word_to_idx


def tokenize(sentence, word2idx, max_length=20):
    res = [0]  # 0 est souvent l'indice de début/fin ou padding
    sentence = sentence.lower()

    for word in sentence.split():
        if word in word2idx:
            res.append(word2idx.get(word, 0))

    if len(res) > max_length:
        return res[:max_length]

    return res + [0] * (max_length - len(res))


def untokenize(tokens, idx2word):
    return [idx2word[int(x)] for x in tokens]


pretrained_embeddings, idx2word, word2idx = create_embedding_layer(
    vocabulary, RANDOMIZED
)

print("Vocabulary size:", len(idx2word))


if __name__ == "__main__":
    print("Embedding shape:", pretrained_embeddings.weight.shape)
    print("Embeddings gelés:", not pretrained_embeddings.weight.requires_grad)

    example = "Two young guys with shaggy hair"
    tokens = tokenize(example, word2idx)

    print("Tokens:", tokens)
    print("Decoded:", untokenize(tokens, idx2word))
