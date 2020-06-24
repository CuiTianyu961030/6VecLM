import torch
from torch.autograd import Variable
import torch.nn.functional as F
from gensim.models import word2vec
import numpy as np
from ipv6_transformer import *

word2vec_model_path = 'models/ipv62vec.model'
data_path = "data/processed_data/word_data.txt"

encoder_input_length = 16
total_epoch = 1

train_data_size = 100000


def greedy_decode(model, word2vec_model, src, src_mask, max_len, start_symbol, temperature):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        # prob = models.generator(out[:, -1])
        # _, next_word = torch.max(prob, dim=1)
        vector = model.generator(out[:, -1])
        next_word = next_generation(word2vec_model, vector, temperature, i + 17)
        # next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def write_data(target_generation, generation_path):
    f = open(generation_path, "w")
    for address in target_generation:
        f.write(address + "\n")
    f.close()


if __name__ == "__main__":
    word2vec_model = word2vec.Word2Vec.load(word2vec_model_path)

    model = torch.load("models/ipv6_transformer_s6_e10_t0010.model")
    model.eval()
    f = open(data_path, "r")
    data = np.array([[word2id(nybble, word2vec_model) for nybble in address[:-1].split()]
                         for address in f.readlines()[:train_data_size]])
    test_data = np.array(data[:, :encoder_input_length])
    start_symbles = np.array(data[:, encoder_input_length])
    f.close()

    for temperature in [0.020, 0.030, 0.040, 0.050,
                        0.060, 0.070, 0.080, 0.090, 0.100, 0.200, 0.500]:
        print(temperature)
        target_generation = []
        for i in range(len(test_data)):
            src = Variable(torch.LongTensor([test_data[i]]))
            src_mask = Variable(torch.ones(1, 1, encoder_input_length))
            predict = greedy_decode(model, word2vec_model, src, src_mask, max_len=32-encoder_input_length,
                                    start_symbol=start_symbles[i], temperature=temperature).numpy()
            predict = np.append(np.array(test_data[i]), predict)
            predict_words = [id2word(i, word2vec_model) for i in predict]
            # predict_words_str = " ".join(predict_words)
            # print(predict_words_str)
            predict_address = [word[0] for word in predict_words]
            count = 0
            predict_address_str = ""
            for i in predict_address:
                predict_address_str += i
                count += 1
                if count % 4 == 0 and count != 32:
                    predict_address_str += ":"
            target_generation.append(predict_address_str)
        generation_path = "data/generation_data/candidate_s6_e1_t" + str(temperature) + ".txt"
        target_generation = list(set(target_generation))
        write_data(target_generation, generation_path)
