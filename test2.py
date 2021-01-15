import torch
import numpy as np
import pandas as pd
from ckiptagger import WS
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
from preprocessing import loadRawData
from model import BERTTransform

PRETRAINED_MODEL = 'bert-base-chinese'
model = BertModel.from_pretrained(PRETRAINED_MODEL)
token = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

def get_data(path):
    words = []

    data = loadRawData(path)
    text = [data[key] for key in data.keys()]

    ws = WS('./data')

    tokens = ws(text)

    for token in tokens:
        words.extend([w for w in token if w not in words])

    return words

def get_bert_embedding(path, _class):
    words = get_data(path)
#    file_word = open("word_embed.txt", "a+", encoding="utf-8")
#    file_word.write(str(len(words)) + " " + "768" + "\n")

    for word in words:
        words_embed = np.zeros(768)
        inputs = token.encode_plus(word, padding="max_length", truncation=True, max_length=512, add_special_tokens=True, return_tensors="pt")
        out = model(**inputs)
        word_len = len(word)
        out_ = out[0].detach().numpy()

        for i in range(1, word_len + 1):
            print(out_.shape)
            out_str = out_[0][i]
            words_embed += out_str

        words_embed = words_embed / word_len
        words_embedding = words_embed.tolist()

        string = word

        for ind, vec in enumerate(_class):
            if 1 - cosine(torch.FloatTensor(words_embedding), vec) >= 0.5:
                string += str(ind)

        print(string)
#        result = word + " " + " ".join("%s" % embed for embed in words_embedding) + "\n"
#        file_word.write(result)

#    file_word.close()

classification = pd.read_csv('./classification.csv')
df = pd.read_csv('./classification.csv')
class_vec = []
#df = df['class'].to_list()
for data in df['class']:
    string = '[CLS]' + data + '[SEP]'
    _, vec = BERTTransform(string)
    class_vec.append(torch.mean(torch.cat((vec), dim=0).reshape(len(data) + 2, 768), dim=0))
    print(data, torch.mean(torch.cat((vec), dim=0).reshape(len(data) + 2, 768), dim=0)[:5])

#words = get_bert_embedding('./textdata/development_2.txt')
words = get_bert_embedding('./textdata/development_3.txt', class_vec)
#words = get_data('./textdata/development_2.txt')
print('Finish!!')
print('Please check the file "word_embed.txt"')
