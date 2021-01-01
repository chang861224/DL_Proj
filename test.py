import torch
import pandas as pd
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from scipy.spatial.distance import cosine
from preprocessing import loadRawData
from model import BERTTransform

## CLASSIFICATION
classification = pd.read_csv('./classification.csv')
df = pd.read_csv('./classification.csv')
class_vec = []
#df = df['class'].to_list()
for data in df['class']:
    string = '[CLS]' + data + '[SEP]'
    _, vec = BERTTransform(string)
    class_vec.append(torch.mean(torch.cat((vec), dim=0).reshape(len(data) + 2, 768), dim=0))
    print(data, torch.mean(torch.cat((vec), dim=0).reshape(len(data) + 2, 768), dim=0)[:5])

## DATA
data = loadRawData('./textdata/development_2.txt')

for key in data.keys():
    conversation = data[key].split('：')

    while('' in conversation):
        conversation.remove('')

    for sentence in conversation:
        text = '[CLS]' + sentence + '[SEP]'
        tokens, token_vecs_sum = BERTTransform(text)

        for i, token_str in enumerate(tokens):
            if i != 0 and i != len(token_str)-1 and token_str != '，' and token_str != '。':
                for j in range(len(class_vec)):
                    if (1 - cosine(token_vecs_sum[i], class_vec[j]) > 0.6):
                        print(token_str, j, 1-cosine(token_vecs_sum[i], class_vec[j]))
#            print(i, token_str, token_vecs_sum[i][:5])

#doc = torch.sum(torch.cat((token_vecs_sum[4:6]), dim=0).reshape(2, 768), dim=0)
#print(doc[:5])
#pat = torch.sum(torch.cat((token_vecs_sum[376], token_vecs_sum[377]), dim=0).reshape(2, 768), dim=0)
#print(1 - cosine(doc, vec))
#print(1 - cosine(token_vecs_sum[493], token_vecs_sum[494]))

