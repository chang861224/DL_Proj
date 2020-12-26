import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import *
from model import CRF

def Dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data=f.readlines()#.encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list=list()
    idx=0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx+=1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)
    
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list=train_test_split(data_list, article_id_list, test_size=0.33, random_state=42)
    
    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list 

def Word2Vector(data_list, embedding_dict):
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0] # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)
    return embedding_list

def Feature(embed_list):
    feature_list = list()
    for idx_list in range(len(embed_list)):
        feature_list_tmp = list()
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict['dim_' + str(idx_vec+1)] = embed_list[idx_list][idx_tuple][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    return feature_list

def Preprocess(data_list):
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list



dim = 0
word_vecs= {}
# open pretrained word vector file
with open('cna.cbow.cwe_p.tar_g.512d.0.txt') as f:
    for line in f:
        tokens = line.strip().split()

        # there 2 integers in the first line: vocabulary_size, word_vector_dim
        if len(tokens) == 2:
            dim = int(tokens[1])
            continue
    
        word = tokens[0] 
        vec = np.array([ float(t) for t in tokens[1:] ])
        word_vecs[word] = vec

#print('vocabulary_size: ',len(word_vecs),' word_vector_dim: ',vec.shape)


file_path = './textdata/train_1_update.txt'
trainingset, position, mentions = loadInputFile(file_path)

data_path='data/train.data'
CRFFormatData(trainingset, position, data_path)

#print(mentions['陳明明醫師'])

data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(data_path)

trainembed_list = Word2Vector(traindata_list, word_vecs)
#print(trainembed_list)
testembed_list = Word2Vector(testdata_list, word_vecs)

# CRF - Train Data (Augmentation Data)
x_train = Feature(trainembed_list)
y_train = Preprocess(traindata_list)

# CRF - Test Data (Golden Standard)
x_test = Feature(testembed_list)
y_test = Preprocess(testdata_list)

y_pred, y_pred_mar, f1score = CRF(x_train, y_train, x_test, y_test)

print(f1score)

