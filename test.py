import torch
import re
from transformers import BertModel, BertTokenizer, BertForTokenClassification
from scipy.spatial.distance import cosine
from preprocessing import loadRawData

PRETRAINED_MODEL = 'bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

data = loadRawData('./textdata/development_1.txt')
conversation = re.split(r'[醫師：|民眾：|家屬：]', data[0])
while('' in conversation):
    conversation.remove('')
text = '[CLS]' + conversation[0] + '[SEP]'

tokens = tokenizer.tokenize(text)
#print(tokens)
#print(type(tokens))
indexes = tokenizer.convert_tokens_to_ids(tokens)

#print(tokens)
#for tup in zip(tokens, indexes):
#    print(tup)

segments_ids = [1] * len(tokens)
#print(segments_ids)

tokens_tensor = torch.tensor([indexes])
segments_tensors = torch.tensor([segments_ids])

#print(tokens_tensor.size())
#print(segments_tensors.size())

model = BertModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
model.eval()

#print(model)

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

print('Number of layers:', len(hidden_states))
layer_i = 0
print('Number of batches:', len(hidden_states[layer_i]))
batch_i = 0
print('Number of tokens:', len(hidden_states[layer_i][batch_i]))
token_i = 0
print('Number of hidden units:', len(hidden_states[layer_i][batch_i][token_i]))
"""
print('Type of hidden_states:', type(hidden_states))
print('Tensor shape for each layer:', hidden_states[0].size())
"""
token_embeddings = torch.stack(hidden_states, dim=0)
print(token_embeddings.size())

token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(token_embeddings.size())

token_embeddings = token_embeddings.permute(1, 0, 2)
print(token_embeddings.size())

token_vec_cat = []

for token in token_embeddings:
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    token_vec_cat.append(cat_vec)

print('Shape is: %d x %d' % (len(token_vec_cat), len(token_vec_cat[0])))

token_vecs_sum = []

for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)

print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

for i, token_str in enumerate(tokens):
    print(i, token_str)

#doc = torch.sum(torch.cat((token_vecs_sum[368], token_vecs_sum[369]), dim=0).reshape(2, 768), dim=0)
#pat = torch.sum(torch.cat((token_vecs_sum[376], token_vecs_sum[377]), dim=0).reshape(2, 768), dim=0)
#print(1 - cosine(doc, pat))
#print(1 - cosine(token_vecs_sum[493], token_vecs_sum[494]))

