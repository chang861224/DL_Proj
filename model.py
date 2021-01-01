import sklearn_crfsuite
import torch
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from transformers import BertModel, BertTokenizer, BertForTokenClassification

def CRF(x_train, y_train, x_test, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    # print(crf)
    y_pred = crf.predict(x_test)
    y_pred_mar = crf.predict_marginals(x_test)

    # print(y_pred_mar)

    labels = list(crf.classes_)
    labels.remove('O')
    f1score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results
    print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    return y_pred, y_pred_mar, f1score


def BERTTransform(text):
    PRETRAINED_MODEL = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    
    tokens = tokenizer.tokenize(text)
    indexes = tokenizer.convert_tokens_to_ids(tokens)

    segments_ids = [1] * len(tokens)

    tokens_tensor = torch.tensor([indexes])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained(PRETRAINED_MODEL, output_hidden_states=True)
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    """
    print('Number of layers:', len(hidden_states))
    layer_i = 0
    print('Number of batches:', len(hidden_states[layer_i]))
    batch_i = 0
    print('Number of tokens:', len(hidden_states[layer_i][batch_i]))
    token_i = 0
    print('Number of hidden units:', len(hidden_states[layer_i][batch_i][token_i]))

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
    return tokens, token_vecs_sum
