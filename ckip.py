from ckiptagger import WS, POS, NER
from preprocessing import loadRawData

ws = WS('./data')
pos = POS('./data')
ner = NER('./data')

text = loadRawData('./textdata/document_1.txt')
#text = loadRawData('./textdata/document_2.txt')

text_list = [text[key] for key in text.keys()]

word_sentence_list = ws(text_list)
pos_sentence_list = pos(word_sentence_list)
entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

del ws
del pos
del ner

for i, sentence in enumerate(text_list):
    print(sentence[i])
    print('article_id\tstart_position\tend_position\tentity_text\tentity_type')
    for entity in sorted(entity_sentence_list):
        print(entity)

