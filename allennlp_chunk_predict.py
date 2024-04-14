import json
import shutil
import sys
from argparse import ArgumentParser
from allennlp.run import run
# from allennlp.commands import main
from allennlp.commands.train import train_model_from_file
from allennlp.commands.fine_tune import fine_tune_model_from_file_paths
from allennlp.common.util import import_submodules

from tqdm import tqdm
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.models.archival import archive_model


import spacy



def predict_from_txt(model_path, file_in):

    archive = load_archive(model_path)
    predictor = Predictor.from_archive(archive, 'chunk_predictor')
    nlp = spacy.load('en_core_web_trf', disable=['ner', 'textcat'])

    sent_list = open(file_in, 'r', encoding='utf-8').readlines()

    sent_dict_list = []

    sent_id = 0
    for sent in tqdm(sent_list):
        if len(sent) < 10:
            continue
        sent_id += 1
        sentence = {}
        sentence['sent'] = sent

        spacy_doc = nlp(sent)

        token_string_list, index_list, pos_list, tag_list, dep_list, head_list, head_index_list = [], [], [], [], [], [], []
        for token in spacy_doc:
            token_string, token_index, token_pos, token_tag, token_dep = token.orth_, token.i, token.pos_, token.tag_, token.dep_

            token_string_list.append(token_string)
            index_list.append(token_index)
            pos_list.append(token_pos)
            tag_list.append(token_tag)
            dep_list.append(token_dep)

            token_head = token.head
            token_head_string, token_head_index = token_head.orth_, token_head.i

            head_list.append(token_head_string)
            head_index_list.append(token_head_index)

        sentence['spacy_pos'] = pos_list
        sentence['tokens'] = token_string_list
        sentence['sent_id'] = sent_id

        adj_dep_edges = []
        for child_index, (head_index, dep_tag) in enumerate(zip(head_index_list, dep_list)):
            adj_dep_edges.append(((head_index, child_index), dep_tag))

        sentence['dep_graph_nodes'] = dep_list
        sentence['dep_graph_edges'] = adj_dep_edges

        result = predictor.predict_json((sentence))
        words = result['words']
        bounds = result['bound_tags']
        types = result['type_tags']

        sentence['bounds'] = bounds
        sentence['types'] = types


        sent_dict_list.append(sentence)

    file_out = open(file_in.replace('.txt', '.json'), 'w', encoding='utf-8')
    json.dump(sent_dict_list, file_out)


if __name__ == '__main__':
    import_submodules('chunk_oie')

    # # this is the chunk_oia model path
    # model_path = "trained_model/chunk_model/model.tar.gz"
    #
    # # file_in = 'data/oie/lsoie_sci_train_chunk.json'
    # # file_out = 'data/oie/lsoie_sci_train.json'
    #
    # file_in = 'data/oie/lsoie_wiki_test.txt'
    # predict_from_txt(model_path, file_in)

    json_data1 = json.load(open('data/oie/lsoie_wiki_test.json', 'r', encoding='utf-8'))
    json_data2 = json.load(open('data/oie/lsoie_wiki_test.json', 'r', encoding='utf-8'))

    print('haha')