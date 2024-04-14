import logging
from typing import Dict, List, Iterable, Tuple, Any

import numpy as np
from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, AdjacencyField, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from chunk_oie.dataset_readers.dataset_helper import *

import json
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("chunk_reader")
class Chunk_Reader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 validation: bool = False,
                 data_type: str = 'oo',
                 chunk_type: str = 'chunk_oia',
                 verbal_indicator: bool = True,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._pos_tag_indexers = {"pos_tags": SingleIdTokenIndexer(namespace="pos_labels")}
        self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dependency_labels")}
        self._const_tag_indexers = {"const_tags": SingleIdTokenIndexer(namespace="constituency_labels")}
        self._domain_identifier = domain_identifier
        self._validation = validation
        self._data_type = data_type
        self._chunk_type = chunk_type
        self._verbal_indicator = verbal_indicator
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.lowercase_input = "uncased" in bert_model_name


    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)
        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]
        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        file_in = open(file_path, 'r', encoding='utf-8')
        json_sent = json.load(file_in)
        file_in.close()
        i = 0
        for index, item in enumerate(json_sent):
            if self._chunk_type == 'conll':
                tokens = item['tokens']
                bound_tags = item['bound']
                token_tags = item['conll']
            else:
                tokens = item['words']
                bound_tags = item['bound_tags']
                token_tags = item['token_tags']

            pos_tags = item['spacy_pos']
            dep_edges = item['dep_graph_edges']
            dep_nodes = item['dep_graph_nodes']

            i += 1
            if self._validation:
                yield self.text_to_instance(tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index)
            else:
                yield self.text_to_instance(tokens, bound_tags, token_tags, pos_tags, dep_edges, dep_nodes, index)


    def text_to_instance(self, tokens: List[str], bound_tags: List[int], token_tags: List[str],
                         pos_tags: List[str], dep_edges: List[List], dep_nodes: List[str], sent_id: int=None) -> Instance:

        metadata_dict = {}

        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(tokens)
        metadata_dict["offsets"] = offsets


        # In order to override the indexing mechanism, we need to set the `text_id` attribute directly.
        # This causes the indexing to use this id.
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces], token_indexers=self._token_indexers)

        pos_wordpc_tags = convert_dep_tags_to_wordpiece_dep_tags(pos_tags, offsets)
        pos_field = TextField([Token(t) for t in pos_wordpc_tags], token_indexers=self._pos_tag_indexers)

        # convert word label dependency labels to wordpiece labels
        dep_graph_nodes = convert_dep_tags_to_wordpiece_dep_tags(dep_nodes, offsets)
        dep_edges_tuple = [(item[0][0], item[0][1]) for item in dep_edges]
        dep_edges_tuple = convert_dep_adj_to_wordpiece_dep_adj(dep_edges_tuple, start_offsets, offsets)
        dep_field = TextField([Token(t) for t in dep_graph_nodes], token_indexers=self._dep_tag_indexers)
        dep_adj_field = AdjacencyField(dep_edges_tuple, dep_field, padding_value=0)

        if bound_tags is not None:
            metadata_dict["bound_tags"] = bound_tags
            metadata_dict["token_tags"] = token_tags
            new_bound_tags = convert_bound_indices_to_wordpiece_indices(bound_tags, offsets)

            if self._chunk_type == 'conll':
                new_token_tags = convert_tags_to_wordpiece_tags(token_tags, offsets)
            else:
                new_token_tags = convert_tag_indices_to_wordpiece_indices(token_tags, offsets)

            bound_tags_field = SequenceLabelField(new_bound_tags, text_field)
            token_tags_field = SequenceLabelField(new_token_tags, text_field)
            fields = {'tokens': text_field, 'token_tags': token_tags_field, 'bound_tags': bound_tags_field,
                      'pos_tags': pos_field, 'dep_nodes': dep_field, 'dep_edges': dep_adj_field}
        else:
            fields = {'tokens': text_field, 'pos_tags': pos_field}

        metadata_dict["words"] = tokens
        metadata_dict["validation"] = self._validation
        if sent_id is not None:
            metadata_dict['sent_id'] = sent_id

        fields["metadata"] = MetadataField(metadata_dict)

        return Instance(fields)
