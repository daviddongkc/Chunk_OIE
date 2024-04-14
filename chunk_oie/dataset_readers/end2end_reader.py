import logging
from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from chunk_oie.dataset_readers.dataset_helper import *
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("e2e_reader")
class OIE_Reader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 validation: bool = False,
                 data_type: str = 'oo',
                 chunk_type: str = 'oia',
                 verbal_indicator: bool = True,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # self._type_indexers = {"types": SingleIdTokenIndexer(namespace="type_labels")}
        self._dep_tag_indexers = {"dep_tags": SingleIdTokenIndexer(namespace="dependency_labels")}
        self._pos_tag_indexers = {"pos_tags": SingleIdTokenIndexer(namespace="pos_labels")}
        self._domain_identifier = domain_identifier
        self._validation = validation
        self._verbal_indicator = verbal_indicator
        self._data_type = data_type
        self._chunk_type = chunk_type

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

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
        n = 0

        for sentence in json_sent:
            # n += 1
            # if n > 1000:
            #     break

            if self._chunk_type == 'conll':
                bounds = sentence['conll_bounds']
                types = sentence['conll_types']
            else:
                bounds = sentence['bounds']
                types = sentence['types']
            tokens = sentence['tokens']
            dep_edges = sentence['dep_graph_edges']
            dep_nodes = sentence['dep_graph_nodes']
            pos_tags = sentence['spacy_pos']

            if self._validation:
                if self._verbal_indicator:
                    # this is when no verbal indicator is given during validation
                    # we need to check the pos tags of each token and determined verbal indicators
                    # each verbal indicator will correspond to a verbal sequence.
                    pos_list = sentence['spacy_pos']
                    for index, pos in enumerate(pos_list):
                        if pos == "VERB":
                            verb_indicator = [0] * len(tokens)
                            verb_indicator[index] = 1
                            yield self.text_to_instance(tokens, verb_indicator, pos_tags, bounds, types, dep_edges, dep_nodes)
                else:
                    verb_indicator = sentence['verb_label']
                    yield self.text_to_instance(tokens, verb_indicator, pos_tags, bounds, types, dep_edges, dep_nodes)

            else:
                if len(tokens) > 120:
                    continue

                if self._data_type == 'oo':
                    tags_oo = sentence['tags_oo']
                    tags_oo_v = sentence['tags_oo_v']

                    for tags, verb_indicator in zip(tags_oo, tags_oo_v):
                        tags_IO_scheme = []
                        for tag in tags:
                            tag = tag.replace('B-', '')
                            tag = tag.replace('I-', '')
                            tags_IO_scheme.append(tag)

                        yield self.text_to_instance(tokens, verb_indicator, pos_tags, bounds, types, dep_edges, dep_nodes, tags=tags_IO_scheme)

                elif self._data_type == 'aug':
                    tags_aug = sentence['tags_aug']
                    tags_aug_v = sentence['tags_aug_v']

                    for tags, verb_indicator in zip(tags_aug, tags_aug_v):
                        tags_IO_scheme = []
                        for tag in tags:
                            tag = tag.replace('B-', '')
                            tag = tag.replace('I-', '')
                            tags_IO_scheme.append(tag)
                        yield self.text_to_instance(tokens, verb_indicator, pos_tags, bounds, types, dep_edges, dep_nodes, tags=tags_IO_scheme)


    def text_to_instance(self, tokens: List[str], verb_label: List[int], pos_tags: List[str], bounds: List[int],
                         types: List[str], dep_edges: List[List], dep_nodes: List[str], tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

        verb_index = verb_label.index(1)
        types[verb_index] = 'Predicate'
        bounds[verb_index] = 1
        try:
            bounds[verb_index-1] = 1
        except:
            print('Verb is the first one')

        metadata_dict: Dict[str, Any] = {}
        phrase_bounds, phrase_types, phrase_word_bounds = [], [], []

        #get node types and bounds at phase level
        start_index = 0
        for i, (bound, type) in enumerate(zip(bounds, types)):
            if bound == 1:
                phrase_types.append(type)
                phrase_word_bounds.append((start_index, i))
                start_index = i + 1

        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(tokens)
        new_verbs = convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
        metadata_dict["offsets"] = start_offsets
        metadata_dict["offsets_end"] = offsets
        # In order to override the indexing mechanism, we need to set the `text_id` attribute directly.
        # This causes the indexing to use this id.
        text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                               token_indexers=self._token_indexers)
        verb_indicator = SequenceLabelField(new_verbs, text_field)

        pos_wordpc_tags = convert_dep_tags_to_wordpiece_dep_tags(pos_tags, offsets)
        pos_field = TextField([Token(t) for t in pos_wordpc_tags], token_indexers=self._pos_tag_indexers)

        wordpiece_bounds = convert_bound_indices_to_wordpiece_indices(bounds, offsets)
        start_index = 0
        for i, bound in enumerate(wordpiece_bounds):
            if bound == 1:
                phrase_bounds.append((start_index, i))
                start_index = i + 1

        fields: Dict[str, Field] = {'tokens': text_field, 'verb_indicator': verb_indicator,  'pos_tags': pos_field}


        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index]

        metadata_dict["words"] = tokens
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index
        metadata_dict["validation"] = self._validation
        metadata_dict["phrase_bounds"] = bounds
        metadata_dict["phrase_wordpiece_bounds"] = wordpiece_bounds
        metadata_dict['phrase_bounds_tuple'] = phrase_bounds
        metadata_dict["phrase_types"] = types
        metadata_dict["dep_edges"] = dep_edges
        metadata_dict["dep_nodes"] = dep_nodes

        if tags:
            new_tags = convert_tags_to_wordpiece_IOtags(tags, offsets)
            fields['tags'] = SequenceLabelField(new_tags, text_field)
            metadata_dict["gold_tags"] = tags

            new_bound_tags = convert_bound_indices_to_wordpiece_indices(bounds, offsets)

            if self._chunk_type == 'conll':
                new_type_tags = convert_tags_to_wordpiece_tags(types, offsets)
            else:
                new_type_tags = convert_tag_indices_to_wordpiece_indices(types, offsets)

            bound_tags_field = SequenceLabelField(new_bound_tags, text_field)
            type_tags_field = SequenceLabelField(new_type_tags, text_field)
            fields['bound_tags'] = SequenceLabelField(bound_tags_field, text_field, label_namespace="bound_labels")
            fields['type_tags'] = SequenceLabelField(type_tags_field, text_field, label_namespace="chunk_labels")



        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
