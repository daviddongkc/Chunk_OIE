from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token



@Predictor.register('chunk_predictor')
class Chunk_Predictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)


    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:

        sent_tokens = inputs["tokens"]
        pos_tags = inputs['spacy_pos']
        dep_nodes = inputs['dep_graph_nodes']
        dep_edges = inputs['dep_graph_edges']
        if 'send_id' in inputs.keys():
            sent_id = inputs['sent_id']
            instance = self._dataset_reader.text_to_instance(sent_tokens, None, None, pos_tags, dep_edges, dep_nodes, sent_id)
        else:
            instance = self._dataset_reader.text_to_instance(sent_tokens, None, None, pos_tags, dep_edges, dep_nodes, 0)

        # Run model
        model_result = self._model.forward_on_instance(instance)

        return model_result


