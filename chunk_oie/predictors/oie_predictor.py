from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token



@Predictor.register('oie_predictor')
class OIEPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)


    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:

        tokens = inputs['tokens']
        bounds = inputs['bounds']
        types = inputs['types']
        dep_edges = inputs['dep_graph_edges']
        dep_nodes = inputs['dep_graph_nodes']
        pos_list = inputs['spacy_pos']

        tuple_list = []
        for index, pos in enumerate(pos_list):
            if pos == "VERB":
                verb_indicator = [0] * len(tokens)
                verb_indicator[index] = 1

                # Create instances
                instance = self._dataset_reader.text_to_instance(tokens, verb_indicator, bounds, types, dep_edges, dep_nodes)

                # Run model
                model_result = self._model.forward_on_instance(instance)

                tuple_list.append(model_result)

        return tuple_list


