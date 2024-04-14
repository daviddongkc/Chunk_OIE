from typing import Dict, List, Optional, Any, Union
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding, LogSoftmax
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import Metric
from chunk_oie.models.modules import DepGCN

@Model.register("chunk_pos_model")
class Chunk_Model(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 embedding_dropout: float = 0.0,
                 dependency_label_dim: int = 400,
                 use_graph: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 tuple_metric: Metric = None) -> None:
        super(Chunk_Model, self).__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_bound_classes = 2
        self.num_type_classes = self.vocab.get_vocab_size("labels")

        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_type_classes)
        self.bound_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_bound_classes)

        self.num_pos_labels = self.vocab.get_vocab_size("pos_labels")
        self.pos_embedding = Embedding(self.num_pos_labels, self.bert_model.config.hidden_size, padding_idx=0)

        self._use_graph = use_graph
        if self._use_graph:
            self.num_dep_labels = self.vocab.get_vocab_size("dependency_labels")
            self.dep_gcn = DepGCN(self.num_dep_labels, dependency_label_dim, self.bert_model.config.hidden_size,
                                  self.bert_model.config.hidden_size)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._token_based_metric = tuple_metric
        self.LogSoftmax = LogSoftmax()
        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                metadata: List[Any],
                pos_tags: torch.Tensor,
                dep_nodes: Dict[str, torch.Tensor]=None,
                dep_edges: Dict[str, torch.Tensor]=None,
                bound_tags: torch.Tensor=None,
                token_tags: torch.Tensor=None
                ):

        # mask dim = (batch_size, len(tokens))
        mask = get_text_field_mask(tokens)
        # bert_embed_dim = (batch_size, len(tokens), 768)
        bert_embeddings, _ = self.bert_model(input_ids=tokens["tokens"],
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)

        embed_bert = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embed_bert.size()
        embed_pos = self.pos_embedding(pos_tags['pos_tags'])
        embedded_text_input = embed_bert + embed_pos
        bound_logits = self.bound_projection_layer(embedded_text_input)
        reshaped_bound_log_probs = bound_logits.view(-1, self.num_bound_classes)
        bound_probabilities = F.softmax(reshaped_bound_log_probs, dim=-1).view([batch_size, sequence_length, self.num_bound_classes])

        # use only the boundary token as input to project layer!
        type_logits = self.tag_projection_layer(embedded_text_input)
        reshaped_tag_log_probs = type_logits.view(-1, self.num_type_classes)
        type_probabilities = F.softmax(reshaped_tag_log_probs, dim=-1).view([batch_size, sequence_length, self.num_type_classes])

        output_dict = {"bound_logits": bound_logits, "bound_probabilities": bound_probabilities,
                       "type_logits": type_logits, "type_probabilities": type_probabilities}

        if not metadata[0]['validation']:
            bound_loss = sequence_cross_entropy_with_logits(bound_logits, bound_tags, mask, label_smoothing=self._label_smoothing)
            tag_loss = sequence_cross_entropy_with_logits(type_logits, token_tags, mask, label_smoothing=self._label_smoothing)
            output_dict["loss"] = bound_loss + 0.8*tag_loss

        # We need to retain the mask in the output dictionary so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        # We add in the offsets here, so we can compute the un-wordpieced tags.
        words, offsets = zip(*[(x["words"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["wordpiece_offsets"] = list(offsets)

        if metadata[0]['validation']:
            output_dict = self.decode(output_dict)
            sent_ids = []
            if bound_tags is not None:
                bound_tags_list, type_tags_list, = [], []
                for item in metadata:
                    bound_tags_list.append(item['bound_tags'])
                    type_tags_list.append(item['token_tags'])
                    sent_ids.append(item['sent_id'])
                self._token_based_metric(output_dict, bound_tags_list, type_tags_list, sent_ids)

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bound_predictions = output_dict['bound_probabilities']
        type_predictions = output_dict['type_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if bound_predictions.dim() == 3:
            bound_predictions_list = [bound_predictions[i].detach().cpu() for i in range(bound_predictions.size(0))]
        else:
            bound_predictions_list = [bound_predictions]

        if type_predictions.dim() == 3:
            type_predictions_list = [type_predictions[i].detach().cpu() for i in range(type_predictions.size(0))]
        else:
            type_predictions_list = [type_predictions]

        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()


        wordpiece_bound_tags, word_bound_tags, wordpiece_bound_probs, word_bound_probs, tag_ids = [], [], [], [], []
        wordpiece_type_tags, word_type_tags, wordpiece_type_probs, word_type_probs = [], [], [], []

        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for bound_predictions, type_predictions, length, offsets \
                in zip(bound_predictions_list, type_predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]):

            bound_tags = torch.argmax(bound_predictions[:length], dim=1).tolist()
            wordpiece_bound_tags.append(bound_tags)
            word_bound_tags.append([bound_tags[i] for i in offsets])

            type_ints, _ = viterbi_decode(type_predictions[:length], transition_matrix, allowed_start_transitions=start_transitions)
            # type_ints = torch.argmax(type_predictions[:length], dim=1).tolist()
            type_tags = [self.vocab.get_token_from_index(x, namespace="labels") for x in type_ints]
            wordpiece_type_tags.append(type_tags)
            word_type_tags.append([type_tags[i] for i in offsets])

            # get the confidence score of predicted tags
            bound_probs = [float(bound_predictions[i][j]) for i, j in enumerate(bound_tags)]
            word_bound_probs.append([bound_probs[i] for i in offsets])
            wordpiece_bound_probs.append(bound_probs)

            type_probs = [float(type_predictions[i][j]) for i, j in enumerate(type_ints)]
            word_type_probs.append([type_probs[i] for i in offsets])
            wordpiece_type_probs.append(type_probs)

            # tag_ids.append([tags[i] for i in offsets])

        output_dict['wordpiece_bound_tags'] = wordpiece_bound_tags
        output_dict['bound_tags'] = word_bound_tags
        output_dict['wordpiece_type_tags'] = wordpiece_type_tags
        output_dict['type_tags'] = word_type_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded by either an identical I-XXX tag or
        a B-XXX tag. In order to achieve this constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.
        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix


    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.
        Returns
        -------
        start_transitions : torch.Tensor
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions
