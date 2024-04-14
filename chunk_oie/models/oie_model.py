from typing import Dict, List, Optional, Any, Union
import logging
from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.modules import Seq2SeqEncoder
from allennlp.training.metrics import Metric
import torch.nn as nn

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("oie_model")
class OIE_Model(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 phrase_attention: Seq2SeqEncoder,
                 embedding_dropout: float = 0.0,
                 dependency_label_dim: int = 400,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 type_label_dim: int = 768,
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 tuple_metric: Metric = None) -> None:
        super(OIE_Model, self).__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("labels")
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)
        self.num_type_labels = self.vocab.get_vocab_size("type_labels")
        self.num_dep_labels = self.vocab.get_vocab_size("dependency_labels")

        self.type_embedding = Embedding(self.num_type_labels, type_label_dim, padding_idx=0)
        self.dep_embedding = Embedding(self.num_dep_labels, dependency_label_dim, padding_idx=0)
        self.dep_attn = Linear(dependency_label_dim + self.bert_model.config.hidden_size,
                               self.bert_model.config.hidden_size)

        self.dep_gcn = DepGCN(self.num_dep_labels, dependency_label_dim, self.bert_model.config.hidden_size,
                              self.bert_model.config.hidden_size)

        self.phrase_attention = phrase_attention
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._token_based_metric = tuple_metric
        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                dep_nodes: Dict[str, torch.Tensor],
                dep_edges: Dict[str, torch.Tensor],
                verb_indicator: torch.Tensor,
                phrase_types: torch.Tensor,
                metadata: List[Any],
                tags: torch.LongTensor=None,
                optimizer=None):

        mask = get_text_field_mask(tokens)

        bert_embeddings, _ = self.bert_model(input_ids=tokens["tokens"],
                                             token_type_ids=verb_indicator,
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)

        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()
        embed_type = self.type_embedding(phrase_types['types'])
        _, phrase_seq_length, embed_dim = embed_type.size()

        cuda_device = embed_type.get_device()
        if cuda_device < 0:
            embed_phrase = torch.zeros([batch_size, phrase_seq_length, embed_dim], dtype=torch.float32)
        else:
            embed_phrase = torch.zeros([batch_size, phrase_seq_length, embed_dim], dtype=torch.float32, device=cuda_device)

        for i in range(0, batch_size):
            phrase_bounds = metadata[i]['phrase_bounds_tuple']
            embed_bert = embedded_text_input[i,:,:]
            for j, (x, y) in enumerate(phrase_bounds):
                if x == y:
                    embed_phrase[i,j,:] = embed_bert[x,:]
                else:
                    embed_phrase[i,j,:] = torch.mean(embed_bert[x:y+1,:], dim=0)

        embed_phrase = embed_phrase + embed_type
        embedded_dep = self.dep_gcn(embed_phrase, dep_edges, dep_nodes['dep_tags'])
        embed_phrase_final = (embed_phrase+embedded_dep)/2
        phrase_logits = self.tag_projection_layer(embed_phrase_final)

        if cuda_device < 0:
            logits = torch.zeros([batch_size, sequence_length, phrase_logits.size(-1)], dtype=torch.float32)
        else:
            logits = torch.zeros([batch_size, sequence_length, phrase_logits.size(-1)], dtype=torch.float32, device=cuda_device)

        for i in range(0, batch_size):
            phrase_bounds = metadata[i]['phrase_bounds_tuple']
            phrase_logit = phrase_logits[i, :, :]
            for j, (x, y) in enumerate(phrase_bounds):
                if x == y:
                    logits[i, x, :] = phrase_logit[j, :]
                else:
                    for k in range(x, y+1):
                        logits[i, k, :] = phrase_logit[j, :]

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, sequence_length, self.num_classes])

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask, label_smoothing=self._label_smoothing)
            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if metadata[0]['validation']:
            output_dict = self.decode(output_dict)
            # think about how to get confidence score
            predicates_index = [x["verb_index"] for x in metadata]
            if 'sent_id' in metadata[0].keys():
                sent_ids = [x["sent_id"] for x in metadata]
                self._token_based_metric(tokens=output_dict["words"], prediction=output_dict["tags"],
                                         predicate_id=predicates_index, confidence=output_dict["tag_probs"], sent_id=sent_ids)
            else:
                self._token_based_metric(tokens=output_dict["words"], prediction=output_dict["tags"],
                                         predicate_id=predicates_index, confidence=output_dict["tag_probs"])
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        wordpiece_tags, word_tags, wordpiece_tag_probs, word_tag_probs, tag_ids = [], [], [], [], []

        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]):
            max_likelihood_sequence = torch.argmax(predictions[:length], dim=1).tolist()
            # get predicted tags from index
            tags = [self.vocab.get_token_from_index(x, namespace="labels") for x in max_likelihood_sequence]
            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])

            # get the confidence score of predicted tags
            tag_probs = [float(predictions[i][j]) for i, j in enumerate(max_likelihood_sequence)]
            wordpiece_tag_probs.append(tag_probs)
            word_tag_probs.append([tag_probs[i] for i in offsets])
            tag_ids.append([max_likelihood_sequence[i] for i in offsets])

        output_dict['wordpiece_tags'] = wordpiece_tags
        output_dict['tags'] = word_tags
        output_dict['wordpiece_tag_probs'] = wordpiece_tag_probs
        output_dict['tag_probs'] = word_tag_probs
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


class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """
    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features
        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)
        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, dep_mat, dep_labels):
        dep_label_embed = self.dep_embedding(dep_labels)
        batch_size, seq_len, feat_dim = text.shape
        val_dep = dep_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, val_dep], dim=-1)
        r = self.dep_attn(val_sum)
        p = torch.sum(r, dim=-1)
        mask = (dep_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        output = val_us + self.dep_fc(val_dep)
        output = torch.mul(p_us, output)
        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        return output_sum