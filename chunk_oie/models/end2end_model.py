from typing import Dict, List, Optional, Any, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout, Embedding, LogSigmoid, LogSoftmax
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import os
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.modules import Seq2SeqEncoder
from allennlp.training.metrics import Metric
from chunk_oie.models.modules import DepGCN, convert_to_adj_tensor

@Model.register("e2e_model")
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
                 tuple_metric: Metric = None,
                 dep_label_file: str = "config/dependency_labels.txt") -> None:
        super(OIE_Model, self).__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("labels")
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)
        self.num_type_labels = self.vocab.get_vocab_size("chunk_labels")

        # this is to provide dependency labels to the model
        lines = open(dep_label_file, "r").readlines()
        self.dep_label = {line.strip(): i + 1 for i, line in enumerate(lines)}
        self.num_dep_labels = len(self.dep_label)+1
        # self.num_dep_labels = 47
        self.dep_gcn = DepGCN(self.num_dep_labels, dependency_label_dim, self.bert_model.config.hidden_size,
                              self.bert_model.config.hidden_size)


        # this is provide pos tags to the model
        self.num_pos_labels = self.vocab.get_vocab_size("pos_labels")
        self.pos_embedding = Embedding(self.num_pos_labels, self.bert_model.config.hidden_size, padding_idx=0)

        # this is to check number of classes for boundary and type
        self.num_bound_classes = 2
        self.num_type_classes = self.vocab.get_vocab_size("chunk_labels")

        # this is to project the output of bert to the number of classes
        self.type_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_type_classes)
        self.bound_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_bound_classes)


        self.type_embedding = Embedding(self.num_type_labels, type_label_dim, padding_idx=0)
        # self.dep_embedding = Embedding(self.num_dep_labels, dependency_label_dim, padding_idx=0)
        # self.dep_attn = Linear(dependency_label_dim + self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)

        # self.phrase_attention = phrase_attention

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self._token_based_metric = tuple_metric
        initializer(self)

    def chunk_layer(self, pos_tags: torch.Tensor, bert_embeddings: torch.Tensor,
                    mask: torch.Tensor, offsets, offsets_end):
        batch_size, sequence_length, _ = bert_embeddings.size()
        embed_pos = self.pos_embedding(pos_tags['pos_tags'])
        embedded_text_input = bert_embeddings + embed_pos

        bound_logits = self.bound_projection_layer(embedded_text_input)
        reshaped_bound_log_probs = bound_logits.view(-1, self.num_bound_classes)
        bound_probabilities = F.softmax(reshaped_bound_log_probs, dim=-1).view([batch_size, sequence_length, self.num_bound_classes])

        # use only the boundary token as input to project layer!
        type_logits = self.type_projection_layer(embedded_text_input)
        reshaped_tag_log_probs = type_logits.view(-1, self.num_type_classes)
        type_probabilities = F.softmax(reshaped_tag_log_probs, dim=-1).view([batch_size, sequence_length, self.num_type_classes])

        output_dict = {"bound_logits": bound_logits, "bound_probabilities": bound_probabilities,
                      "type_logits": type_logits, "type_probabilities": type_probabilities,
                      "mask": mask, "wordpiece_offsets": offsets, "wordpiece_offsets_end": offsets_end}

        output_dict = self.chunk_decode(output_dict)

        return output_dict

    def oie_layer(self, bert_embeddings: torch.Tensor, mask: torch.Tensor, boundary_predictions, type_predictions,
                  metadata):

        batch_size, sequence_length, embed_dim = bert_embeddings.size()
        cuda_device = bert_embeddings.get_device()

        # this is to covert the wordpiece boundaries to phrase boundaries
        phrase_types_batch, phrase_bounds_batch, max_phrase_len = [], [], 0
        for i, (bounds, types) in enumerate(zip(boundary_predictions, type_predictions)):
            phrase_types, phrase_bounds = [], []
            start_index = 0
            for j, (bound, type) in enumerate(zip(bounds, types)):
                if bound == 1:
                    phrase_bounds.append((start_index, j))
                    phrase_types.append(type)
                    start_index = j + 1
            max_phrase_len = max(max_phrase_len, len(phrase_types))
            phrase_bounds_batch.append(phrase_bounds)
            phrase_types_batch.append(phrase_types)

        # by having chunk boundaries, we can get the phrase embeddings
        if cuda_device < 0:
            embed_phrase = torch.zeros([batch_size, max_phrase_len, embed_dim], dtype=torch.float32)
        else:
            embed_phrase = torch.zeros([batch_size, max_phrase_len, embed_dim], dtype=torch.float32, device=cuda_device)

        for i in range(0, batch_size):
            phrase_bounds = phrase_bounds_batch[i]
            embed_bert = bert_embeddings[i, :, :]
            for j, (x, y) in enumerate(phrase_bounds):
                if x == y:
                    embed_phrase[i, j, :] = embed_bert[x, :]
                else:
                    embed_phrase[i, j, :] = torch.mean(embed_bert[x:y+1, :], dim=0)

        # by having chunked phrases, we can leverage on the chunk type information
        pad_value = 0
        type_dict = self.vocab.get_index_to_token_vocabulary(namespace="chunk_labels")
        for key, val in type_dict.items():
            if val == "O" or val == "Nil":
                pad_value = key
                break

        if cuda_device < 0:
            padded_phrase_types = pad_sequence([torch.tensor(x) for x in phrase_types_batch],
                                           batch_first=True, padding_value=pad_value)
        else:
            padded_phrase_types = pad_sequence([torch.tensor(x, device=cuda_device) for x in phrase_types_batch],
                                           batch_first=True, padding_value=pad_value)
        embed_type = self.type_embedding(padded_phrase_types)

        # phrase embedding is the sum of bert phrase embedding and chunk type embedding
        embed_phrase = embed_phrase + embed_type

        # dependency nodes and edges are in word level.  boundary predictions are in word-pieces level.
        # we need to convert the dependency nodes and edges to phrase level
        adj_tensor, node_tensor = self.convert_phrase_dep(phrase_bounds_batch, metadata, device=cuda_device)
        # use dep gcn to encode graph information
        phrase_dep = self.dep_gcn(embed_phrase, adj_tensor, node_tensor)

        embed_phrase = 0.5 * phrase_dep + 0.5 * embed_phrase

        # project the phrase embedding to the number of classes
        phrase_logits = self.tag_projection_layer(embed_phrase)

        # note that the phrase_logits is in phrase level. We need to convert it back to wordpiece level
        if cuda_device < 0:
            logits = torch.zeros([batch_size, sequence_length, phrase_logits.size(-1)], dtype=torch.float32)
        else:
            logits = torch.zeros([batch_size, sequence_length, phrase_logits.size(-1)], dtype=torch.float32, device=cuda_device)
        for i in range(0, batch_size):
            phrase_bounds = phrase_bounds_batch[i]
            phrase_logit = phrase_logits[i, :, :]
            for j, (x, y) in enumerate(phrase_bounds):
                if x == y:
                    logits[i, x, :] = phrase_logit[j, :]
                else:
                    for k in range(x, y+1):
                        logits[i, k, :] = phrase_logit[j, :]
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size, sequence_length, self.num_classes])


        return logits, class_probabilities


    def phrase_wordpc_align(self, phrase_bounds, dep_edges, dep_nodes, offsets_start, offsets_end):
        dep_phrase_edges = []
        for i, dep in enumerate(dep_edges):
            dep_i, dep_j = offsets_start[dep[0][0]], offsets_start[dep[0][1]]
            dep_i_match, dep_j_match = -1, -1

            for j, (bound_i, bound_j) in enumerate(phrase_bounds):
                if dep_i <= bound_j and dep_i >= bound_i:
                    # dep node i is inside phrase j
                    dep_i_match = j
                if dep_j <= bound_j and dep_j >= bound_i:
                    # dep node j is inside phrase j
                    dep_j_match = j

                # break from the loop if both token i and j are matched with the corresponding nodes.
                if dep_i_match != -1 and dep_j_match != -1:
                    dep_phrase_edges.append(((dep_i_match, dep_j_match), dep[1]))
                    break

        dep_phrase_edges_final = [(x, j) for x, j in dep_phrase_edges if x[0] != x[1] or j == 'ROOT']

        dep_phrase_edges_final.insert(0, ((0, 0), "CLS"))
        dep_phrase_edges_final.append(((len(phrase_bounds) - 1, len(phrase_bounds) - 1), "SEP"))

        dep_phrase_list = []
        for (_, i), dep_type in dep_phrase_edges_final:
            if i == len(dep_phrase_list):
                dep_phrase_list.append(self.dep_label[dep_type])

        if len(dep_phrase_list) != len(phrase_bounds):
            print("dep phrase and bounds not align", len(dep_phrase_list), len(phrase_bounds))

        dep_edges_tuple = [(i, j) for (i, j), _ in dep_phrase_edges_final]
        dep_edges_tuple = list(set(dep_edges_tuple))

        return dep_edges_tuple, dep_phrase_list
    def convert_phrase_dep(self, phrase_bounds, metadata, device = 0):
        dep_edges_list, dep_nodes_list = [x["dep_edges"] for x in metadata], [x["dep_nodes"] for x in metadata]
        offsets_start, offsets_end = [x["offsets"] for x in metadata], [x["offsets_end"] for x in metadata]

        # convert word-level dep graph into phase-level ones
        dep_edges_batch, dep_nodes_batch = [], []
        for i in range(len(phrase_bounds)):
            dep_edges, dep_nodes = self.phrase_wordpc_align(phrase_bounds[i], dep_edges_list[i], dep_nodes_list[i], offsets_start[i], offsets_end[i])
            dep_edges_batch.append(dep_edges)
            dep_nodes_batch.append(dep_nodes)

        # covert the batached dep edges and nodes into adjacency matrix
        adj_tensor, node_tensor = convert_to_adj_tensor(dep_edges_batch, dep_nodes_batch, cuda_device=device)

        return adj_tensor, node_tensor


    def forward(self,
                tokens: Dict[str, torch.Tensor],
                verb_indicator: torch.Tensor,
                pos_tags: torch.Tensor,
                metadata: List[Any],
                tags: torch.LongTensor=None,
                bound_tags: torch.LongTensor=None,
                type_tags: torch.LongTensor=None,
                optimizer=None):

        output_dict = {}

        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets, offsets_end = zip(*[(x["words"], x["verb"], x["offsets"], x["offsets_end"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)
        output_dict["wordpiece_offsets_end"] = list(offsets_end)

        mask = get_text_field_mask(tokens)
        # We need to retain the mask in the output dictionary so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        bert_embeddings, _ = self.bert_model(input_ids=tokens["tokens"],
                                             token_type_ids=verb_indicator,
                                             attention_mask=mask,
                                             output_all_encoded_layers=False)

        embedded_text_input = self.embedding_dropout(bert_embeddings)

        # use chunk layer to obtain chunk boundary and type
        chunk_outputs = self.chunk_layer(pos_tags, embedded_text_input, mask, output_dict["wordpiece_offsets"], output_dict["wordpiece_offsets_end"])

        # after getting the chunk boundary and type, we use the chunk boundary to get the phrase embedding
        logits, class_probabilities = self.oie_layer(embedded_text_input, mask, chunk_outputs['wordpiece_bound_tags'],
                                                     chunk_outputs['wordpiece_type_ints'], metadata)
        output_dict["logits"] = logits
        output_dict["class_probabilities"] = class_probabilities

        if not metadata[0]['validation']:
            bound_loss = sequence_cross_entropy_with_logits(chunk_outputs['bound_logits'], bound_tags, mask, label_smoothing=self._label_smoothing)
            type_loss = sequence_cross_entropy_with_logits(chunk_outputs['type_logits'], type_tags, mask, label_smoothing=self._label_smoothing)
            chunk_loss = bound_loss + 0.8 * type_loss
            oie_loss = sequence_cross_entropy_with_logits(logits, tags, mask, label_smoothing=self._label_smoothing)
            output_dict["loss"] = oie_loss + chunk_loss
        else:
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


    def chunk_decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        wordpiece_type_ints = []
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for bound_predictions, type_predictions, length, offsets, offsets_end \
                in zip(bound_predictions_list, type_predictions_list, sequence_lengths,
                       output_dict["wordpiece_offsets"], output_dict["wordpiece_offsets_end"]):

            bound_tags = torch.argmax(bound_predictions[:length], dim=1).tolist()

            bound_tags_new = []
            for i, bound in enumerate(bound_tags):
                if i in offsets_end:
                    bound_tags_new.append(bound)
                elif i == 0 or i == length - 1:
                    bound_tags_new.append(1)
                else:
                    bound_tags_new.append(0)

            wordpiece_bound_tags.append(bound_tags_new)
            word_bound_tags.append([bound_tags_new[i] for i in offsets])

            type_ints, _ = viterbi_decode(type_predictions[:length], transition_matrix,
                                          allowed_start_transitions=start_transitions)
            # type_ints = torch.argmax(type_predictions[:length], dim=1).tolist()
            type_tags = [self.vocab.get_token_from_index(x, namespace="chunk_labels") for x in type_ints]
            wordpiece_type_ints.append(type_ints)
            wordpiece_type_tags.append(type_tags)
            word_type_tags.append([type_tags[i] for i in offsets])

            # get the confidence score of predicted tags
            bound_probs = [float(bound_predictions[i][j]) for i, j in enumerate(bound_tags_new)]
            word_bound_probs.append([bound_probs[i] for i in offsets])
            wordpiece_bound_probs.append(bound_probs)

            type_probs = [float(type_predictions[i][j]) for i, j in enumerate(type_ints)]
            word_type_probs.append([type_probs[i] for i in offsets])
            wordpiece_type_probs.append(type_probs)

            # tag_ids.append([tags[i] for i in offsets])

        output_dict['wordpiece_bound_tags'] = wordpiece_bound_tags
        output_dict['bound_tags'] = word_bound_tags
        output_dict['wordpiece_type_tags'] = wordpiece_type_tags
        output_dict['wordpiece_type_ints'] = wordpiece_type_ints
        output_dict['type_tags'] = word_type_tags
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
        all_labels = self.vocab.get_index_to_token_vocabulary("chunk_labels")
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
        all_labels = self.vocab.get_index_to_token_vocabulary("chunk_labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions


# class DepGCN(nn.Module):
#     """
#     Label-aware Dependency Convolutional Neural Network Layer
#     """
#     def __init__(self, dep_num, dep_dim, in_features, out_features):
#         super(DepGCN, self).__init__()
#         self.dep_dim = dep_dim
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)
#         self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
#         self.dep_fc = nn.Linear(dep_dim, out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, text, dep_mat, dep_labels):
#         dep_label_embed = self.dep_embedding(dep_labels)
#         batch_size, seq_len, feat_dim = text.shape
#         val_dep = dep_label_embed.unsqueeze(dim=2)
#         val_dep = val_dep.repeat(1, 1, seq_len, 1)
#         val_us = text.unsqueeze(dim=2)
#         val_us = val_us.repeat(1, 1, seq_len, 1)
#         val_sum = torch.cat([val_us, val_dep], dim=-1)
#         r = self.dep_attn(val_sum)
#         p = torch.sum(r, dim=-1)
#         mask = (dep_mat == 0).float() * (-1e30)
#         p = p + mask
#         p = torch.softmax(p, dim=2)
#         p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
#         output = val_us + self.dep_fc(val_dep)
#         output = torch.mul(p_us, output)
#         output_sum = torch.sum(output, dim=2)
#         output_sum = self.relu(output_sum)
#         return output_sum
