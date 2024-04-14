from overrides import overrides
from allennlp.training.metrics.metric import Metric
import os
from sklearn.metrics import precision_recall_fscore_support

@Metric.register("chunk_conll_match")
class Chunk_Conll_Matcher(Metric):
    """
    Computes scores according to carb framework
    """
    def __init__(self, output_path: str = None, dev_set: str = None):
        super(Chunk_Conll_Matcher, self).__init__()
        self._all_sentences = []
        self._all_tokens = []
        self._all_gold_bounds = []
        self._all_gold_types = []
        self._all_pred_bounds = []
        self._all_pred_types = []
        self._all_confidences = []
        self._all_ids = []
        self._dev_set = dev_set
        self._epoch_num = 0

        if output_path is not None and output_path is not '':
            self._output_path = output_path+'/predictions'


    @overrides
    def __call__(self, dict_file: dict, bound_tags_list: list, type_tags_list: list, sent_ids: list):
        tokens = dict_file['words']
        pred_bound_tags = dict_file['bound_tags']
        pred_type_tags = dict_file['type_tags']

        if len(tokens) == len(pred_bound_tags) == len(pred_type_tags) == len(bound_tags_list) == len(type_tags_list):
            sent_list = [' '.join(x) for x in tokens]
            self._all_sentences.extend(sent_list)
            self._all_tokens.extend(tokens)
            self._all_gold_bounds.extend(bound_tags_list)
            self._all_gold_types.extend(type_tags_list)
            self._all_pred_bounds.extend(pred_bound_tags)
            self._all_pred_types.extend(pred_type_tags)
            self._all_ids.extend(sent_ids)

        else:
            print("check prediction output")
            raise Exception


    def get_metric(self, reset: bool = False):
        if reset:
            all_pred_bounds, all_pred_types, all_gold_bounds, all_gold_types = [], [], [], []
            bound_gold_list, bound_correct_list, bound_pred_list = [], [], []

            num_adv_gold, num_adv_pred, num_noun_gold, num_noun_pred, num_verb_gold, num_verb_pred = 0, 0, 0, 0, 0, 0
            num_prep_gold, num_prep_pred, num_sbar_gold, num_sbar_pred, num_adj_gold, num_adj_pred = 0, 0, 0, 0, 0, 0

            num_O_gold, num_O_pred = 0, 0
            num_adv_correct, num_noun_correct, num_verb_correct, num_prep_correct, \
            num_sbar_correct, num_adj_correct, num_O_correct = 0, 0, 0, 0, 0, 0, 0

            num_prt_gold, num_conj_gold, num_intj_gold, num_lst_gold, num_ucp_gold = 0, 0, 0, 0, 0
            num_prt_pred, num_conj_pred, num_intj_pred, num_lst_pred, num_ucp_pred = 0, 0, 0, 0, 0
            num_prt_correct, num_conj_correct, num_intj_correct, num_lst_correct, num_ucp_correct = 0, 0, 0, 0, 0

            str_dict = {}

            for tokens, pred_bounds, pred_types, gold_bounds, gold_types, sent_id in \
                    zip(self._all_tokens, self._all_pred_bounds, self._all_pred_types, self._all_gold_bounds,
                        self._all_gold_types, self._all_ids):

                str_sent = ' '.join(tokens) + '\n'

                all_pred_bounds.extend(pred_bounds)
                all_pred_types.extend(pred_types)
                all_gold_bounds.extend(gold_bounds)
                all_gold_types.extend(gold_types)

                gold_len, pred_len = 0, 0

                str_gold_list, str_pred_list, str_miss_list, str_wrong_list = [], [], [], []
                str_gold, str_pred, str_wrong = '', '', ''

                for token, pred_bound, pred_type, gold_bound, gold_type in zip(tokens, pred_bounds, pred_types, gold_bounds, gold_types):

                    gold_type = 'O' if gold_type == 'O' else gold_type.split('-')[1]
                    pred_type = 'O' if pred_type == 'O' else pred_type.split('-')[1]


                    gold_len += 1
                    pred_len += 1
                    str_gold += token + ' '
                    str_pred += token + ' '

                    if gold_bound == 1:
                        bound_gold_list.append(gold_len)
                        str_gold_list.append(str_gold)

                        if gold_len == pred_len and pred_bound == 1:
                            bound_correct_list.append(gold_len)
                        else:
                            str_miss_list.append(str_gold)

                        if gold_type == 'O':
                            num_O_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'O': num_O_correct += 1
                        elif gold_type == 'NP':
                            num_noun_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'NP': num_noun_correct += 1
                        elif gold_type == 'VP':
                            num_verb_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'VP': num_verb_correct += 1
                        elif gold_type == 'PP':
                            num_prep_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'PP': num_prep_correct += 1
                        elif gold_type == 'SBAR':
                            num_sbar_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'SBAR': num_sbar_correct += 1
                        elif gold_type == 'ADVP':
                            num_adv_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'ADVP': num_adv_correct += 1
                        elif gold_type == 'ADJP':
                            num_adj_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'ADJP': num_adj_correct += 1
                        elif gold_type == 'PRT':
                            num_prt_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'PRT': num_prt_correct += 1
                        elif gold_type == 'CONJP':
                            num_conj_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'CONJP': num_conj_correct += 1
                        elif gold_type == 'INTJ':
                            num_intj_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'INTJ': num_intj_correct += 1
                        elif gold_type == 'LST':
                            num_lst_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'LST': num_lst_correct += 1
                        elif gold_type == 'UCP':
                            num_ucp_gold += 1
                            if gold_len == pred_len and pred_bound == 1 and pred_type == 'UCP': num_ucp_correct += 1


                    if pred_bound == 1:
                        bound_pred_list.append(pred_len)
                        str_pred_list.append(str_pred)

                        if gold_len != pred_len or gold_bound != 1:
                            str_wrong_list.append(str_pred)

                        if pred_type == 'O': num_O_pred += 1
                        elif pred_type == 'NP': num_noun_pred += 1
                        elif pred_type == 'VP': num_verb_pred += 1
                        elif pred_type == 'PP': num_prep_pred += 1
                        elif pred_type == 'SBAR': num_sbar_pred += 1
                        elif pred_type == 'ADVP': num_adv_pred += 1
                        elif pred_type == 'ADJP': num_adj_pred += 1
                        elif pred_type == 'PRT': num_prt_pred += 1
                        elif pred_type == 'CONJP': num_conj_pred += 1
                        elif pred_type == 'INTJ': num_intj_pred += 1
                        elif pred_type == 'LST': num_lst_pred += 1
                        elif pred_type == 'UCP': num_ucp_pred += 1

                        pred_len = 0
                        str_pred = ''

                    # clear the gold string and reset the gold length
                    if gold_bound == 1:
                        gold_len = 0
                        str_gold = ''

                str_sent += 'gold    phrases: ' + '   '.join(str_gold_list) + '\n'
                str_sent += 'pred    phrases: ' + '   '.join(str_pred_list) + '\n'
                str_sent += 'missing phrases: ' + '   '.join(str_miss_list) + '\n'
                str_sent += 'wrong   phrases: ' + '   '.join(str_wrong_list) + 3*'\n'

                str_dict[sent_id] = str_sent



            bound_r = len(bound_correct_list)/len(bound_gold_list)
            bound_1_r = bound_correct_list.count(1) / bound_gold_list.count(1)
            bound_2_r = bound_correct_list.count(2) / bound_gold_list.count(2)
            bound_3_r = bound_correct_list.count(3) / bound_gold_list.count(3)
            bound_4_r = bound_correct_list.count(4) / bound_gold_list.count(4)

            bound_p = len(bound_correct_list) / len(bound_pred_list)
            bound_1_p = bound_correct_list.count(1) / bound_pred_list.count(1)
            bound_2_p = bound_correct_list.count(2) / bound_pred_list.count(2)
            bound_3_p = bound_correct_list.count(3) / bound_pred_list.count(3)
            bound_4_p = bound_correct_list.count(4) / bound_pred_list.count(4)

            num_correct_5 = len(bound_correct_list) - bound_correct_list.count(1) - bound_correct_list.count(2) - bound_correct_list.count(3) - bound_correct_list.count(4)
            num_gold_5 = len(bound_gold_list) - bound_gold_list.count(1) - bound_gold_list.count(2) - bound_gold_list.count(3) - bound_gold_list.count(4)
            num_pred_5 = len(bound_pred_list) - bound_pred_list.count(1) - bound_pred_list.count(2) - bound_pred_list.count(3) - bound_pred_list.count(4)
            bound_5_r = num_correct_5 / num_gold_5
            bound_5_p = num_pred_5 / num_gold_5

            bound_1_f1 = f1(bound_1_p, bound_1_r)
            bound_2_f1 = f1(bound_2_p, bound_2_r)
            bound_3_f1 = f1(bound_3_p, bound_3_r)
            bound_4_f1 = f1(bound_4_p, bound_4_r)
            bound_5_f1 = f1(bound_5_p, bound_5_r)
            bound_f1 = f1(bound_p, bound_r)


            print('bound numbers for total, 1, 2, 3, 4, 5 are:  ', len(bound_gold_list), bound_gold_list.count(1),
                  bound_gold_list.count(2), bound_gold_list.count(3), bound_gold_list.count(4), num_gold_5)

            noun_r = num_noun_correct / num_noun_gold
            adj_r = num_adj_correct / num_adj_gold
            adv_r = num_adv_correct / num_adv_gold
            sbar_r = num_sbar_correct / num_sbar_gold
            prep_r = num_prep_correct / num_prep_gold
            verb_r = num_verb_correct / num_verb_gold
            O_r = num_O_correct / num_O_gold
            prt_r = r(num_prt_correct, num_prt_gold)
            conj_r = r(num_conj_correct, num_conj_gold)
            intj_r = r(num_intj_correct, num_intj_gold)
            lst_r = r(num_lst_correct, num_lst_gold)
            ucp_r = r(num_ucp_correct, num_ucp_gold)


            noun_p = num_noun_correct / num_noun_pred
            adj_p = p(num_adj_correct, num_adj_pred)
            adv_p = p(num_adv_correct, num_adv_pred)
            sbar_p = p(num_sbar_correct, num_sbar_pred)
            prep_p = num_prep_correct / num_prep_pred
            verb_p = num_verb_correct / num_verb_pred
            O_p = num_O_correct / num_O_pred
            prt_p = p(num_prt_correct, num_prt_pred)
            conj_p = p(num_conj_correct, num_conj_pred)
            intj_p = p(num_intj_correct, num_intj_pred)
            lst_p = p(num_lst_correct, num_lst_pred)
            ucp_p = p(num_ucp_correct, num_ucp_pred)


            noun_f1 = f1(noun_p, noun_r)
            adj_f1 = f1(adj_p, adj_r)
            adv_f1 = f1(adv_p, adv_r)
            sbar_f1 = f1(sbar_p, sbar_r)
            prep_f1 = f1(prep_p, prep_r)
            verb_f1 = f1(verb_p, verb_r)
            O_f1 = f1(O_p, O_r)
            prt_f1 = f1(prt_p, prt_r)
            conj_f1 = f1(conj_p, conj_r)
            intj_f1 = f1(intj_p, intj_r)
            lst_f1 = f1(lst_p, lst_r)
            ucp_f1 = f1(ucp_p, ucp_r)

            num_gold = num_noun_gold+num_adj_gold+num_adv_gold+num_sbar_gold+num_prep_gold+num_verb_gold+num_O_gold
            num_gold += num_prt_gold+num_conj_gold+num_intj_gold+num_lst_gold+num_ucp_gold

            num_correct = num_noun_correct+num_adj_correct+num_adv_correct+num_sbar_correct+num_prep_correct+num_verb_correct+num_O_correct
            num_correct += num_prt_correct+num_conj_correct+num_intj_correct+num_lst_correct+num_ucp_correct

            num_pred = num_noun_pred+num_adj_pred+num_adv_pred+num_sbar_pred+num_prep_pred+num_verb_pred+num_O_pred
            num_pred += num_prt_pred+num_conj_pred+num_intj_pred+num_lst_pred+num_ucp_pred

            type_r = num_correct / num_gold
            type_p = num_correct / num_pred
            type_f1 = f1(type_p, type_r)

            print('numbers for total, O, noun, verb, prep, sbar, adv, and adj  are: ', num_gold, num_O_gold,
                  num_noun_gold, num_verb_gold, num_prep_gold, num_sbar_gold, num_adv_gold, num_adj_gold)
            print('numbers for prt, conj, intj, lst, ucp are: ', num_prt_gold, num_conj_gold, num_intj_gold, num_lst_gold, num_ucp_gold)


            bound_p_micro, bound_r_micro, bound_f1_micro, _ = precision_recall_fscore_support(all_gold_bounds, all_pred_bounds, average='micro')
            bound_p_macro, bound_r_macro, bound_f1_macro, _ = precision_recall_fscore_support(all_gold_bounds, all_pred_bounds, average='macro')

            type_p_micro, type_r_micro, type_f1_micro, _ = precision_recall_fscore_support(all_gold_types, all_pred_types, average='micro')
            type_p_macro, type_r_macro, type_f1_macro, _ = precision_recall_fscore_support(all_gold_types, all_pred_types, average='macro')

            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

            str_txt = ''
            # new_id_list = sorted(str_dict.keys())
            for sent_id in sorted(str_dict.keys()):
                str_txt += str_dict[sent_id]

            output_txt_file = self._output_path + "/predictions_epoch_{}.txt".format(self._epoch_num)
            with open(output_txt_file, 'w') as f:
                f.write(str_txt)

            self._epoch_num += 1
            self.reset()
            return {
                'bound_p_micro': bound_p_micro, 'bound_r_micro': bound_r_micro, 'bound_f1_micro': bound_f1_micro,
                'type_p_micro': type_p_micro, 'type_r_micro': type_r_micro, 'type_f1_micro': type_f1_micro,


                'bound_f1': bound_f1, 'bound_p': bound_p, 'bound_r': bound_r,
                'bound_1_f1': bound_1_f1, 'bound_1_p': bound_1_p, 'bound_1_r': bound_1_r,
                'bound_2_f1': bound_2_f1, 'bound_2_p': bound_2_p, 'bound_2_r': bound_2_r,
                'bound_3_f1': bound_3_f1, 'bound_3_p': bound_3_p, 'bound_3_r': bound_3_r,
                'bound_4_f1': bound_4_f1, 'bound_4_p': bound_4_p, 'bound_4_r': bound_4_r,
                'bound_5_f1': bound_5_f1, 'bound_5_p': bound_5_p, 'bound_5_r': bound_5_r,

                'type_f1': type_f1, 'type_p': type_p, 'type_r': type_r,
                'noun_f1': noun_f1, 'noun_p': noun_p, 'noun_r': noun_r,
                'type_verb_f1': verb_f1, 'type_verb_p': verb_p, 'type_verb_r': verb_r,
                'type_prep_f1': prep_f1, 'type_prep_p': prep_p, 'type_prep_r': prep_r,
                'type_sbar_f1': sbar_f1, 'type_sbar_p': sbar_p, 'type_sbar_r': sbar_r,
                'type_adv_f1': adv_f1, 'type_adv_p': adv_p, 'type_adv_r': adv_r,
                'type_adj_f1': adj_f1, 'type_adj_p': adj_p, 'type_adj_r': adj_r,
                'type_O_f1': O_f1, 'type_O_p': O_p, 'type_O_r': O_r,
                'type_prt_f1': prt_f1, 'type_prt_p': prt_p, 'type_prt_r': prt_r,
                'type_conj_f1': conj_f1, 'type_conj_p': conj_p, 'type_conj_r': conj_r,
                'type_intj_f1': intj_f1, 'type_intj_p': intj_p, 'type_intj_r': intj_r,
                'type_lst_f1': lst_f1, 'type_lst_p': lst_p, 'type_lst_r': lst_r,
                'type_ucp_f1': ucp_f1, 'type_ucp_p': ucp_p, 'type_ucp_r': ucp_r
                    }

        else:
            return {
                'bound_p_micro': 0.0, 'bound_r_micro': 0.0, 'bound_f1_micro': 0.0,
                'type_p_micro': 0.0, 'type_r_micro': 0.0, 'type_f1_micro': 0.0,

                'bound_f1': 0.0, 'bound_p': 0.0, 'bound_r': 0.0,
                'bound_1_f1': 0.0, 'bound_1_p': 0.0, 'bound_1_r': 0.0,
                'bound_2_f1': 0.0, 'bound_2_p': 0.0, 'bound_2_r': 0.0,
                'bound_3_f1': 0.0, 'bound_3_p': 0.0, 'bound_3_r': 0.0,
                'bound_4_f1': 0.0, 'bound_4_p': 0.0, 'bound_4_r': 0.0,
                'bound_5_f1': 0.0, 'bound_5_p': 0.0, 'bound_5_r': 0.0,
                'type_f1': 0.0, 'type_p': 0.0, 'type_r': 0.0,
                'noun_f1': 0.0, 'noun_p': 0.0, 'noun_r': 0.0,
                'type_verb_f1': 0.0, 'type_verb_p': 0.0, 'type_verb_r': 0.0,
                'type_prep_f1': 0.0, 'type_prep_p': 0.0, 'type_prep_r': 0.0,
                'type_sbar_f1': 0.0, 'type_sbar_p': 0.0, 'type_sbar_r': 0.0,
                'type_adv_f1': 0.0, 'type_adv_p': 0.0, 'type_adv_r': 0.0,
                'type_adj_f1': 0.0, 'type_adj_p': 0.0, 'type_adj_r': 0.0,
                'type_O_f1': 0.0, 'type_O_p': 0.0, 'type_O_r': 0.0,
                'type_prt_f1': 0.0, 'type_prt_p': 0.0, 'type_prt_r': 0.0,
                'type_conj_f1': 0.0, 'type_conj_p': 0.0, 'type_conj_r': 0.0,
                'type_intj_f1': 0.0, 'type_intj_p': 0.0, 'type_intj_r': 0.0,
                'type_lst_f1': 0.0, 'type_lst_p': 0.0, 'type_lst_r': 0.0,
                'type_ucp_f1': 0.0, 'type_ucp_p': 0.0, 'type_ucp_r': 0.0
            }

    @overrides
    def reset(self):
        self._all_sentences = []
        self._all_tokens = []
        self._all_gold_bounds = []
        self._all_gold_types = []
        self._all_pred_bounds = []
        self._all_pred_types = []
        self._all_confidences = []
        self._all_ids = []

def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)

def p(correct, pred):
    if pred == 0:
        return 0
    return correct / pred

def r(correct, gold):
    if gold == 0:
        return 0
    return correct / gold