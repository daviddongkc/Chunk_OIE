import json
from oie_project import bert_utils
import regex as re

def bert_replace(seq_tokens):
    new_seq_tokens = []
    for st in seq_tokens:
        new_st = []
        for t in st:
            if t in bert_utils.unk_mapping:
                new_st.append(bert_utils.unk_mapping[t])
            else:
                if (t[:2] == '##'):
                    if len(new_st) != 0:
                        new_st[-1] = new_st[-1] + t[2:]
                    else:
                        new_st.append(t[2:])  # Will never be correct but can happend
                else:
                    new_st.append(t)
        new_seq_tokens.append(new_st)
    return new_seq_tokens


def find_tuple_args(extraction, type_str):
    type_start = "<{}>".format(type_str)
    type_end = "</{}>".format(type_str)

    try:
        arg = extraction[extraction.index(type_start) + len(type_start): extraction.index(type_end)]
    except:
        arg = ''

    return arg.strip()


def convert_extraction(extraction):
    extraction_str = ''

    rel = find_tuple_args(extraction, "rel")
    arg1 = find_tuple_args(extraction, "arg1")
    arg2 = find_tuple_args(extraction, "arg2")
    arg3 = find_tuple_args(extraction, "arg3")
    arg4 = find_tuple_args(extraction, "arg4")

    if rel != '':
        extraction_str = rel
    if arg1 != '':
        extraction_str += "\t" + arg1
    if arg2 != '':
        extraction_str += "\t" + arg2
    if arg3 != '':
        extraction_str += "\t" + arg3
    if arg4 != '':
        extraction_str += "\t" + arg4
    return extraction_str


def process_single(input_lines, threshold=None):
    # input_lines in the list of json format
    out_lines = []
    for i, json_line in enumerate(input_lines):
        seq_tokens = json_line['predicted_tokens']
        scores = json_line['predicted_log_probs']
        seq_sent = json_line['sentence']

        seq_tokens = [" ".join(st) for st in seq_tokens]
        seq_tokens = [re.sub(r'\[\ unused\ ##(\d+)\ \]', r'[unused\1]', st) for st in seq_tokens]
        seq_tokens = [st.split(' ') for st in seq_tokens]
        seq_tokens = bert_replace(seq_tokens)

        for j in range(len(scores)):
            if(type(threshold) != type(None) and scores[j] < threshold):
                continue
            extraction = " ".join(seq_tokens[j])
            extraction = convert_extraction(extraction.strip())
            if len(extraction.split('\t')) > 1:
                out_line = seq_sent + '\t' + str(scores[j]) + '\t' + extraction
                out_lines.append(out_line)

    return "\n".join(out_lines)


def process_append(input_lines):
    # input_lines in the list of json format
    out_lines = []
    for i, json_line in enumerate(input_lines):
        seq_tokens = json_line['predicted_tokens'][0]  # Consider only the best beam
        seq_probs = json_line['predicted_log_probs']  # Contains score of only the best beam
        seq_sent = json_line['sentence']

        seq_tokens = bert_replace([seq_tokens])[0]

        extraction_scores, extractions, extraction = [], [], ''
        extraction_num = 0

        for token_index, token in enumerate(seq_tokens):
            if (token == '[SEP]' and len(extraction.split()) != 0):
                # extraction_scores.append(seq_probs[extraction_num] / len(extraction.split()))
                extraction_scores.append(seq_probs[extraction_num])
                extraction_num += 1
                extractions.append(convert_extraction(extraction.strip()))
                extraction = ''
                continue
            extraction += token + ' '

        dedup_extractions, dedup_scores = [], []
        for extraction_num in range(len(extractions)):
            extraction = extractions[extraction_num]
            score = extraction_scores[extraction_num]
            if extraction not in dedup_extractions:
                dedup_extractions.append(extraction)
                dedup_scores.append(score)

        extractions, extraction_scores = dedup_extractions, dedup_scores

        for extraction_num, extraction in enumerate(extractions):
            if len(extraction.split('\t')) > 1:
                out_line = seq_sent + '\t' + str(extraction_scores[extraction_num]) + '\t' + extraction
                out_lines.append(out_line)

    return '\n'.join(out_lines)