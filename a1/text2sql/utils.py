import os
import numpy as np
import torch
from nltk import word_tokenize
import re
import spacy
import json
from process_sql import tokenize

# Load the spaCy English tokenizer
nlp = spacy.load('en_core_web_sm')

NUM_VALUE_TOKEN = "<num_value>"
STR_VALUE_TOKEN = "<str_value>"
VALUE_NUM_SYMBOL = "{value}"

# def sanatize_string(s):
    
#     s = s.strip().lower()
#     s = s.replace("\t", " ").replace(";", " ; ")
#     s = s.replace(",", " , ").replace("?", " ? ")
#     s = s.replace(")", " ) ").replace("(", " ( ")
#     s = s.replace("> =", ">=").replace("! =", "!=").replace("< =", "<=").replace("< >", "<>")
#     s = s.replace(">=", " >= ").replace("<=", " <= ")
#     s = s.replace(">", " > ").replace("<", " < ")
#     s = s.replace("!=", " != ").replace("=", " = ")
#     s = s.replace("<>", " <> ")

#     return s

def identify_values(s):
    # identify string or number values

    s = s.strip()
    

    regex_str2 = "\'[^\']*\'" #r'"([A-Za-z_\./\\-]*)"' 
    str2 = re.findall(regex_str2, s)
    for v in str2:
        s = s.replace(v.strip(), STR_VALUE_TOKEN)

    regex_str1 = '\"[^\"]*\"' #r"'([A-Za-z_\./\\-]*)'" 
    str1 = re.findall(regex_str1, s)
    for v in str1:
        s = s.replace(v.strip(), STR_VALUE_TOKEN)

    
    s = s.strip().split()

    regex_nums = "[-+]?\d*\.\d+"
    nums = re.findall(regex_nums, " ".join(s))
    s = [NUM_VALUE_TOKEN if tok in nums else tok for tok in s]

    regex_nums = r'\b\d+\b'
    nums = re.findall(regex_nums, " ".join(s))    
    s = [NUM_VALUE_TOKEN if tok in nums else tok for tok in s]

    return s



def tokenize_question(question):
    """WARNING: THIS IS A VERY NAIVE TOKENIZER. IMPROVE THIS LATER"""    
    # ques_tokens = word_tokenize(question)        
    # return [q.lower() for q in ques_tokens]

    """Developing sophisticated tokenizer"""
    # tokens = list()
    # question = sanatize_string(question)
    # # question = identify_values(question)

    # for tok in question.split():
    #     if "." in tok:
    #         tokens.extend(tok.replace(".", " . ").split())
    #     elif "'" in tok and tok[0]!="'" and tok[-1]!="'":
    #         tokens.extend(word_tokenize(tok))
    #     else:
    #         tokens.append(tok)

    tokens = [token.text.lower() for token in nlp(question)]
    
    return tokens



def tokenize_query(string):


    """WARNING: THIS IS A VERY NAIVE TOKENIZER. IMPROVE THIS LATER"""    
    # query_tokens = word_tokenize(query)    
    # return [q.lower() for q in query_tokens]

    # tokens = list()
    
    # query = sanatize_string(query)
    # query = identify_values(query)

    # for tok in query:
    #     if "." in tok:
    #         tokens.extend(tok.replace(".", " . ").split())
    #     else:
    #         tokens.append(tok)

    # return tokens
    """=================================================================="""
    """below functino has been taken from the provided code by DAMAN"""
    """=================================================================="""
    # string = str(string)
    # string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    # quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    # assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # # keep string value as token
    # vals = {}
    # for i in range(len(quote_idxs)-1, -1, -2):
    #     qidx1 = quote_idxs[i-1]
    #     qidx2 = quote_idxs[i]
    #     val = string[qidx1: qidx2+1]
    #     key = "__val_{}_{}__".format(qidx1, qidx2)
    #     string = string[:qidx1] + key + string[qidx2+1:]
    #     vals[key] = val

    # toks = [word.lower() for word in word_tokenize(string)]
    # # replace with string value token
    # for i in range(len(toks)):
    #     if toks[i] in vals:
    #         toks[i] = vals[toks[i]]

    # # find if there exists !=, >=, <=
    # eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    # eq_idxs.reverse()
    # prefix = ('!', '>', '<')
    # for eq_idx in eq_idxs:
    #     pre_tok = toks[eq_idx-1]
    #     if pre_tok in prefix:
    #         toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]
    toks = tokenize(string)
    tokens = []
    for tok in toks:
        if "." in tok and "t" in tok:
            tokens.extend(tok.replace(".", " . ").split())
        else:
            tokens.append(tok)
    return tokens

def load_checkpoint(args, chkpt = "best"):

    if chkpt == "best":
        model_name = os.path.join(args.checkpoint_dir, "best_loss_checkpoint_{}.pth".format(args.model_type))
        status_file = os.path.join(args.checkpoint_dir, "best_loss_chkpt_status_{}.json".format(args.model_type))
    else:
        model_name = os.path.join(args.checkpoint_dir, "latest_checkpoint_{}.pth".format(args.model_type))
        status_file = os.path.join(args.checkpoint_dir, "latest_chkpt_status_{}.json".format(args.model_type))

    assert os.path.isfile(model_name), f"Model path/name invalid: {model_name}"
    
    net = torch.load(model_name)
    with open(status_file, "r") as file:
        model_dict = json.load(file)
    print(f"\n|--------- Model Load Success. Trained Epoch: {str(model_dict['epoch'])}")

    return net

