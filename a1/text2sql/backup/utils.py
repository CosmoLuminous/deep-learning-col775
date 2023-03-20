import os
import numpy as np
import torch
from nltk import word_tokenize
import re

NUM_VALUE_TOKEN = "<num_value>"
STR_VALUE_TOKEN = "<str_value>"
VALUE_NUM_SYMBOL = "{value}"

def sanatize_string(s):
    
    s = s.strip().lower()
    s = s.replace("\t", " ").replace(";", " ; ")
    s = s.replace(",", " , ").replace("?", " ? ")
    s = s.replace(")", " ) ").replace("(", " ( ")
    s = s.replace("> =", ">=").replace("! =", "!=").replace("< =", "<=").replace("< >", "<>")
    s = s.replace(">=", " >= ").replace("<=", " <= ")
    s = s.replace(">", " > ").replace("<", " < ")
    s = s.replace("!=", " != ").replace("=", " = ")
    s = s.replace("<>", " <> ")

    return s

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
    tokens = list()
    question = sanatize_string(question)
    question = identify_values(question)

    for tok in question:
        if "." in tok:
            tokens.extend(tok.replace(".", " . ").split())
        elif "'" in tok and tok[0]!="'" and tok[-1]!="'":
            tokens.extend(word_tokenize(tok))
        else:
            tokens.append(tok)
    
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
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]
    
    tokens = []
    for tok in toks:
        if "." in tok and "t" in tok:
            tokens.extend(tok.replace(".", " . ").split())
        else:
            tokens.append(tok)
    return tokens


# def tokenize_question(nl):
#     '''
#     return keywords of nl query
#     '''
#     nl_keywords = []
#     nl = nl.strip()
#     nl = nl.replace(";"," ; ").replace(",", " , ").replace("?", " ? ").replace("\t"," ")
#     nl = nl.replace("(", " ( ").replace(")", " ) ")
    
#     str_1 = re.findall("\"[^\"]*\"", nl)
#     str_2 = re.findall("\'[^\']*\'", nl)
#     float_nums = re.findall("[-+]?\d*\.\d+", nl)
    
#     values = str_1 + str_2 + float_nums
#     for val in values:
#         nl = nl.replace(val.strip(), VALUE_NUM_SYMBOL)
    
    
#     raw_keywords = nl.strip().split()
#     for tok in raw_keywords:
#         if "." in tok:
#             to = tok.replace(".", " . ").split()
#             to = [t.lower() for t in to if len(t)>0]
#             nl_keywords.extend(to)
#         elif "'" in tok and tok[0]!="'" and tok[-1]!="'":
#             to = word_tokenize(tok)
#             to = [t.lower() for t in to if len(t)>0]
#             nl_keywords.extend(to)      
#         elif len(tok) > 0:
#             nl_keywords.append(tok.lower())
#     return nl_keywords


# def tokenize_query(query):
#     '''
#     return keywords of sql query
#     '''
#     query_keywords = []
#     query = query.strip().replace(";","").replace("\t","")
#     query = query.replace("(", " ( ").replace(")", " ) ")
#     query = query.replace(">=", " >= ").replace("<=", " <= ").replace("!=", " != ").replace("=", " = ")

    
#     # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
#     str_1 = re.findall("\"[^\"]*\"", query)
#     str_2 = re.findall("\'[^\']*\'", query)
    
#     values = str_1 + str_2
#     for val in values:
#         query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

#     query_tokenized = query.split()
#     float_nums = re.findall("[-+]?\d*\.\d+", query)
#     query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]
#     query = " ".join(query_tokenized)
#     int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    
#     query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
#     # print int_nums, query, query_tokenized
    
#     for tok in query_tokenized:
#         if "." in tok:
#             table = re.findall("[Tt]\d+\.", tok)
#             if len(table)>0:
#                 to = tok.replace(".", " . ").split()
#                 to = [t.lower() for t in to if len(t)>0]
#                 query_keywords.extend(to)
#             else:
#                 query_keywords.append(tok.lower())

#         elif len(tok) > 0:
#             query_keywords.append(tok.lower())
#     query_keywords = [w for w in query_keywords if len(w)>0]
#     query_sentence = " ".join(query_keywords)
#     query_sentence = query_sentence.replace("> =", ">=").replace("! =", "!=").replace("< =", "<=")
# #     if '>' in query_sentence or '=' in query_sentence:
# #        print query_sentence
#     return query_sentence.split()