import json
from collections import Counter, defaultdict
import pickle
from utils import *
import pandas as pd

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>", "<num_value>", "<str_value>"]
SQL_KEYWORDS = ["t"+str(i+1) for i in range(10)] + [".", ",", "(", ")", "in", "not", "and", "between", "or", "where"] + ["except", "union", "intersect",
            "group", "by", "order", "limit", "having","asc", "desc"] + ["count", "sum", "avg", "max", "min",
           "<", ">", "=", "!=", ">=", "<="] + ["like", "distinct", "*", "join", "on", "as", "select", "from"]

SQL_KEYWORDS = dict(zip(SQL_KEYWORDS, [10]*len(SQL_KEYWORDS)))

def generate_schema_vocab(file_path):
    
    with open(file_path, "r") as f:
        schemas = json.load(f)
        
    databases = set()
    tokens_db_lookup = defaultdict(set)
    schema_vocab = Counter()
    
    for schema in schemas:
        db_id = schema["db_id"]
        schema_tokens = []
        
        if db_id not in databases:
            databases.add(db_id)
        
        for column in schema["column_names_original"]:
            schema_tokens.append(column[1].lower())
        
        for table in schema["table_names_original"]:
            schema_tokens.append(table.lower())
        
        for token in list(Counter(schema_tokens).keys()):
            tokens_db_lookup[token].add(db_id)
        
        schema_vocab.update(schema_tokens)    
    
    return schema_vocab, databases, tokens_db_lookup


def generate_sql_vocab(file_path):
    
    query_vocab = Counter()
    max_query = -1
    data_points = []

    
    data_file = pd.read_csv(file_path)

    for idx, dp in data_file.iterrows():
        query = dp["query"]
        query_tokens = tokenize_query(query)
        
        max_query = max(max_query, len(query_tokens))
        query_vocab.update(query_tokens)

    return query_vocab, max_query

def generate_question_vocab(file_path):

    data_file = pd.read_csv(file_path)

    ques_vocab = Counter()
    max_ques = -1

    for idx, dp in data_file.iterrows():
        
        question = dp["question"]
        ques_tokens = tokenize_question(question)

        max_query = max(max_ques, len(ques_tokens))
        ques_vocab.update(ques_tokens)

    return ques_vocab, max_ques

def word_to_index(vocab):
    ctr = len(SPECIAL_TOKENS)
    word2idx = {SPECIAL_TOKENS[v]: v for v in range(len(SPECIAL_TOKENS))}
    idx2word = {v: SPECIAL_TOKENS[v] for v in range(len(SPECIAL_TOKENS))}
    
    for k, v in vocab.items():
        if k not in word2idx:
            word2idx[k] = ctr
            idx2word[ctr] = k
            ctr += 1
    assert len(word2idx) == len(idx2word)
    return word2idx, idx2word

def generate_encoder_decoder_vocab(train_path, tables_path, output_path, max_en_vocab=10000, max_de_vocab=10000, min_en_freq=2, min_de_freq=2, save=True):

    encoder_vocab = Counter()
    decoder_vocab = Counter()

    ques_vocab, max_ques = generate_question_vocab(train_path)
    query_vocab, max_query = generate_sql_vocab(train_path)
    schema_vocab, databases, tokens_db_lookup = generate_schema_vocab(tables_path)

    print("Max tokens in question = {}, Max tokens in query = {}".format(max_ques, max_query))

    encoder_vocab.update(ques_vocab)
    decoder_vocab.update(query_vocab)
    decoder_vocab.update(schema_vocab)
    decoder_vocab.update(SQL_KEYWORDS)

    encoder_vocab = dict(encoder_vocab.most_common())

    if min_en_freq > 0:
        encoder_vocab = {k: count for k, count in encoder_vocab.items() if count >= min_en_freq}

    if max_en_vocab < len(encoder_vocab):
        encoder_vocab = dict(encoder_vocab.most_common(max_vocab_size))

    decoder_vocab = dict(decoder_vocab.most_common())

    if min_de_freq > 0:
        decoder_vocab = {k: count for k, count in decoder_vocab.items() if count >= min_de_freq}

    if max_de_vocab < len(decoder_vocab):
        decoder_vocab = dict(decoder_vocab.most_common(max_de_vocab))

    with open(os.path.join(output_path, "encoder.vocab"), "w") as out:
        for k, count in encoder_vocab.items():
            try:
                out.write("{}\n".format(k))
            except:
                out.write("{}\n".format(str(k).encode('utf-8')))
    
    with open(os.path.join(output_path, "decoder.vocab"), "w") as out:
        for k, count in decoder_vocab.items():
            try:
                out.write("{}\n".format(k))
            except:
                out.write("{}\n".format(str(k).encode('utf-8')))

    en_word2idx, en_idx2word = word_to_index(encoder_vocab)
    de_word2idx, de_idx2word = word_to_index(decoder_vocab)
    
    with open(os.path.join(output_path, "encoder_word2idx.pickle"), 'wb') as out:
        pickle.dump(en_word2idx, out, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "encoder_idx2word.pickle"), 'wb') as out:
        pickle.dump(en_idx2word, out, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "decoder_word2idx.pickle"), 'wb') as out:
        pickle.dump(de_word2idx, out, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "decoder_idx2word.pickle"), 'wb') as out:
        pickle.dump(de_idx2word, out, protocol=pickle.HIGHEST_PROTOCOL)

    print("Encoder Vocab Size = {}, Decoder Vocab Size = {}".format(len(en_word2idx), len(de_word2idx)))


    return

def process_data(file_path, output_path, prefix = 'train'):
    
    data_points = []    
    data_file = pd.read_csv(file_path)

    for idx, dp in data_file.iterrows():
        question = dp["question"]
        ques_tokens = " ".join(tokenize_question(question))
        query = dp["query"]
        query_tokens = " ".join(tokenize_query(query))
        db_id = dp['db_id']

        data_points.append([db_id, ques_tokens, query_tokens, query.lower().replace('\t', ' ')])
    
    df = pd.DataFrame(data_points, columns=["db_id", "question", "query", "orig_query"])
    file_name = os.path.join(output_path, "{}_data.xlsx".format(prefix))
    df.to_excel(file_name, index=False)
    return



def generate_query_question_vocab(file_path, output_path, file_type="train", save=False):
    ques_vocab = Counter()
    query_vocab = Counter()
    
    max_query = -1
    data_points = []
    
#     with open(file_path, "r") as f:
#         data_file = json.load(f)
    max_ques = -1
    max_query = -1
    data_file = pd.read_csv(file_path)
    
    # if save:
    #     ques_outfile = open(os.path.join(output_path, f"{file_type}_questions.txt"), "w")
    #     query_outfile = open(os.path.join(output_path, f"{file_type}_query.txt"), "w")
    #     query_db_outfile = open(os.path.join(output_path, f"{file_type}_query_db.txt"), "w")
        
    data_points = []
    for idx, dp in data_file.iterrows():
        question = dp["question"]
        query = dp["query"]
        db_id = dp["db_id"]
        ques_tokens = tokenize_question(question)
        query_tokens = tokenize_query(query)

        max_ques = max(max_ques, len(ques_tokens))
        max_query = max(max_query, len(query_tokens))
        
        ques_vocab.update(ques_tokens)
        query_vocab.update(query_tokens)
        
        ques_sentence = " ".join(ques_tokens)
        query_sentence = " ".join(query_tokens)
        
        data_points.append([db_id, ques_sentence, query_sentence, query.lower().replace('\t', ' ')])

    if save:
        df = pd.DataFrame(data_points, columns=["db_id", "question", "query", "orig_query"])
        file_name = os.path.join(output_path, "{}_data.xlsx".format(file_type))
        df.to_excel(file_name, index=False)
    
    return ques_vocab, query_vocab, max_ques, max_query





if __name__ == "__main__":

    generate_encoder_decoder_vocab("./data/train.csv", "./data/tables.json", "./processed_data/", max_en_vocab=10000, max_de_vocab=10000, min_en_freq=3, min_de_freq=3, save=True)
    print("processing training data...")
    process_data("./data/train.csv", "./processed_data/", "train")
    print("processing val data...")
    process_data("./data/val.csv", "./processed_data/", "val")