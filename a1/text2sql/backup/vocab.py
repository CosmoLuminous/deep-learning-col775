import json
from collections import Counter, defaultdict
import pickle
from utils import *
import pandas as pd

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>", "<num_value>", "<str_value>"]

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



def generate_query_question_vocab(file_path, output_path, file_type="train", save=False):
    ques_vocab = Counter()
    query_vocab = Counter()
    
    
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

def get_all_vocab(output_path, min_freq=0, max_vocab_size=None):
    vocab = Counter()
    schema_vocab, databases, tokens_db_lookup = generate_schema_vocab("./data/tables.json")
    ques_vocab, query_vocab, max_ques, max_query = generate_query_question_vocab("./data/train.csv", "./intermediate_files/", "train", True)

    print(f"Max question len = {max_ques}, Max query len = {max_query}")

    vocab.update(schema_vocab)
    vocab.update(ques_vocab)
    vocab.update(query_vocab)
    vocab = dict(vocab.most_common())

    if min_freq > 0:
        vocab = {k: count for k, count in vocab.items() if count >= min_freq}

    if max_vocab_size is not None:
        vocab = dict(vocab.most_common(max_vocab_size))
    
    with open(os.path.join(output_path, "train.vocab"), "w") as out:
        for k, count in vocab.items():
            try:
                out.write("{}\n".format(k))
            except:
                out.write("{}\n".format(str(k).encode('utf-8')))
    print(f"VOCAB SIZE = {len(vocab)}")
    return vocab

def word_to_index(vocab, output_path):
    ctr = len(SPECIAL_TOKENS)
    word2idx = {SPECIAL_TOKENS[v]: v for v in range(len(SPECIAL_TOKENS))}
    idx2word = {v: SPECIAL_TOKENS[v] for v in range(len(SPECIAL_TOKENS))}
    
    for k, v in vocab.items():
        if k not in word2idx:
            word2idx[k] = ctr
            idx2word[ctr] = k
            ctr += 1
    assert len(word2idx) == len(idx2word)
    with open(os.path.join(output_path, "word2idx.pickle"), 'wb') as out:
        pickle.dump(word2idx, out, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_path, "idx2word.pickle"), 'wb') as out:
        pickle.dump(idx2word, out, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    vocab = get_all_vocab("./intermediate_files/", min_freq=3)
    word_to_index(vocab, "./intermediate_files/")