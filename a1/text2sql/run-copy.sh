python3 train.py --model_type lstm_lstm_attn --batch_size 32 --epochs 200 --en_num_layers 1 --de_num_layers 1 --en_hidden 512 --de_hidden 512 --search_type beam --lr 0.001 2>&1 | tee logs/lstm_lstm_attn_0.0001_1_1_512_512_300_32_200.log && \
python3 train.py --batch_size 32 --epochs 200 --en_num_layers 1 --de_num_layers 1 --en_hidden 512 --de_hidden 512 --search_type beam 2>&1 --lr 0.001 | tee logs/lstm_lstm_0.0001_1_1_512_512_300_32_200.log

# python3 train.py --model_type bert_lstm_attn_tuned --en_hidden 768 --de_hidden 768 --batch_size 32 --search_type beam --bert_tune_layers 4 2>&1 | tee logs/bert_lstm_attn_tuned4_1_1_786_786_300_32_200.log && \
# python3 train.py --model_type bert_lstm_attn_tuned --en_hidden 768 --de_hidden 768 --batch_size 32 --search_type beam --bert_tune_layers 8 2>&1 | tee logs/bert_lstm_attn_tuned8_1_1_786_786_300_32_200.log
# python3 train.py --model_type bert_lstm_attn_tuned --en_hidden 768 --de_hidden 768 --batch_size 32 --search_type beam 2>&1 | tee logs/bert_lstm_attn_tuned_1_1_786_786_300_32_200.log
# python3 train.py --model_type bert_lstm_attn_frozen --en_hidden 768 --de_hidden 768 --batch_size 32 --search_type beam 2>&1 | tee logs/bert_lstm_attn_frozen_1_1_786_786_300_32_200.log

# python3 train.py --model_type lstm_lstm_attn --batch_size 32 --search_type beam 2>&1 | tee logs/lstm_lstm_attn_1_1_512_512_300_32_200.log
# python3 train.py --batch_size 32 2>&1 | tee logs/s2s_test.log
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 1  2>&1 | tee logs/seq2seq_2_1_512_512_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 1 --de_num_layers 2  2>&1 | tee logs/seq2seq_1_2_512_512_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 2  2>&1 | tee logs/seq2seq_2_2_512_512_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 2 --en_hidden 1024 2>&1 | tee logs/seq2seq_2_2_1024_512_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 2 --de_hidden 1024 2>&1 | tee logs/seq2seq_2_2_512_1024_200_32_300.log
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 2 --en_hidden 1024 --de_hidden 1024 2>&1 | tee logs/seq2seq_2_2_1024_1024_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 4 --de_num_layers 4 --en_hidden 512 --de_hidden 512 2>&1 | tee logs/seq2seq_4_4_512_512_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 4 --de_num_layers 4 --en_hidden 1024 --de_hidden 1024 2>&1 | tee logs/seq2seq_4_4_1024_1024_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 2 --de_num_layers 2 --en_hidden 256 --de_hidden 256 2>&1 | tee logs/seq2seq_2_2_256_256_200_32_300.log && \
# python3 train.py --batch_size 32 --epochs 200 --en_num_layers 1 --de_num_layers 1 --en_hidden 256 --de_hidden 256 2>&1 | tee logs/seq2seq_1_1_256_256_200_32_300.log