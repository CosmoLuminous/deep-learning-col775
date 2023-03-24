python3 train.py --batch_size 3 --search_type beam 2>&1 | tee logs/s2s_beam_exp.log
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