python3 infer.py --model_file models/lstm_lstm.pth --model_type lstm_lstm --test_data_file sql_query.csv --output_file output.csv && \
python3 infer.py --model_file models/lstm_lstm_attn.pth --model_type lstm_lstm_attn --test_data_file sql_query.csv --output_file output.csv && \
python3 infer.py --model_file models/bert_lstm_attn_frozen.pth --model_type bert_lstm_attn_frozen --test_data_file sql_query.csv --output_file output.csv && \
python3 infer.py --model_file models/bert_lstm_attn_tuned.pth --model_type bert_lstm_attn_tuned --test_data_file sql_query.csv --output_file output.csv