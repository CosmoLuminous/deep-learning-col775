python3 infer.py --model_file models/part_1.1.pth --normalization inbuilt --test_data_file cifar_test.csv --output_file output.csv
# python3 infer.py --model_file models/part_1.2_bn.pth --normalization bn --test_data_file cifar_test.csv --output_file output.csv && \
# python3 infer.py --model_file models/part_1.2_bin.pth --normalization bin --test_data_file cifar_test.csv --output_file output.csv && \
# python3 infer.py --model_file models/part_1.2_gn.pth --normalization gn --test_data_file cifar_test.csv --output_file output.csv && \
# python3 infer.py --model_file models/part_1.2_in.pth --normalization in --test_data_file cifar_test.csv --output_file output.csv && \
# python3 infer.py --model_file models/part_1.2_ln.pth --normalization ln --test_data_file cifar_test.csv --output_file output.csv && \
# python3 infer.py --model_file models/part_1.2_nn.pth --normalization nn --test_data_file cifar_test.csv --output_file output.csv