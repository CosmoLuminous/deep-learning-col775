python3 -m src.train_resnet --norm_type in --num_workers 16 --result_dir ./results_temp 2>&1 | tee logs/in.log && \
python3 -m src.train_resnet --norm_type gn --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/gn.log && \
python3 -m src.train_resnet --norm_type bin --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/bin.log && \
python3 -m src.train_resnet --norm_type nn --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/nn.log && \
python3 -m src.train_resnet --norm_type ln --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/ln.log && \
python3 -m src.train_resnet --norm_type torch_bn --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/torch_bn.log && \
python3 -m src.train_resnet --norm_type bn --num_workers 16 --result_dir ./results_temp  2>&1 | tee logs/bn.log