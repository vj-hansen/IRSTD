MODEL=$1

python3 model_main_tf2.py --model_dir=models/$MODEL --sample_1_of_n_eval_examples=10 --pipeline_config_path=models/$MODEL/pipeline.config  --alsologtostderr
