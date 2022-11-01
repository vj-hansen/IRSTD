MODEL=$1

rm -rf ~/$MODEL/c* && \
rm -rf ~/$MODEL/train/ && \

python3 model_main_tf2.py --model_dir=~/$MODEL --sample_1_of_n_eval_examples=10 --pipeline_config_path=~/$MODEL/pipeline.config
