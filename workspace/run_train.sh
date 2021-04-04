# update: 02.04.21, VJH
MODEL=$1

rm -rf /home/paperspace/$MODEL/c* && \
rm -rf /home/paperspace/$MODEL/train/ && \

python3 model_main_tf2.py --model_dir=/home/paperspace/$MODEL --sample_1_of_n_eval_examples=10 --pipeline_config_path=/home/paperspace/$MODEL/pipeline.config
