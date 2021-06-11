# update: 06-06-21, VJH

MODEL=$1

# remove old files
rm -rf /home/my_folder/$MODEL/c* && \
rm -rf /home/my_folder/$MODEL/train/ && \

python3 model_main_tf2.py --model_dir=/home/my_folder/$MODEL --sample_1_of_n_eval_examples=10 --pipeline_config_path=/home/my_folder/$MODEL/pipeline.config
