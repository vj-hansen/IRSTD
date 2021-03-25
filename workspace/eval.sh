# update: 24.03.21, VJH
MODEL=$1

# add --run_once=True

python3 model_main_tf2.py --model_dir=/home/paperspace/$MODEL --pipeline_config_path=/home/paperspace/$MODEL/pipeline.config --checkpoint_dir=/home/paperspace/$MODEL
