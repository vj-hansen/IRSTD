# update: 24.03.21, VJH
MODEL=$1

python3 model_main_tf2.py --model_dir=/home/paperspace/$MODEL --pipeline_config_path=/home/paperspace/$MODEL/pipeline.config --checkpoint_dir=/home/paperspace/$MODEL --run_once=True
