# update: 06-06-21, VJH
MODEL=$1

python3 model_main_tf2.py --model_dir=/home/my_folder/$MODEL --pipeline_config_path=/home/my_folder/$MODEL/pipeline.config --checkpoint_dir=/home/my_folder/$MODEL --run_once=True
