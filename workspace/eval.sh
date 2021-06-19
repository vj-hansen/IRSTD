MODEL=$1

python3 model_main_tf2.py --model_dir=~/$MODEL --pipeline_config_path=~/$MODEL/pipeline.config --checkpoint_dir=~/$MODEL --run_once=True
