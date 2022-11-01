MODEL=$1

python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path ~/$MODEL/pipeline.config --trained_checkpoint_dir ~/$MODEL/ --output_directory ~/exported-models/my_$MODEL
