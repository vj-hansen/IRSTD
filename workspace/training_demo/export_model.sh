MODEL=$1

python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/$MODEL/pipeline.config --trained_checkpoint_dir ./models/$MODEL/ --output_directory ./exported-models/my_$MODEL
