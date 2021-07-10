
MODEL=$1

python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path /home/paperspace/$MODEL/pipeline.config --trained_checkpoint_dir /home/paperspace/$MODEL/ --output_directory /home/paperspace/exported-models/my_$MODEL
