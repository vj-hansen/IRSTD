# Update: 06-06-21, VJH
MODEL=$1

# you have to set the path

python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path /home/my_folder/$MODEL/pipeline.config --trained_checkpoint_dir /home/my_folder/$MODEL/ --output_directory /home/my_folder/exported-models/my_$MODEL
