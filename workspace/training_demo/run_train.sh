MODEL=$1

python3 model_main_tf2.py --model_dir=models/$MODEL --pipeline_config_path=models/$MODEL/pipeline.config  --alsologtostderr --eval_on_train_data=true
