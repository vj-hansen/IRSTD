# InfraRed Small Target Detection

Keep the following folders outside of the repo:
```
# too large for GitHub
workspace/training_demo/pre_trained_model/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8/* 
workspace/training_demo/models/centernet/ckpt-* 
```


```bash
bash setup.sh

git clone https://github.com/tensorflow/models.git
```

```bash
# From within TensorFlow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```


### Object Detection API
* Notice the ```.```!
```bash
# From within /models/research/
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .

# Test installation
python3 object_detection/builders/model_builder_tf2_test.py
```


### Install GPU support for TensorFlow (Ubuntu 18.04 LTS)

* If trouble -> check out: https://www.tensorflow.org/install/gpu

```bash
sudo apt-get install cuda
sudo apt install nvidia-cuda-toolkit

# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin && \
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && \
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-450
# Reboot. Check that GPUs are visible using the command: 
nvidia-smi

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb && \
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb && \
sudo apt-get update

# Install development and runtime libraries (~4 GB)
sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Install TensorRT. Requires that libcudnn8 is installed .
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

nvcc --version
```

### Download model
* cd to ```workspace/training_demo/pre_trained_model```
* Pick a model from [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

```
# e.g. CenterNet MobileNetV2 FPN 512x512
wget -c http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz -O - | tar -xz


# Need to copy pipeline.config from the downloaded model into a new dir: models/[my_model]/
cd models/

# create a folder
mkdir my_model


# from workspace/training_demo

cp -v pre_trained_model/[downloaded_model]/pipeline.config models/[my_model]
```

### Configurate pipeline

```
# pipeline config

num_classes: 1

...
batch_size: 32 # e.g. higher batch size = more memory used

... 
# Comment fine_tune_checkpoint: "p" 
# Comment fine_tune_checkpoint_type: "detection"
use_bfloat16: false # Set this to false if you are not training on a TPU

... train
label_map_path: "data/label_map.pbtxt"
input_path: "data/train.record"


... in eval_config
num_examples: 268 # num_examples = number of items in 'test.record'-file
max_evals: 10


... eval_input_reader
  label_map_path: "data/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "data/test.record"
```


### Folder structure
```
TensorFlow/
├─ models/
│   ├─ .../
│   ├─ .../
│   ├─ .../
│   └── research
│       ├── object_detection
│       └── ...
└─ workspace/
    └─ training_demo/
        ├─ data/
        │  ├─ label_map.pbtxt
        │  ├─ train.record
        │  └─ test.record
        ├─ exported-models/
        ├─ images/
        │  ├─ test/
        │  └─ train/
        ├─ models/
        │   └── [a_model] (model under evaluation)
        │       └── pipeline.config
        ├─ model_main_tf2.py
        ├─ exporter_main_v2.py
        ├─ pre_trained_model/
        └─ run_train.sh
```

---


#### To start the training from scratch
```bash
# Remove ckpt files in models/[a_model] 
rm -rf checkpoint && rm -rf ckpt-* && rm -rf train
```


### Start training

```bash
cd workspace/training_demo

# Start TensorBoard, this can be done after the training is done
tensorboard --logdir=models/[a_model]
# opens at http://localhost:6006/

bash run_train.sh [a_model]
```

### Export model

```bash
bash export_model.sh [a_model]
```




### Download exported model and TensorBoard data 
#### Compress the follow dirs
```

tar -czvf saved_model.tar.gz /home/paperspace/exported-models/my_centernet
tar -czvf tensorboard_data.tar.gz /home/paperspace/centernet/train


curl --upload-file saved_model.tar.gz https://transfer.sh/saved_model.tar.gz
curl --upload-file tensorboard_data.tar.gz https://transfer.sh/tensorboard_data.tar.gz
```

