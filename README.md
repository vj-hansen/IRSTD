# InfraRed Small Target Detection


* PaperSpace
* Machine
* Terminal


```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt install python3-pip
sudo apt-get install python-tk 
sudo apt-get install protobuf-compiler python-lxml python-pil
pip3 install -r requirements.txt (add Cython pandas tf-slim lvis)

pip3 install --user --upgrade tensorflow
pip3 install --user --upgrade tensorflow-gpu

git clone https://github.com/tensorflow/models.git

```

```bash
# From within TensorFlow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```
Dont need
```bash
# From within TensorFlow/models/research/
git clone https://github.com/cocodataset/cocoapi.git # not needed
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```

### Object Detection API
```bash
# From within /models/research/
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .

# Test installation
python3 object_detection/builders/model_builder_tf2_test.py
```


### Install GPU support for TensorFlow (Ubuntu 18.04 LTS)

* https://www.tensorflow.org/install/gpu

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

# Install development and runtime libraries (~4GB)
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
cd to workspace/training_demo/pre-trained-models
```
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# e.g. CenterNet MobileNetV2 FPN 512x512
wget -c http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz -O - | tar -xz

```

### Folder structure
```
TensorFlow/
├─ models/
│   ├─ community/
│   ├─ official/
│   ├─ orbit/
│   └── research
│       ├── object_detection
│       └── pycocotools
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
        │   └── [some model] (model under evaluation)
        │       └── pipeline.config
        ├─ model_main_tf2.py
        ├─ exporter_main_v2.py
        ├─ pre-trained-models/
        └─ run_train.sh

```

---


#### To start the training from scracth
```bash
# Remove ckpt files in models/ssd_mobilenet_v2. 
rm -rf checkpoint && rm -rf ckpt-* && rm -rf train
```


### Start training

```bash
cd InfraRedSmallTargetDetector/workspace/training_demo

# Start TensorBoard
tensorboard --logdir=models/ssd_mobilenet_v2
# opens at http://localhost:6006/


python model_main_tf2.py --model_dir=models/ssd_mobilenet_v2 --pipeline_config_path=models/ssd_mobilenet_v2/pipeline.config  --alsologtostder --eval_on_train_data=true


#or bash run_train.sh
```

### Export model

```bash
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/ssd_mobilenet_v2/pipeline.config --trained_checkpoint_dir ./models/ssd_mobilenet_v2/ --output_directory ./exported-models/my_model
# python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/[name_of_pre-trained-model you downloaded]/pipeline.config --trained_checkpoint_dir ./models/[name_of_pre-trained-model_you_downloaded]/ --output_directory ./exported-models/my_model


bash export_model.sh ssd_mobilenet_v2

```
