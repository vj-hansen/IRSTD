# InfraRed Small Target Detection


* PaperSpace
* Machine
* Terminal


```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt install python3-pip
sudo apt-get install python-tk protobuf-compiler python-lxml python-pil
pip3 install -r requirements.txt

pip3 install --user --upgrade tensorflow
pip3 install --user --upgrade tensorflow-gpu

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
* cd to ```workspace/training_demo/pre-trained-models```
* Pick a model from [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

```
# e.g. CenterNet MobileNetV2 FPN 512x512
wget -c http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz -O - | tar -xz
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
        ├─ pre-trained-models/
        └─ run_train.sh
```

---


#### To start the training from scracth
```bash
# Remove ckpt files in models/[a_model] 
rm -rf checkpoint && rm -rf ckpt-* && rm -rf train
```


### Start training

```bash
cd workspace/training_demo

# Start TensorBoard
tensorboard --logdir=models/[a_model]
# opens at http://localhost:6006/

bash run_train.sh [a_model]
```

### Export model

```bash
bash export_model.sh [a_model]
```
