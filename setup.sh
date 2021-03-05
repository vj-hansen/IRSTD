sudo apt-get update && sudo apt-get upgrade && \
    sudo apt install python3-pip && \
    sudo apt-get install python-tk protobuf-compiler python-lxml python-pil && \
    pip3 install -r requirements.txt && \
    pip3 install --user --upgrade tensorflow tensorflow-gpu && \
    git clone https://github.com/tensorflow/models.git
