sudo apt update && sudo apt upgrade && \
    sudo apt install python3-pip && \
    sudo apt install python-tk protobuf-compiler python-lxml python-pil && \
    pip3 install -r requirements.txt && \
    pip3 install --user --upgrade tensorflow tensorflow-gpu
