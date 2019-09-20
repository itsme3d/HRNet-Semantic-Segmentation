FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev -y
RUN apt-get install wget file -y

WORKDIR /root

RUN wget https://s3.amazonaws.com/assets.dwble.com/nn_images/hrnet_w48_lip_cls20_473x473.pth

RUN git clone https://github.com/itsme3d/HRNet-Semantic-Segmentation.git
RUN cd HRNet-Semantic-Segmentation && pip install -r requirements.txt

RUN mkdir HRNet-Semantic-Segmentation/data/kevin/
RUN wget https://s3.amazonaws.com/assets.dwble.com/nn_images/100034_483681.jpg -O HRNet-Semantic-Segmentation/data/kevin/100034_483681.jpg
RUN wget https://s3.amazonaws.com/assets.dwble.com/nn_images/100034_483681.png -O HRNet-Semantic-Segmentation/data/kevin/100034_483681.png

RUN apt-get install nano -y

# docker build -t highresnet.pytorch.v0.4.1 .
# docker run --runtime=nvidia -i -t highresnet.pytorch.v0.4.1 bash

# cd HRNet-Semantic-Segmentation && python tools/test.py --cfg experiments/kevin/kevin.yaml \
#   DATASET.TEST_SET list/kevin/testvalList.txt \
#   TEST.MODEL_FILE ../hrnet_w48_lip_cls20_473x473.pth \
#   TEST.FLIP_TEST True \
#   TEST.NUM_SAMPLES 0
