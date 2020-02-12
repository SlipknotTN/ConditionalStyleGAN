Use docker based on Tensorflow 1.14
-----------------------------------

```
docker pull docker.io/tensorflow/tensorflow:1.14.0-gpu-py3
nvidia-docker run -it --rm -v /home/dl/ConditionalStyleGAN:/app docker.io/tensorflow/tensorflow:1.14.0-gpu-py3
```

```
pip install scipy pillow requests
cd /app
python train.py
```