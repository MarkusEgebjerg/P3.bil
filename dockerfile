FROM node ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    libusb-1.0-0-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install opencv-python numpy


WORKDIR /app
COPY main.py .


CMD ["python3", "main.py"]