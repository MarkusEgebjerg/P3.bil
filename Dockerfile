FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    python3 python3-pip python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    libusb-1.0-0-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python packages (OpenCV, NumPy, RealSense)
# -------------------------------------------------------
RUN pip3 install opencv-python numpy pyrealsense2 Jetson.GPIO

# -------------------------------------------------------
# Copy your application
# -------------------------------------------------------
WORKDIR /app
COPY . .

CMD ["python3", "main.py"]
