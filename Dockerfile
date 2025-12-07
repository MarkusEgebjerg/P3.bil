FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
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

# Install Arduino CLI + AVR + Servo library
RUN curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh \
    && mv bin/arduino-cli /usr/local/bin/ \
    && arduino-cli core update-index \
    && arduino-cli core install arduino:avr \
    && arduino-cli lib install Servo

# Python packages
RUN pip3 install opencv-python numpy pyrealsense2 pyserial

# Copy application
WORKDIR /app
COPY . .

# Make upload script executable
RUN chmod +x /app/upload_arduino.py

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting AAU Racing RC Car System..."\n\
echo ""\n\
# Upload Arduino sketch if it exists\n\
if [ -f "/app/arduino/motorcontroller/motorcontroller.ino" ]; then\n\
    python3 /app/upload_arduino.py\n\
else\n\
    echo "Warning: Arduino sketch not found, skipping upload"\n\
fi\n\
echo ""\n\
# Run main program\n\
python3 main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]