```
$ xhost +local:docker
```

```
$ sudo docker container run -it --rm \
  --runtime=nvidia --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /proc/device-tree/compatible:/proc/device-tree/compatible \
  -v /proc/device-tree/chosen:/proc/device-tree/chosen \
  --device /dev/gpiochip0 \
  tonton04/p3-bil:latest
```
  
