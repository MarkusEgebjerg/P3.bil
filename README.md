```
$ xhost +local:docker
```

```
sudo docker pull tonton04/p3-bil:latest
```
```
sudo docker run -it --rm \
--runtime nvidia \
--network host \
--privileged \
--gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev:/dev \
-v /run:/run \
tonton04/p3-bil:latest
```
  
