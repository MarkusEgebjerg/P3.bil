
program can be stop using `Ctrl + c` 


```
sudo docker pull tonton04/p3-bil:latest
```

 Basic run
```
sudo docker run -it --rm \
  --privileged \
  -v /dev:/dev \
  -v /run:/run \
  tonton04/p3-bil:latest

```
if want moitor und 
```
sudo docker run -it --rm \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -v /run:/run \
  tonton04/p3-bil:latest
```

run on jetson use ssh to connet and pull the same 
```
ssh -v jetson@192.168.55.1
```
use this command to run
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

Test camera only
```
sudo docker run -it --rm \
  --privileged \
  -v /dev:/dev \
  -v $(pwd):/app \
  tonton04/p3-bil:latest \
  python3 test_system.py --camera
```
Test motors only
```
sudo docker run -it --rm \
  --privileged \
  -v /dev:/dev \
  -v $(pwd):/app \
  tonton04/p3-bil:latest \
  python3 test_system.py --arduino```


  
