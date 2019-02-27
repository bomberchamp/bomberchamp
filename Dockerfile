FROM python:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y ffmpeg

RUN pip install -q numpy scipy sklearn
RUN pip install -q pygame
RUN pip install -q jupyter
RUN pip install -q tf-nightly

WORKDIR /wd

EXPOSE 8888

CMD ["python", "main.py"]



# BUILD: docker build -t bomberchamp .

# MACOS:
# docker rm bomberchamp; xhost + 127.0.0.1; docker run -v $(pwd):/wd --name bomberchamp -it -p 8888:8888 -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix bomberchamp:latest
# LINUX:
# docker rm bomberchamp; docker run -v $(pwd):/wd --name bomberchamp -it -p 8888:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix bomberchamp:latest
