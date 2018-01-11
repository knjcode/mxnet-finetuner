# Setup

```
$ git clone https://github.com/knjcode/mxnet-finetuner
$ cd mxnet-finetuner
$ bash setup.sh
```

`setup.sh` will automatically generate` docker-compose.yml` and `config.yml` which are necessary for executing this tool according to your environment such as existence of the GPU.


## After updating the GPU driver of the host machine

When updating the GPU driver of the host machine, it is necessary to stop all GPU containers and delete the volume.

```
$ docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
```

To create a volume, execute the `nvidia-docker` command once.

```
$ nvidia-docker run --rm nvidia/cuda nvidia-smi
```

And then, re-running `bash setup.sh`.
