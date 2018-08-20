DOCKER_IMAGE_VERSION=cpu
DOCKER_IMAGE_NAME=knjcode/mxnet-finetuner
DOCKER_IMAGE_TAGNAME=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

default: build

build:
	docker pull ubuntu:xenial
	docker build -t $(DOCKER_IMAGE_TAGNAME) .

rebuild:
	docker pull ubunt:xenial
	docker build --no-cache -t $(DOCKER_IMAGE_TAGNAME) .

push:
	docker push $(DOCKER_IMAGE_TAGNAME)

test:
	docker-compose run finetuner help
