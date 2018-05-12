#!/usr/bin/env bash

# Generate docker-compose.yml and config.yml for mxnet-finetuner

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$CUR_DIR/util/functions"
source "$CUR_DIR/util/vendor/mo"

CONFIG_FILE='config.yml'
DOCKER_COMPOSE_FILE='docker-compose.yml'

if ! which jq > /dev/null 2>&1; then
  echo 'jq command not found.' 1>&2
  exit 1
fi

# check nvidia-docker version
if nvidia-docker > /dev/null 2>&1; then
  NVIDIA_DOCKER_VERSION=$(nvidia-docker version|head -n1|cut -d' ' -f3|cut -d'.' -f1)
fi

### for nvidia-docker ver 1.x
if [[ $NVIDIA_DOCKER_VERSION != "2" ]]; then
  if which wget > /dev/null 2>&1; then
    config=$(wget -qO- 'http://localhost:3476/docker/cli/json')
  else
    if which curl > /dev/null 2>&1; then
      config=$(curl -s 'http://localhost:3476/docker/cli/json')
    else
      echo 'wget and curl commands not found.' 1>&2
      exit 1
    fi
  fi

  VOLUMES=($(echo "$config" | jq -r ".Volumes | .[]"))
  ExistDEV=$(echo "$config" | jq .Devices)
  DEVICES=($(echo "$config" | jq -r ".Devices | .[]"))
fi
### for nvidia-docker ver 1.x end

# Generate docker-compose.yml according to your environment
if [ -e docker-compose.yml ]; then
  echo -n "Overwrite docker-compose.yml? (y/n [n]): "
  read -r ANS
  case $ANS in
    "Y" | "y" | "yes" | "Yes" | "YES" )
      generate_compose "$CUR_DIR" "$DOCKER_COMPOSE_FILE" "$NVIDIA_DOCKER_VERSION" \
      && update_compose "$CUR_DIR" "$DEVICES" "$DOCKER_COMPOSE_FILE" "$NVIDIA_DOCKER_VERSION"
      ;;
    * )
      echo "not overwritten" 1>&2
      ;;
  esac
else
  generate_compose "$CUR_DIR" "$DOCKER_COMPOSE_FILE" \
  && update_compose "$CUR_DIR" "$DEVICES" "$DOCKER_COMPOSE_FILE"
fi

# Generate config.yml if it does not exist yet.
if [ -e config.yml ]; then
  echo -n "Overwrite config.yml? (y/n [n]): "
  read -r ANS
  case $ANS in
    "Y" | "y" | "yes" | "Yes" | "YES" )
      generate_config "$CUR_DIR" "$CONFIG_FILE" \
      && update_config "$CUR_DIR" "$DEVICES" "$CONFIG_FILE"
      ;;
    * )
      echo "not overwritten" 1>&2
      ;;
  esac
else
  generate_config "$CUR_DIR" "$CONFIG_FILE" \
  && update_config "$CUR_DIR" "$DEVICES" "$CONFIG_FILE"
fi
