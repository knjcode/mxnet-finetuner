#!/usr/bin/env bash

# Save and/or compress specified model and logs
#
# Usage:
#   $ util/save_model.sh [options] <model_prefix_with_epoch> <output_dir>
#   $ util/save_model.sh -z -f 201705292200-imagenet1k-nin-sgd-0001 my-nin-model
#

set -eu

### usage
function usage() {
  local err=${1:-}
  local ret=0
  if [ "${err}" ]; then
    exec 1>&2
    echo "${err}"
    ret=1
  fi
  cat <<EOF
Usage: $(basename "$0") [options] <model_prefix_with_epoch> <output_dir>
  Options are:
  -h, --help      show help
  -f, --force     overwrite dst_dir and compressed file
  -m, --model     copy model only (config, model, valid logs (default - test logs))
  -z, --compress  compress dst_dir in tar.gz format
Example usage: $(basename "$0") -z -f 201705292200-imagenet1k-nin-sgd-0001 my-nin-model
EOF
  exit ${ret}
}

### options
opt_overwrite=
opt_copy_mode="default"
opt_compress=
while [ ${#} -ne 0 ]; do
  case ${1} in
  -h | --help)
    usage
    ;;
  -f | --force | --overwrite)
    opt_overwrite=yes
    shift
    ;;
  -m | --model | --copy_model_only)
    opt_copy_mode="model_only"
    shift
    ;;
  -z | --compress)
    opt_compress=yes
    shift
    ;;
  --)
    break
    ;;
  -*)
    usage "'${1}' not supported"
    ;;
  *)
    break
    ;;
  esac
done

### arguments
if [ ${#} -eq 2 ]; then
  model_prefix_with_epoch=${1}
  dst_dir=${2}
else
  usage
fi

if [ -e "${dst_dir}" ]; then
  if [ "${opt_overwrite}" == "yes" ]; then
    rm -fr "${dst_dir}"
  else
    echo "${dst_dir} is exist!"
    exit 1
  fi
fi

### setup
model_prefix=${model_prefix_with_epoch%-*}
params="model/${model_prefix_with_epoch}.params"
symbol="model/${model_prefix}-symbol.json"
labels="model/${model_prefix}-labels.txt"

test -e "${params}" || { echo "${params} is not exist"; exit 1; }

### copying
test -d "${dst_dir}" || mkdir "${dst_dir}"

# model
mkdir "${dst_dir}/model"
for f in "${params}" "${symbol}" "${labels}"; do
  cp -i "${f}" "${dst_dir}/model"
done

# logs
mkdir "${dst_dir}/logs"
find "logs"                          \
  | grep "${model_prefix}"                      \
  | { test "${opt_copy_mode}" == "model_only"   \
      && grep -v "${model_prefix}-.*-test-" \
      || cat ; }                                \
  | xargs -I{} cp -i {} "${dst_dir}/logs"

# show tree
if which tree >/dev/null; then
  echo "# tree -L 3 ${dst_dir}"
  tree -L 3 "${dst_dir}"
fi

# compress
if [ "${opt_compress}" == "yes" ]; then
  compressed="${dst_dir}.tar.gz"
  if [ -e "${compressed}" ]; then
    if [ "${opt_overwrite}" == "yes" ]; then
      rm -f "${compressed}"
    else
      echo "${compressed} is exist! Compression skipped."
      exit 1
    fi
  fi
  tar czf "${dst_dir}.tar.gz" "${dst_dir}" \
  && echo "compressed model and logs: ${dst_dir}.tar.gz"
fi

# end
