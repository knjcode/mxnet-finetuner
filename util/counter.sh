#!/usr/bin/env bash

# Count the number of files/directories in each subdirectory
#
# Usage:
#   $ util/counter.sh [options] <target_dir>
#
# Options:
#   -r   Sort in descending order of the number of files
#

shopt -s expand_aliases

usage_exit() {
  echo "Usage: util/counter.sh [options] <target_dir>" 1>&2
  echo "Options:" 1>&2
  echo "  -r   Sort in descending order of the number of files" 1>&2
  exit 1
}

count_files() {
  local DIR="$1"
  find "$DIR" -maxdepth 1 -mindepth 1 -type d | while read -r subdir;do
    echo "$(basename "$subdir"),$(echo $(ls "$subdir" | wc -l))";
  done
}

count_directories() {
  local DIR="$1"
  echo $(find ${DIR}/* -maxdepth 0 -type d | wc -l)
}

# Reference: http://qiita.com/b4b4r07/items/dcd6be0bb9c9185475bb
declare -i argc=0
declare -a argv=()
while (( $# > 0 ))
do
  case "$1" in
    -*)
      if [[ "$1" =~ 'r' ]]; then
          REVERSE=1
      fi
      if [[ "$1" =~ 'h' ]]; then
          usage_exit
      fi
      shift
      ;;
    *)
      ((++argc))
      argv=("${argv[@]}" "$1")
      shift
      ;;
  esac
done

if [ $argc -lt 1 ]; then
  usage_exit
fi

DIR="${argv[0]}"

FILES=$(count_files "$DIR")
CLASSES=$(count_directories "$DIR")
RESULT=$(echo "$FILES" | sort -n -t',' -k2 | tr ',' ' ')

if which tac >/dev/null; then
  alias tac='tac'
else
  alias tac='tail -r'
fi

if [[ "$REVERSE" = 1 ]]; then
  RESULT=$(echo "$RESULT" | tac)
fi

echo "$DIR contains "$CLASSES" directories"
echo "$RESULT" | column -t
