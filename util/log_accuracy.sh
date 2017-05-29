#!/usr/bin/env bash

LOG="$1"

cat "$LOG" | grep Validation-acc | sort -t'=' -k2
