#!/usr/bin/env python
# coding: utf-8

# Upload a file to slack
#
# Environments:
#   $ export SLACK_API_TOKEN="your-slack-api-token"
# Usage:
#   $ ./slack_file_upload.py <target channels> <upload filename>
#   $ ./slack_file_upload.py "general random" logs/$MODEL_PREFIX-$MODEL-$OPTIMIZER.png

import re
import os
import sys
from slackclient import SlackClient

channels = sys.argv[1]
upload_file = sys.argv[2]

# "general random" or "general\nrandom" -> "#general,#random"
channels = ','.join(['#'+ch for ch in re.split(r' |\n', channels)])

print("Uploading a file to Slack. channels=%s" % channels)

try:
    slack_token = os.environ["SLACK_API_TOKEN"]
except KeyError:
    print("Error: Set your Slack API token as SLACK_API_TOKEN environment variable.")
    sys.exit(1)

sc = SlackClient(slack_token)

response = sc.api_call(
    'files.upload',
    channels=channels,
    filename=upload_file,
    file=open(upload_file, 'rb')
)

if response['ok']:
    print("  Upload success.")
else:
    print("  Upload failed.")
    print(response)
