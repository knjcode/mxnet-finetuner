# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Modified from https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py

"""`MXNetVisionService` defines a MXNet base vision service
"""

from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import image, ndarray
import mxnet as mx

class MXNetVisionService(MXNetBaseService):
    """MXNetVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """
    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            try:
                rgb_mean = self.signature['inputs'][idx]['rgb_mean']
                rgb_std = self.signature['inputs'][idx]['rgb_std']
            except KeyError:
                rgb_mean = None
                rgb_std = None
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = image.read(img)
            # resize and center_crop
            img_arr = mx.image.resize_short(img_arr, h)
            img_arr = mx.image.center_crop(img_arr, (h, w))[0]
            if rgb_mean:
                img_arr = image.color_normalize(img_arr, mx.nd.array(rgb_mean), std=mx.nd.array(rgb_std))
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [ndarray.top_probability(d, self.labels, top={{TOP_K}}) for d in data]

