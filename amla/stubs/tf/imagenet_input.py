# Copyright 2018 Cisco Systems All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Imagenet input functions"""

import tensorflow as tf
from stubs.tf.imagenet.imagenet_data import ImagenetData
import stubs.tf.imagenet.image_processing as imgnet

FLAGS = tf.app.flags.FLAGS

def distorted_inputs(batch_size, image_size, data_dir):
    dataset = ImagenetData("train", data_dir)
    images, labels = imgnet.distorted_inputs(
        dataset,
        batch_size=batch_size,
        image_size=image_size,
        num_preprocess_threads=4
    )
    return images, labels

def inputs(batch_size, image_size, data_dir):
    dataset = ImagenetData("validation", data_dir)
    images, labels = imgnet.inputs(dataset,
        batch_size=batch_size,
        image_size=image_size)
    return images, labels
