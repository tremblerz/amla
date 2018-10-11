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
"""Classification cell"""

import tensorflow as tf

from stubs.tf.cell import Cell

slim = tf.contrib.slim

def trunc_normal(stddev):
    return tf.truncated_normal_initializer(0.0, stddev)

class Classification(Cell):
    """Classification cell: The final classification block of a CNN"""
    def __init__(self):
        self.cellname = "Classification"
        Cell.__init__(self)
    def __del__(self):
        pass
    def cell(self, inputs, arch, is_training):
        """Create the cell by instantiating the cell blocks"""
        nscope = 'Cell_' + self.cellname
        net = inputs
        reuse = None
        #print(nscope, inputs, [inputs.get_shape().as_list()])
        with tf.variable_scope(nscope, 'classification_block', [inputs], reuse=reuse) as scope:
            for layer in sorted(arch.keys()):
                for branch in sorted(arch[layer].keys()):
                    block = arch[layer][branch]
                    if block["block"] == "reduce_mean":
                        net = tf.reduce_mean(net, [1, 2])
                    elif block["block"] == "avgpool":
                        kernel_size = block["kernel_size"]
                        net = slim.avg_pool2d(net, kernel_size=kernel_size)
                    elif block["block"] == "flatten":
                        net = slim.flatten(net)
                    elif block["block"] == "fc":
                        outputs = block["outputs"]
                        net = slim.fully_connected(net, outputs)
                    elif block["block"] == "fc-final":
                        outputs = block["outputs"]
                        inputs = block["inputs"]
                        weights_initializer = trunc_normal(1 / float(inputs))
                        biases_initializer = tf.zeros_initializer()
                        net = slim.fully_connected(
                            net,
                            outputs,
                            biases_initializer=biases_initializer,
                            weights_initializer=weights_initializer,
                            weights_regularizer=None,
                            activation_fn=None)
                    elif block["block"] == "dropout":
                        keep_prob = block["keep_prob"]
                        net = slim.dropout(
                            net, keep_prob=keep_prob, is_training=is_training)
                    else:
                        print("Invalid block")
                        exit(-1)
        #print(nscope, net, [net.get_shape().as_list()])
        return net
