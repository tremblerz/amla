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
"""Initialization (Stem) cell"""

import tensorflow as tf
from stubs.tf.cell import Cell

slim = tf.contrib.slim


def trunc_normal(stddev):
    return tf.truncated_normal_initializer(0.0, stddev)



class Init(Cell):
    """Initialization (Stem) cell: The first cell of a CNN"""
    def __init__(self, cellidx, network):
        self.cellidx = cellidx
        self.cellname = "Init"
        self.network = network
        Cell.__init__(self)
    def __del(self):
        pass
    def cell(self, inputs, arch, is_training):
        """Create the cell by instantiating the cell blocks"""
        nscope = 'Cell_' + self.cellname + '_' + str(self.cellidx)
        reuse = None
        #print(nscope, inputs, [inputs.get_shape().as_list()])
        with tf.variable_scope(nscope, 'initial_block', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                    net = inputs
                    layeridx = 0
                    for layer in sorted(arch.keys()):
                        cells = []
                        for branch in sorted(arch[layer].keys()):
                            block = arch[layer][branch]
                            if block["block"] == "conv2d":
                                output_filters = int(block["outputs"])
                                kernel_size = block["kernel_size"]
                                if "stride" not in block.keys():
                                    stride = 1
                                else:
                                    stride = block["stride"]
                                cell = slim.conv2d(
                                    net,
                                    output_filters,
                                    kernel_size,
                                    stride=stride,
                                    padding='SAME')  # , scope=scope)
                            elif block["block"] == "max_pool":
                                kernel_size = block["kernel_size"]
                                cell = slim.max_pool2d(
                                    net, kernel_size, padding='SAME', stride=2)
                            elif block["block"] == "lrn":
                                cell = tf.nn.lrn(
                                    net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                            elif block["block"] == "dropout":
                                keep_prob = block["keep_prob"]
                                cell = slim.dropout(net, keep_prob=keep_prob)
                            else:
                                print("Invalid block")
                                exit(-1)
                            cells.append(cell)
                        net = tf.concat(cells, axis=-1)

                        layeridx += 1
        #print(nscope, net, [net.get_shape().as_list()])
        return net
