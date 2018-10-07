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

# This code is derived from TensorFlow: https://github.com/tensorflow/models
# by The TensorFlow Authors, Google
"""Evaluate class: Task to train a generated network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import ast
import sys
import time
import json

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', './configs/config.json',
                           """Configuration file""")
tf.app.flags.DEFINE_string('base_dir', '.',
                           """Working directory to run from""")
tf.app.flags.DEFINE_string('task', '.',
                           """Task information""")
sys.path.insert(0, FLAGS.base_dir)
from train_eval.tf import net
from common.task import Task

# Globals needed by log hook
global_batch_size = 128
global_log_frequency = 10


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, global_step_init, loss):
        self.global_step_init = global_step_init
        self.loss = loss

    def begin(self):
        self._step = self.global_step_init
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        # Asks for loss value.
        return tf.train.SessionRunArgs(self.loss)

    def after_run(self, run_context, run_values):
        if self._step % global_log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = global_log_frequency * global_batch_size / duration
            sec_per_batch = float(duration / global_log_frequency)

            format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                'sec/batch)')
            print(
                format_str %
                (datetime.now(),
                 self._step,
                 loss_value,
                 examples_per_sec,
                 sec_per_batch))

class Train(Task):
    """Trainingtask
    """

    def __init__(self, base_dir, config, task):
        super().__init__(base_dir)
        self.name = 'train'
        self.task_config_key = config
        self.task_config = self.read(self.task_config_key)
        self.base_dir = base_dir
        self.task = json.loads(task)
        self.iteration = self.task['iteration']
        self.max_steps = self.task['steps']
        self.get_task_params()

    def __del__(self):
        pass

    def get_task_params(self):
        global global_batch_size
        global global_log_frequency
        self.count_params = False
        self.log_device_placement = False
        global_log_frequency = 10
#        with open(self.config, 'r') as f:
#                self.task_config = json.load(f)
#        sys_config=self.base_dir+"/system.json"
#        with open(sys_config, 'r') as f:
#                self.sys_config = json.load(f)
        # Mode: oneshot, construct, test
        # Alg: deterministic, random, envelopenet

        # TODO : Move to train/eval/common, pass path to json config file
        #self.execmode = self.task_config["parameters"]["exec"]
        self.mode = self.task_config["parameters"]["mode"]
        self.log_stats = self.task_config["parameters"]["log_stats"]
        self.algorithm = self.task_config["parameters"]["algorithm"]

        self.steps = self.task_config["parameters"]["steps"]
        self.batch_size = self.task_config["parameters"]["batch_size"]
        self.dataset = self.task_config["parameters"]["dataset"]
        self.eval_interval = self.task_config["parameters"]["eval_interval"]
        self.image_size = self.task_config["parameters"]["image_size"]

        self.arch_name = self.task_config["parameters"]["arch_name"]
        self.narch_name = self.task_config["parameters"]["arch_name"]

        self.iterations = self.task_config["parameters"]["iterations"]
        self.init_cell = self.task_config["init_cell"]
        self.classification_cell = self.task_config["classification_cell"]
        self.child_training = self.task_config["child_training"] if "child_training" in self.task_config.keys() else {}
        if self.algorithm == "deterministic":
            self.task_config[self.algorithm] = {
                "arch": self.task_config["arch"]}
        if self.mode == "oneshot":
            self.iterations = 1
        #self.data_dir = self.base_dir + "/" + \
        #    self.task_config["parameters"]["data_dir"] + '/' + str(self.iteration)+"/train"
        self.gpus = self.task_config["parameters"]["gpus"]
        gpu_fraction = self.task_config["parameters"].get("gpu_usage", 1.0)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        self.arch = self.task_config["arch"]
        global_batch_size = self.batch_size
        self.train_dir = self.base_dir + "/results/" + \
            self.arch_name + "/" + str(self.iteration) + "/train"

#    def count_params(self):
#        i=0
#        if self.count_params:
#           self.train(5, redirect='>')
#        return
    def train(self, network):
        with tf.Graph().as_default():
            ckpt = tf.train.get_checkpoint_state(self.train_dir)
            global_step_init = -1
            if ckpt and ckpt.model_checkpoint_path:
                global_step_init = int(
                    ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                global_step = tf.Variable(
                    global_step_init,
                    name='global_step',
                    dtype=tf.int64,
                    trainable=False)
            else:
                global_step = tf.contrib.framework.get_or_create_global_step()

            images, labels = network.distorted_inputs()
            if self.gpus:
                batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                        [images, labels], capacity=2 * len(self.gpus))
                tower_grads = []

            arch = self.arch
            arch_name = self.arch_name
            init_cell = self.init_cell
            classification_cell = self.classification_cell
            log_stats = self.log_stats
            scope = "Nacnet"
            is_training = True

            if self.count_params:
                # For counting parameters
                param_stats = tf.profiler.profile(
                    tf.get_default_graph(),
                    options=tf.profiler.ProfileOptionBuilder()
                    .with_max_depth(2)
                    .with_accounted_types(['_trainable_variables'])
                    .select(['params'])
                    .build())
                # For counting flops
                flop_stats = tf.profiler.profile(
                    tf.get_default_graph(),
                    options=tf.profiler.ProfileOptionBuilder() .with_max_depth(1) .select(
                        ['float_ops']).build())
                print(param_stats)
                print(flop_stats)
                exit()

            if self.gpus:
                # Multi-gpu setting
                learning_rate = network.get_learning_rate(global_step, self.child_training)
                tf.summary.scalar('learning_rate', learning_rate)
                opt = network.get_opt(learning_rate, self.child_training)
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in self.gpus:
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                                # Dequeues one batch for the GPU
                                image_batch, label_batch = batch_queue.dequeue()
                                logits = network.inference(image_batch,
                                                           arch,
                                                           arch_name,
                                                           init_cell,
                                                           classification_cell,
                                                           log_stats,
                                                           is_training,
                                                           scope)
                                # Calculate the loss for one tower of the CIFAR model. This function
                                # constructs the entire CIFAR model but shares the variables across
                                # all towers.
                                loss = network.tower_loss(scope, logits, label_batch)
                                loss = network.get_regularization_loss(loss, self.child_training)
                                tf.get_variable_scope().reuse_variables()
                                # Retain the summaries from the final tower. TODO:
                                # not a nice way to use the last iteration of the
                                # loop
                                summaries = tf.get_collection(
                                    tf.GraphKeys.SUMMARIES, scope)
                                grads = opt.compute_gradients(loss)
                                grads = network.clip_gradients(grads, self.child_training)

                                tower_grads.append(grads)
                grads = network.average_gradients(tower_grads)
                for grad, var in grads:
                    if grad is not None:
                        summaries.append(
                            tf.summary.histogram(
                                var.op.name + '/gradients', grad))
        
                apply_gradient_op = opt.apply_gradients(
                    grads, global_step=global_step)

                variable_averages = tf.train.ExponentialMovingAverage(
                    net.MOVING_AVERAGE_DECAY, global_step)
                variables_averages_op = variable_averages.apply(
                    tf.trainable_variables())

                train_op = tf.group(apply_gradient_op, variables_averages_op)
            else:
                logits = network.inference(images,
                                       arch,
                                       arch_name,
                                       init_cell,
                                       classification_cell,
                                       log_stats,
                                       is_training,
                                       scope)
                loss = network.loss(logits, labels)
                train_op = network.get_train_op(loss, global_step, self.child_training)

            saver = tf.train.Saver()
            with tf.train.MonitoredTrainingSession(
                        checkpoint_dir=self.train_dir,
                        hooks=[tf.train.StopAtStepHook(last_step=self.max_steps),
                               tf.train.NanTensorHook(loss),
                               _LoggerHook(global_step_init, loss)],
                        save_checkpoint_secs=300,
                        save_summaries_steps=100,
                        config=tf.ConfigProto(
                            log_device_placement=self.log_device_placement,
                            allow_soft_placement=True,
                            gpu_options=self.gpu_options
                            )) as mon_sess:

                ckpt = tf.train.get_checkpoint_state(self.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Restoring existing model")
                    saver.restore(mon_sess, ckpt.model_checkpoint_path)

                while not mon_sess.should_stop():
                    mon_sess.run(train_op)

    def main(self):
        network = net.Net(self.base_dir, self.task_config)
        network.maybe_download_and_extract()
        if not tf.gfile.Exists(self.train_dir):
            tf.gfile.MakeDirs(self.train_dir)
        self.train(network)
        if self.sys_config['exec']['scheduler'] == "service":
             self.put_results()

    def put_results(self):
        task = {"task_id": int(self.task['task_id']), "op": "POST"}
        if self.task["steps"] == self.task_config["parameters"]["steps"]:
            task['state'] = "complete"
        else:
            task['state'] = "running"
        self.send_request("scheduler", "tasks/update", task)


def main(argv=None):  # pylint: disable=unused-argument
    config = FLAGS.config
    base_dir = FLAGS.base_dir
    task = FLAGS.task
    train = Train(base_dir, config, task)
    train.run()


if __name__ == '__main__':
    tf.app.run()
