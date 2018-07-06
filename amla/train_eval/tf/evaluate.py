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

"""Evaluate class: Task to evaluate a generated network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import sys
import json
import numpy as np
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


class Evaluate(Task):
    """Evaluate task
    """

    def __init__(self, base_dir, config, task):
        super().__init__(base_dir)
        self.name = 'evaluate'
        self.task_config_key = config
        self.task_config = self.read(self.task_config_key)
        self.base_dir = base_dir
        self.task = json.loads(task)
        self.iteration = self.task['iteration']
        self.get_task_params()

    def __del__(self):
        pass

    def get_task_params(self):
        self.mode = self.task_config["parameters"]["mode"]
        self.log_stats = self.task_config["parameters"]["log_stats"]
        self.algorithm = self.task_config["parameters"]["algorithm"]
        self.dataset = self.task_config["parameters"]["dataset"]
        self.eval_interval = self.task_config["parameters"]["eval_interval"]
        self.image_size = self.task_config["parameters"]["image_size"]
        self.arch_name = self.task_config["parameters"]["arch_name"]
        self.init_cell = self.task_config["init_cell"]
        self.classification_cell = self.task_config["classification_cell"]
        self.arch = self.task_config["arch"]
        self.arch_name = self.task_config["parameters"]["arch_name"]
        self.batch_size = self.task_config["parameters"]["batch_size"]
        self.num_examples = 10000
        self.run_once = True
        self.train_dir = self.base_dir + "/results/" + \
            self.arch_name + "/" + str(self.iteration) + "/train"
        self.eval_dir = self.base_dir + "/results/" + \
            self.arch_name + "/" + str(self.iteration) + "/evaluate"
        self.checkpoint_dir = self.train_dir

    def eval_once(self, saver, summary_writer, top_k_op, summary_op, k=1):
        """Run Eval once.
        Args:
          saver: Saver.
          summary_writer: Summary writer.
          top_k_op: Top K op.
          summary_op: Summary op.
        """
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for q_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(
                        q_runner.create_threads(
                            sess,
                            coord=coord,
                            daemon=True,
                            start=True))

                num_iter = int(math.ceil(self.num_examples / self.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * self.batch_size
                step = 0
                #print(
                #    "total_sample_count is {} and num_examples is {}".format(
                #        total_sample_count,
                #        self.num_examples))
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                if k == 1:
                    # Compute precision @ 1.
                    precision = true_count / total_sample_count
                    print(
                        '%s: precision @ 1 = %.3f' %
                        (datetime.now(), precision))
                elif k == 5:
                    # Compute precision @ 5.
                    precision = true_count / total_sample_count
                    print(
                        '%s: precision @ 5 = %.3f' %
                        (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(
                    tag='Precision @ %d' %
                    (k), simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as excpn:  # pylint: disable=broad-except
                coord.request_stop(excpn)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    def evaluate(self, network):
        """Eval CIFAR-10 for a number of steps."""
        with tf.Graph().as_default() as grph:
            # Get images and labels for CIFAR-10.
            eval_data = True
            images, labels = network.inputs(eval_data=eval_data)

            # Build a Graph that computes the logits predictions from the
            # inference model.
            # TODO: Clean up all args
            arch = self.arch
            init_cell = self.init_cell
            classification_cell = self.classification_cell
            arch_name = self.arch_name
            log_stats = self.log_stats
            scope = "Nacnet"
            is_training = False
            logits = network.inference(images,
                                       arch,
                                       arch_name,
                                       init_cell,
                                       classification_cell,
                                       log_stats,
                                       is_training,
                                       scope)

            # Calculate predictions.
            # if imagenet is running then run precision@1,5
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            if self.dataset == "imagenet":
                    # Quick dirty fixes to incorporate changes brought by
                    # imagenet
                self.num_examples = 50000
                top_5_op = tf.nn.in_top_k(logits, labels, 5)

            # Restore the moving average version of the learned variables for
            # eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                net.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of
            # Summaries.
            summary_op = tf.summary.merge_all()

            summary_writer = tf.summary.FileWriter(self.eval_dir, grph)

            while True:
                self.eval_once(saver, summary_writer, top_k_op, summary_op)
                if self.dataset == "imagenet":
                    self.eval_once(saver, summary_writer, top_5_op,
                                   summary_op,
                                   k=5)
                if self.run_once:
                    break
                #time.sleep(self.eval_interval_secs)

    def main(self):
        network = net.Net(self.base_dir, self.task_config)
        network.maybe_download_and_extract()
        if not tf.gfile.Exists(self.eval_dir):
            tf.gfile.MakeDirs(self.eval_dir)
        self.evaluate(network)
        if self.sys_config['exec']['scheduler'] == "service":
             self.put_results()

    def put_results(self):
        task = {"task_id": int(self.task['task_id']), "op": "POST"}
        task['state'] = "complete"
        #self.send_request("scheduler", "tasks/update", task)

def main(argv=None):  # pylint: disable=unused-argument
    config = FLAGS.config
    base_dir = FLAGS.base_dir
    task = FLAGS.task
    evaluate = Evaluate(base_dir, config, task)
    evaluate.run()


if __name__ == '__main__':
    tf.app.run()
