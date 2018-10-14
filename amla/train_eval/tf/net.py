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

""" Net module:
Functions/variables common to both training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
from six.moves import urllib

import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_eval.tf.generate_network import gen_amlanet

from stubs.tf import cifar10_input
from stubs.tf import imagenet_input


# Basic model parameters.
# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999
# Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = 2
LEARNING_RATE_DECAY_FACTOR = 0.999    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1             # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

"""Net: Neural Net Construction
"""


class Net:
    def __init__(self, base_dir, task_config):
        self.base_dir = base_dir
        self.task_config = task_config
        self.cells = []
        self.end_points = []
        self.nets = []
        self.use_fp16 = False
        self.get_task_params()

    def get_task_params(self):
        self.batch_size = self.task_config["parameters"]["batch_size"]
        self.dataset = self.task_config["parameters"]["dataset"]
        self.image_size = self.task_config["parameters"]["image_size"]
        self.data_dir = self.base_dir + "/" + \
            self.task_config["parameters"]["data_dir"]
        self.arch = self.task_config["arch"]

    def _activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

        Args:
            x: Tensor
        Returns:
            nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

        Returns:
            Variable Tensor
        """
        with tf.device('/cpu:0'):
            dtype = tf.float16 if self.use_fp16 else tf.float32
            var = tf.get_variable(
                name, shape, initializer=initializer, dtype=dtype)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wdecay):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wdecay: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.

        Returns:
            Variable Tensor
        """
        dtype = tf.float16 if self.use_fp16 else tf.float32
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wdecay is not None:
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wdecay, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def distorted_inputs(self):
        """Construct distorted input for a given dataset using the Reader ops.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.

        Raises:
            ValueError: If no data_dir
        """
        if not self.data_dir:
            raise ValueError('Please supply a data_dir')
        if self.dataset == 'cifar10':
            data_dir = os.path.join(self.data_dir, 'cifar-10-batches-bin')
            images, labels = cifar10_input.distorted_inputs(
                data_dir=data_dir, batch_size=self.batch_size, image_size=self.image_size)
        elif self.dataset == 'imagenet':
            images, labels = imagenet_input.distorted_inputs(batch_size=self.batch_size, image_size=self.image_size, data_dir=self.data_dir)
        if self.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
        return images, labels

    def inputs(self, eval_data):
        """Construct input for CIFAR evaluation using the Reader ops.

        Args:
            eval_data: bool, indicating if one should use the train or eval data set.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.

        Raises:
            ValueError: If no data_dir
        """
        if not self.data_dir:
            raise ValueError('Please supply a data_dir')
        if self.dataset == 'cifar10':
            data_dir = os.path.join(self.data_dir, 'cifar-10-batches-bin')
            images, labels = cifar10_input.inputs(
                eval_data=eval_data, data_dir=data_dir, batch_size=self.batch_size, image_size=self.image_size)
        elif self.dataset == 'imagenet':
            data_dir = self.data_dir
            #if self.dataset_split_name == "test":
            self.dataset_split_name = "validation"
            images, labels = imagenet_input.inputs(batch_size=self.batch_size, image_size=self.image_size, data_dir=self.data_dir)
        if self.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
        return images, labels

    def inference(self, images, arch=None,
                  archname=None,
                  initcell=None,
                  classificationcell=None,
                  log_stats=False,
                  is_training=None,
                  scope='Nacnet'
                 ):

        softmax_linear = gen_amlanet(
            images,
            arch,
            archname,
            initcell,
            classificationcell,
            log_stats,
            is_training,
            scope)
        return softmax_linear

    def loss(self, logits, labels, child_training):
        """Add L2Loss to all the trainable variables.

        Add summary for "Loss" and "Loss/avg".
        Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                            of shape [batch_size]

        Returns:
            Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        label_smoothing = child_training.get("label_smoothing", None)
        if label_smoothing:
            epsilon = child_training["label_smoothing"]["value"]
            one_hot_labels = tf.one_hot(labels, depth=logits.get_shape()[1].value)
            cross_entropy = tf.losses.softmax_cross_entropy(
                             one_hot_labels, logits, label_smoothing=epsilon) 
        else:
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Get auxiliary loss, TODO remove hardcoded weight of 0.4
        aux_logits = tf.get_collection('auxiliary_loss')
        weight = 0.4
        for num, logits in enumerate(aux_logits):
            #print(logits)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='aux_loss_{}'.format(num))
            cross_entropy_mean = weight * tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return total_loss

    def tower_loss(self, scope, logits, labels, child_training):
        """Calculate the total loss on a single tower running the CIFAR model.
            Args:
            scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
            images: Images. 4D tensor of shape [batch_size, height, width, 3].
            labels: Labels. 1D tensor of shape [batch_size].
            Returns:
            Tensor of shape [] containing the total loss for a batch of data
        """
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        _ = self.loss(logits, labels, child_training)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for loss in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
            # training
            # # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % 'tower', '', loss.op.name)
            tf.summary.scalar(loss_name, loss)

        return total_loss

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all
           towers.
           Note that this function provides a synchronization point across all
           towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer
            list is over individual gradients. The inner list is over the
            gradient calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been 
            average across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for grad, _ in grad_and_vars:
                """
                gradient for variance and mean calculation is None and hence shouldn't be added to the list of grads
                """
                # if g is not None:
                if True:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(grad, 0)
                    # Append on a 'tower' dimension which we will average over
                    # below.
                    grads.append(expanded_g)
            '''
            # This break also to avoid adding empty list because of variance and mean calculation
            if len(grads) == 0:
                break
            '''
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            # try:
            grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's
            # pointer to # the Variable.
            var = grad_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads.append(grad_and_var)
        return average_grads

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for loss in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(loss.op.name + ' (raw)', loss)
            tf.summary.scalar(loss.op.name, loss_averages.average(loss))

        return loss_averages_op

    def get_regularization_loss(self, total_loss, child_training):
        if child_training["regularization"]["type"] == "l2":
            # l2-regularization
            l2_reg = child_training["regularization"]["value"]
            tf_variables = [var for var in tf.trainable_variables()]
            l2_losses = []
            for var in tf_variables:
               l2_losses.append(tf.reduce_sum(var**2))
            l2_loss = tf.add_n(l2_losses)
            total_loss += l2_reg * l2_loss
        return total_loss

    def get_learning_rate(self, global_step, child_training, gpus=None):
        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // self.batch_size
        if gpus:
            global_step *= len(gpus)
        curr_epoch = global_step // num_batches_per_epoch

        if "lr" not in child_training.keys() or child_training["lr"]["type"] == "exponential_decay":
            initial_lr = child_training["lr"].get("initial", INITIAL_LEARNING_RATE)
            lr_decay = child_training["lr"].get("decay", LEARNING_RATE_DECAY_FACTOR)
            epochs_per_decay = child_training["lr"].get("epochs_per_decay", NUM_EPOCHS_PER_DECAY)

            decay_steps = int(num_batches_per_epoch * epochs_per_decay)
            learning_rate = tf.train.exponential_decay(initial_lr,
                                                       curr_epoch,
                                                       decay_steps,
                                                       lr_decay,
                                                       staircase=True)            

        elif child_training["lr"]["type"] == "cosine_decay":
            curr_epoch = tf.to_int32(curr_epoch)
            last_reset = tf.Variable(0, dtype=tf.int32, trainable=False,
                                                 name="last_reset")
            lr_max = child_training["lr"]["max"]
            lr_min = child_training["lr"]["min"]
            lr_T_0 = child_training["lr"]["T_0"]
            lr_T_mul = child_training["lr"]["T_mul"]
            T_curr = curr_epoch - last_reset
            T_i = tf.Variable(lr_T_0, dtype=tf.int32,trainable=False, name="T_i")

            def _update():
                update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
                update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
                with tf.control_dependencies([update_last_reset, update_T_i]):
                  rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                  learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
                return learning_rate

            def _no_update():
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
                return learning_rate

            learning_rate = tf.cond(
                tf.greater_equal(T_curr, T_i), _update, _no_update)

        return learning_rate

    def get_opt(self, learning_rate, child_training):
        if "optimizer" not in child_training.keys() or child_training["optimizer"]["type"] == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif child_training["optimizer"]["type"] == "rms":
            opt = tf.train.RMSPropOptimizer(lr, 0.9, 0.9, 1.0)
        elif child_training["optimizer"]["type"] == "momentum":
            momentum = child_training["optimizer"]["momentum"]
            opt = tf.train.MomentumOptimizer(learning_rate,
              momentum, use_locking=True, use_nesterov=True)

        return opt

    def clip_gradients(self, grads, child_training):
        if "gradient_clipping" in child_training.keys():
            if child_training["gradient_clipping"]["type"] == "norm":
                # Gradient clipping based on norm
                clipped = []
                grad_bound = child_training["gradient_clipping"]["value"]
                for grad, var in grads:
                   if isinstance(grad, tf.IndexedSlices):
                       c_g = tf.clip_by_norm(grad.values, grad_bound)
                       c_g = tf.IndexedSlices(grad.indices, c_g)
                   else:
                       c_g = tf.clip_by_norm(grad, grad_bound)
                   clipped.append((c_g, var))
                grads = clipped
        return grads

    def get_train_op(self, total_loss, global_step, child_training):
        """Train CIFAR-10 model.

        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.

        Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
                processed.
        Returns:
            train_op: op for training.
        """
        learning_rate = self.get_learning_rate(global_step, child_training)
        tf.summary.scalar('learning_rate', learning_rate)

        if "regularization" in child_training.keys():
            total_loss = self.get_regularization_loss(total_loss, child_training)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = self.get_opt(learning_rate, child_training)
            grads = opt.compute_gradients(total_loss)

        grads = self.clip_gradients(grads, child_training)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        #for var in tf.trainable_variables():
        #    tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        #for grad, var in grads:
        #    if grad is not None:
        #        tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def maybe_download_and_extract(self):
        """Download and extract the tarball from Alex's website."""
        if self.dataset == 'cifar10':
            dest_directory = self.data_dir
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)
            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(dest_directory, filename)
            if not os.path.exists(filepath):
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                         float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                filepath, _ = urllib.request.urlretrieve(
                    DATA_URL, filepath, _progress)
                print()
                statinfo = os.stat(filepath)
                print(
                    'Successfully downloaded',
                    filename,
                    statinfo.st_size,
                    'bytes.')
            extracted_dir_path = os.path.join(
                dest_directory, 'cifar-10-batches-bin')
            if not os.path.exists(extracted_dir_path):
                tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        elif self.dataset == 'imagenet':
            """ It is assumed that if imagenet dataset is specified then it already exists
                and not supposed to be downloaded
            """
            if not os.path.exists(self.data_dir):
                print("Directory {} doesn't exist!".format(self.data_dir))
                exit(-1)
        else:
            print("Unknown dataset {}".format(self.dataset))
            exit(-1)

    
    def get_params(self):
        for cell in self.cells:
            cell.get_params()

    
