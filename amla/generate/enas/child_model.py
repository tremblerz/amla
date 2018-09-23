import tensorflow as tf
slim = tf.contrib.slim

from train_eval.tf import net
from stubs.tf import cifar10_input
from generate.enas import model_specs
from generate.enas.utils import get_train_ops

from generate.enas.models import Model
from generate.enas.image_ops import conv
from generate.enas.image_ops import fully_connected
from generate.enas.image_ops import batch_norm
from generate.enas.image_ops import batch_norm_with_mask
from generate.enas.image_ops import relu
from generate.enas.image_ops import max_pool
from generate.enas.image_ops import global_avg_pool

from generate.enas.utils import count_model_params
from generate.enas.utils import get_train_ops
from generate.enas.common_ops import create_weight

class ChildModel():
    def __init__(self, base_dir, name, sync_replicas=False, num_layers=12):
        self.name = "child_model"
        self.base_dir = base_dir
        #self.network = net.Net(self.base_dir, self.task_config)
        self.batch_size = 32
        self.num_layers = num_layers
        self.pool_layers = [2, 5, 8]
        self.sync_replicas = sync_replicas

    def _generate_task_config(self, inputs, is_training):
        with tf.variable_scope("stem_conv"):
            cell = model_specs.get_init_cell(inputs, is_training)
        cell = self._generate_model_config(cell, is_training)
        with tf.variable_scope("classification_cell"):
            cell = model_specs.get_classification_cell(cell, is_training)
        return cell

    def _conv_branch(self, x, k, is_training, count, out_filters, start_idx=0, separable=False):

        inp_c = x.get_shape()[3].value

        initializer = tf.contrib.keras.initializers.he_normal()

        if separable:
            kernel_d = tf.get_variable("kernel_depth", [k, k, out_filters, 1], initializer=initializer, trainable=is_training)
            kernel_w = tf.get_variable("kernel_width", [out_filters, out_filters])
            kernel_w = kernel_w[start_idx:start_idx+count, :]
            kernel_w = tf.transpose(kernel_w, [1, 0])
            kernel_w = tf.reshape(kernel_w, [1, 1, out_filters, count])
            x = tf.nn.separable_conv2d(x, kernel_d, kernel_w,
                [1,1,1,1], padding='SAME')
        else:
            kernel = tf.get_variable("kernel", [k, k, inp_c, out_filters], initializer=initializer, trainable=is_training)
            x = tf.nn.conv2d(x, kernel, [1,1,1,1], "SAME")

        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        return x

    def _pool_branch(self, inputs, is_training, type_):
        if type_ == "max":
            x = slim.max_pool2d(inputs, 2, stride=1, padding='SAME')
        elif type_ == "avg":
            x = slim.avg_pool2d(inputs, 2, stride=1, padding='SAME')
        return x

    '''def _factorized_reduction(self, x, out_filters, stride, is_training):
      """Reduces the shape of x without information loss due to striding."""
      assert out_filters % 2 == 0, (
          "Need even number of filters when using this factorized reduction.")
      if stride == 1:
        with tf.variable_scope("path_conv"):
          inp_c = x.shape[3].value
          w = create_weight("w", [1, 1, inp_c, out_filters])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
          x = batch_norm(x, is_training)
          return x

      stride_spec = [1, stride, stride, 1]
      # Skip path 1
      path1 = tf.nn.avg_pool(
          x, [1, 1, 1, 1], stride_spec, "VALID")
      with tf.variable_scope("path1_conv"):
        inp_c = path1.shape[3].value
        w = create_weight("w", [1, 1, inp_c, out_filters // 2])
        path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "SAME")
    
      # Skip path 2
      # First pad with 0"s on the right and bottom, then shift the filter to
      # include those 0"s that were added.
      pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
      path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
      concat_axis = 3
    
      path2 = tf.nn.avg_pool(
          path2, [1, 1, 1, 1], stride_spec, "VALID")
      with tf.variable_scope("path2_conv"):
        inp_c = path2.shape[3].value
        w = create_weight("w", [1, 1, inp_c, out_filters // 2])
        path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "SAME")
    
      # Concat and apply BN
      final_path = tf.concat(values=[path1, path2], axis=concat_axis)
      final_path = batch_norm(final_path, is_training)

      return final_path'''

    def _enas_layer(self, layer_id, layers, start_idx, out_filters, is_training):
        inputs = layers[-1]
        print(inputs)
        count = self.sample_arc[start_idx]
        branches = {}
        with tf.variable_scope("branch_0"):
            y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                                  start_idx=0)
            branches[tf.equal(count, 0)] = lambda: y
        with tf.variable_scope("branch_1"):
            y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                              start_idx=0, separable=True)
            branches[tf.equal(count, 1)] = lambda: y
        with tf.variable_scope("branch_2"):
            y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                              start_idx=0)
            branches[tf.equal(count, 2)] = lambda: y
        with tf.variable_scope("branch_3"):
            y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                              start_idx=0, separable=True)
            branches[tf.equal(count, 3)] = lambda: y
        with tf.variable_scope("branch_4"):
            y = self._pool_branch(inputs, is_training, "avg")
            branches[tf.equal(count, 4)] = lambda: y
        with tf.variable_scope("branch_5"):
            y = self._pool_branch(inputs, is_training, "max")
            branches[tf.equal(count, 5)] = lambda: y
        out_shape = [self.batch_size, inputs.shape[1], inputs.shape[2], out_filters]
        out = tf.case(branches, default=lambda: tf.constant(0, tf.float32, shape=out_shape),
                    exclusive=True)
        print(out)
        if layer_id > 0:
            skip_start = start_idx + 1

            skip = self.sample_arc[skip_start: skip_start + layer_id]
            with tf.variable_scope("skip"):
                res_layers = []
                for i in range(layer_id):
                  res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                            lambda: layers[i],
                                            lambda: tf.zeros_like(layers[i])))
                res_layers.append(out)
                out = tf.add_n(res_layers)
                out = tf.layers.batch_normalization(out, training=is_training)
        return out


    def custom_factorized_reduction(self, x, is_training):
        x = slim.avg_pool2d(x, 2, padding='SAME')
        return x

    def _generate_model_config(self, inputs, is_training):
        
        layers =[inputs]
        start_idx = 0
        out_filters = inputs.get_shape()[3].value
        for layer_id in range(self.num_layers):
            with tf.variable_scope("layer_{0}".format(layer_id)):
                x = self._enas_layer(layer_id, layers, start_idx, out_filters, is_training)
            layers.append(x)
            if layer_id in self.pool_layers:
                with tf.variable_scope("pool_at_{0}".format(layer_id)):
                    pooled_layers = []
                    for i, layer in enumerate(layers):
                        with tf.variable_scope("from_{0}".format(i)):
                            x = self.custom_factorized_reduction(layer, is_training)
                        pooled_layers.append(x)
                    layers = pooled_layers
            start_idx += 1 + layer_id
        net = x
        return net
        #for arch in arch_encoding

    def _model(self, images, is_training, reuse=False):
        # FIGURE OUT HOW TO INTERFACE AND LOAD MODEL STUBS HERE
        with tf.variable_scope(self.name, reuse=reuse):
          with slim.arg_scope([], reuse=reuse):
            logits = self._generate_task_config(images, is_training)
        return logits

    '''def _conv_branch(self, inputs, filter_size, is_training, count, out_filters,
                   ch_mul=1, start_idx=None, separable=False):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """

      inp_c = inputs.get_shape()[3].value
      with tf.variable_scope("inp_conv_1"):
        w = create_weight("w", [1, 1, inp_c, out_filters])
        x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME")
        x = batch_norm(x, is_training)
        x = tf.nn.relu(x)

      with tf.variable_scope("out_conv_{}".format(filter_size)):
        if start_idx is None:
          if separable:
            w_depth = create_weight(
              "w_depth", [self.filter_size, self.filter_size, out_filters, ch_mul])
            w_point = create_weight("w_point", [1, 1, out_filters * ch_mul, count])
            x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                       padding="SAME")
            x = batch_norm(x, is_training)
          else:
            w = create_weight("w", [filter_size, filter_size, inp_c, count])
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
            x = batch_norm(x, is_training)
        else:
          if separable:
            w_depth = create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul])
            w_point = create_weight("w_point", [out_filters, out_filters * ch_mul])
            w_point = w_point[start_idx:start_idx+count, :]
            w_point = tf.transpose(w_point, [1, 0])
            w_point = tf.reshape(w_point, [1, 1, out_filters * ch_mul, count])

            x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                       padding="SAME")
            mask = tf.range(0, out_filters, dtype=tf.int32)
            mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
            x = batch_norm_with_mask(
              x, is_training, mask, out_filters)
          else:
            w = create_weight("w", [filter_size, filter_size, out_filters, out_filters])
            w = tf.transpose(w, [3, 0, 1, 2])
            w = w[start_idx:start_idx+count, :, :, :]
            w = tf.transpose(w, [1, 2, 3, 0])
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
            mask = tf.range(0, out_filters, dtype=tf.int32)
            mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
            x = batch_norm_with_mask(
              x, is_training, mask, out_filters)
        x = tf.nn.relu(x)
      return x

    def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """

      inp_c = inputs.get_shape()[3].value
      with tf.variable_scope("conv_1"):
        w = create_weight("w", [1, 1, inp_c, 36])
        x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME")
        x = batch_norm(x, is_training)
        x = tf.nn.relu(x)

      with tf.variable_scope("pool"):
        actual_data_format = "channels_last"
        if avg_or_max == "avg":
          x = tf.layers.average_pooling2d(
            x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
        elif avg_or_max == "max":
          x = tf.layers.max_pooling2d(
            x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
        else:
          raise ValueError("Unknown pool {}".format(avg_or_max))

        if start_idx is not None:
          x = x[:, :, :, start_idx : start_idx+count]

      return x

    def _enas_layer(self, layer_id, prev_layers, start_idx, out_filters, is_training):
        """
        Args:
          layer_id: current layer
          prev_layers: cache of previous layers. for skip connections
          start_idx: where to start looking at. technically, we can infer this
            from layer_id, but why bother...
          is_training: for batch_norm
        """

        inputs = prev_layers[-1]
        
        inp_h = inputs.get_shape()[1].value
        inp_w = inputs.get_shape()[2].value
        inp_c = inputs.get_shape()[3].value
        
        count = self.sample_arc[start_idx]
        branches = {}
        with tf.variable_scope("branch_0"):
          y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                                start_idx=0)
          branches[tf.equal(count, 0)] = lambda: y
        with tf.variable_scope("branch_1"):
          y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                                start_idx=0, separable=True)
          branches[tf.equal(count, 1)] = lambda: y
        with tf.variable_scope("branch_2"):
          y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                                start_idx=0)
          branches[tf.equal(count, 2)] = lambda: y
        with tf.variable_scope("branch_3"):
          y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                                start_idx=0, separable=True)
          branches[tf.equal(count, 3)] = lambda: y
        with tf.variable_scope("branch_4"):
          y = self._pool_branch(inputs, is_training, out_filters, "avg",
                                  start_idx=0)
          branches[tf.equal(count, 4)] = lambda: y
        with tf.variable_scope("branch_5"):
          y = self._pool_branch(inputs, is_training, out_filters, "max",
                                  start_idx=0)
          branches[tf.equal(count, 5)] = lambda: y
        out = tf.case(branches, default=lambda: tf.constant(0, tf.float32),
                      exclusive=True)

        
        out_shape = [self.batch_size, inp_h, inp_w, out_filters]
        out = tf.case(branches, default=lambda: tf.constant(0, tf.float32, shape=out_shape),
              exclusive=True)

        if layer_id > 0:
          skip_start = start_idx + 1
          skip = self.sample_arc[skip_start: skip_start + layer_id]
          with tf.variable_scope("skip"):
            res_layers = []
            for i in range(layer_id):
              res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                        lambda: prev_layers[i],
                                        lambda: tf.zeros_like(prev_layers[i])))
            res_layers.append(out)
            out = tf.add_n(res_layers)
            out = batch_norm(out, is_training)

        return out'''

    '''def _model(self, images, is_training, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
          layers = []

          out_filters = 36
          with tf.variable_scope("stem_conv"):
            w = create_weight("w", [3, 3, 3, out_filters])
            x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME")
            x = batch_norm(x, is_training)
            layers.append(x)

          start_idx = 0
          for layer_id in range(self.num_layers):
            with tf.variable_scope("layer_{0}".format(layer_id)):
              x = self._enas_layer(layer_id, layers, start_idx, out_filters, is_training)

              layers.append(x)
              if layer_id in self.pool_layers:
                with tf.variable_scope("pool_at_{0}".format(layer_id)):
                  pooled_layers = []
                  for i, layer in enumerate(layers):
                    with tf.variable_scope("from_{0}".format(i)):
                      x = self._factorized_reduction(
                        layer, out_filters, 2, is_training)
                    pooled_layers.append(x)
                  layers = pooled_layers

            start_idx += 1 + layer_id
            print(layers[-1])

          x = global_avg_pool(x)
          if is_training:
            x = tf.nn.dropout(x, 0.8)
          with tf.variable_scope("fc"):
            inp_c = x.get_shape()[-1].value
            w = create_weight("w", [inp_c, 10])
            x = tf.matmul(x, w)
        return x'''

    def build_train_module(self):
        data_path = self.base_dir + "/data/cifar10/"
        x_train, y_train = cifar10_input.inmemory_distorted_inputs(data_path, self.batch_size)
        self.num_train_examples = x_train.shape[0]
        self.num_train_batches = (
            self.num_train_examples + self.batch_size - 1) // self.batch_size
        logits = self._model(x_train, True)
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y_train)
        self.loss = tf.reduce_mean(log_probs)

        train_preds = tf.argmax(logits, axis=1)
        train_preds = tf.to_int32(train_preds)
        train_acc = tf.equal(train_preds, y_train)
        train_acc = tf.to_float(train_acc)
        self.train_acc = tf.reduce_mean(train_acc)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        tf_variables = [var
            for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
          self.loss,
          tf_variables,
          self.global_step,
          clip_mode="norm",
          grad_bound=5.0,
          l2_reg=2.5e-4,
          lr_init=0.1,
          lr_dec_start=0,
          lr_dec_every=100,
          lr_dec_rate=0.1,
          lr_cosine=True,
          lr_max=0.05,
          lr_min=0.0005,
          lr_T_0=10,
          lr_T_mul=2,
          num_train_batches=self.num_train_batches,
          optim_algo="momentum",
          sync_replicas=self.sync_replicas,
          num_aggregate=None,
          num_replicas=1)

    def connect_controller(self, controller_model):
        self.sample_arc = controller_model.sample_arc
        self.build_train_module()

    def build_valid_rl(self, shuffle=False):
        data_path = self.base_dir + "/data/cifar10/"
        x_valid_shuffle, y_valid_shuffle = cifar10_input.inmemory_inputs(data_path, self.batch_size)
        logits = self._model(x_valid_shuffle, False, reuse=True)
        valid_shuffle_preds = tf.argmax(logits, axis=1)
        valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
        self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
        self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
        self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

    def _build_test(self):
        data_path = self.base_dir + "/data/cifar10/"
        x_test, y_test = cifar10_input.inmemory_inputs(data_path, self.batch_size, type_="test")
        logits = self._model(x_test, False, reuse=True)
        self.test_preds = tf.argmax(logits, axis=1)
        self.test_preds = tf.to_int32(self.test_preds)
        self.test_acc = tf.equal(self.test_preds, y_test)
        self.test_acc = tf.to_float(self.test_acc)
        self.test_acc = tf.reduce_mean(self.test_acc)