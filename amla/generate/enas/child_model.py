import tensorflow as tf
slim = tf.contrib.slim

from train_eval.tf import net
from stubs.tf import cifar10_input
from generate.enas import model_specs


class ChildModel():
    def __init__(self, base_dir, name, num_layers=9):
        self.name = name + "child_model"
        self.base_dir = base_dir
        #self.network = net.Net(self.base_dir, self.task_config)
        self.batch_size = 32
        self.num_layers = num_layers
        self.pool_layers = [2, 5, 8]

    def _generate_task_config(self, inputs, is_training):
        config = {}
        config["arch_name"] = self.name
        config["init_cell"] = model_specs.get_init_cell(inputs, is_training)
        config["arch"] = self._generate_model_config(config["init_cell"], is_training)
        config["classification_cell"] = model_specs.get_classification_cell(config["arch"], is_training)
        config["log_stats"] = False
        config["is_training"] = False
        return config["classification_cell"]

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


    def _factorized_reduction(self, x, is_training):
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
                            x = self._factorized_reduction(layer, is_training)
                        pooled_layers.append(x)
                    layers = pooled_layers
            start_idx += 1 + layer_id
        net = x
        return net
        #for arch in arch_encoding

    def _model(self, images, is_training, reuse=False):
        # FIGURE OUT HOW TO INTERFACE AND LOAD MODEL STUBS HERE
        with tf.variable_scope(self.name, reuse=reuse):
            '''config = self._generate_model_config(images)
            logits = self.network.inference(images,
                                   arch=config["arch"],
                                   arch_name=self.arch_name,
                                   init_cell=config["init_cell"],
                                   classification_cell=config["classification_cell"],
                                   log_stats=False,
                                   is_training=is_training)'''
            logits = self._generate_task_config(images, is_training)
        return logits

    def build_train_module(self, global_step):
        self.global_step = global_step
        data_path = self.base_dir + "/data/cifar10/"
        x_train, y_train = cifar10_input.inmemory_distorted_inputs(data_path, self.batch_size)
        logits = self._model(x_train, True)
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y_train)
        self.loss = tf.reduce_mean(log_probs)

        train_preds = tf.argmax(logits, axis=1)
        train_preds = tf.to_int32(train_preds)
        train_acc = tf.equal(train_preds, y_train)
        train_acc = tf.to_float(train_acc)
        self.acc = tf.reduce_mean(train_acc)

        learning_rate = 3e-4
        self.train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3).minimize(self.loss, global_step=self.global_step)

    def connect_controller(self, controller_model, global_step):
        self.sample_arc = controller_model.sample_arc
        self.build_train_module(global_step)

    def build_valid_rl(self, shuffle=False):
        data_path = self.base_dir + "/data/cifar10/"
        x_valid_shuffle, y_valid_shuffle = cifar10_input.inmemory_inputs(data_path, self.batch_size)
        logits = self._model(x_valid_shuffle, False, reuse=True)
        valid_shuffle_preds = tf.argmax(logits, axis=1)
        valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
        self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
        self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
        self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)