import argparse
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Base directory')
parser.add_argument('--config', help='Configuration file key')
parser.add_argument('--task', help='Task information')

args = parser.parse_args()
sys.path.insert(0, args.base_dir)
from common.task import Task
from generate.enas.general_controller import GeneralController
from generate.enas.general_child import GeneralChild
from stubs.tf import cifar10_input
from generate.enas.child_model import ChildModel

class FLAGS():
    """docstring for FLAGS"""
    def __init__(self):
        pass

        
class Generate(Task):
    def __init__(self, base_dir, config, task):
        super().__init__(base_dir)
        self.base_dir = base_dir
        self.name = 'generate'
        self.get_parameters()
        self.build_controller()
        self.generate_ops()

    def get_parameters(self):
        """ Obtains parameters for the controller
        TODO: Put it in json later
        """
        self.FLAGS = FLAGS()
        self.FLAGS.controller_search_whole_channels = True
        self.FLAGS.controller_entropy_weight = 0.0001
        self.FLAGS.controller_train_every = 1
        self.FLAGS.controller_sync_replicas = True
        self.FLAGS.controller_num_aggregate = 20
        self.FLAGS.controller_train_steps = 50
        self.FLAGS.controller_lr = 0.001
        self.FLAGS.controller_tanh_constant = 1.5
        self.FLAGS.controller_op_tanh_reduce = 2.5
        self.FLAGS.controller_skip_target = 0.4
        self.FLAGS.controller_skip_weight = 0.8
        self.FLAGS.controller_temperature = None
        self.FLAGS.controller_l2_reg = 0.0
        self.FLAGS.controller_entropy_weight = 0.0001
        self.FLAGS.controller_bl_dec = 0.99
        self.FLAGS.controller_use_critic = False
        self.FLAGS.controller_optim_algo = "adam"
        self.FLAGS.controller_num_replicas = 1
        self.FLAGS.controller_num_aggregate = 1
        self.FLAGS.controller_sync_replicas = False#True
        self.FLAGS.controller_lr_dec_start = 0
        self.FLAGS.controller_lr_dec_every = 1000000

        self.FLAGS.child_num_cells = 5
        self.FLAGS.child_num_layers = 12
        self.FLAGS.child_num_branches = 6
        self.FLAGS.child_out_filters = 36

        self.FLAGS.lstm_size = 64
        self.FLAGS.lstm_num_layers = 1
        self.FLAGS.lstm_keep_prob = 1.0
        

    def generate_ops(self):
        controller_model = self.controller_model
        self.controller_ops = {
            "train_step": controller_model.train_step,
            "loss": controller_model.loss,
            "train_op": controller_model.train_op,
            "lr": controller_model.lr,
            "grad_norm": controller_model.grad_norm,
            "valid_acc": controller_model.valid_acc,
            "optimizer": controller_model.optimizer,
            "baseline": controller_model.baseline,
            "entropy": controller_model.sample_entropy,
            "sample_arc": controller_model.sample_arc,
            "skip_rate": controller_model.skip_rate,
        }

        child_model = self.child_model
        self.child_ops = {
            "global_step": child_model.global_step,
            "loss": child_model.loss,
            "train_op": child_model.train_op,
            "lr": child_model.lr,
            "grad_norm": child_model.grad_norm,
            "train_acc": child_model.train_acc,
            "optimizer": child_model.optimizer,
            "num_train_batches": child_model.num_train_batches,
        }

    def build_controller(self):
        self.controller_model = GeneralController(
            search_for="macro",
            search_whole_channels=self.FLAGS.controller_search_whole_channels,
            skip_target=self.FLAGS.controller_skip_target,
            skip_weight=self.FLAGS.controller_skip_weight,
            num_cells=self.FLAGS.child_num_cells,
            num_layers=self.FLAGS.child_num_layers,
            num_branches=self.FLAGS.child_num_branches,
            out_filters=self.FLAGS.child_out_filters,
            lstm_size=self.FLAGS.lstm_size,
            lstm_num_layers=self.FLAGS.lstm_num_layers,
            lstm_keep_prob=self.FLAGS.lstm_keep_prob,
            tanh_constant=self.FLAGS.controller_tanh_constant,
            op_tanh_reduce=self.FLAGS.controller_op_tanh_reduce,
            temperature=self.FLAGS.controller_temperature,
            lr_init=self.FLAGS.controller_lr,
            lr_dec_start=self.FLAGS.controller_lr_dec_start,
            lr_dec_every=self.FLAGS.controller_lr_dec_every,  # never decrease learning rate
            l2_reg=self.FLAGS.controller_l2_reg,
            entropy_weight=self.FLAGS.controller_entropy_weight,
            bl_dec=self.FLAGS.controller_bl_dec,
            use_critic=self.FLAGS.controller_use_critic,
            optim_algo=self.FLAGS.controller_optim_algo,
            sync_replicas=self.FLAGS.controller_sync_replicas,
            num_aggregate=self.FLAGS.controller_num_aggregate,
            num_replicas=self.FLAGS.controller_num_replicas
            )

        '''images, labels = cifar10_input.read_data(self.base_dir + "/data/cifar10/")
        self.child_model = GeneralChild(
            images,
            labels,
            use_aux_heads=False,
            cutout_size=16,
            whole_channels=True,
            num_layers=12,
            #num_cells=FLAGS.child_num_cells,
            #num_branches=FLAGS.child_num_branches,
            fixed_arc=None,
            out_filters_scale=1,
            out_filters=36,
            keep_prob=0.9,
            drop_path_keep_prob=0.6,
            num_epochs=310,
            l2_reg=0.00025,
            data_format="NHWC",
            batch_size=32,
            clip_mode="norm",
            grad_bound=5.0,
            lr_init=0.1,
            lr_dec_every=100,
            lr_dec_rate=0.1,
            lr_cosine=True,
            lr_max=0.05,
            lr_min=0.0005,
            lr_T_0=10,
            lr_T_mul=2,
            optim_algo="momentum",
            sync_replicas=False,
            num_aggregate=1,
            num_replicas=1,
          )'''

        self.child_model = ChildModel(self.base_dir, self.name, self.FLAGS.controller_sync_replicas)
        
        self.child_model.connect_controller(self.controller_model)
        self.controller_model.build_trainer(self.child_model)

    '''def get_global_step(self):
        train_dir = self.base_dir + "/results/"
        ckpt = tf.train.get_checkpoint_state(train_dir)
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

        return global_step'''


    def generate(self):
        # Partition training set into validation and train
        # Write save and load interface for models from the encoding defined
        #init_op = tf.initialize_all_variables()
        controller_ops = self.controller_ops
        child_ops = self.child_ops

        num_iters = 1024

        hooks = []
        #sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
        #hooks.append(sync_replicas_hook)


        #with tf.train.MonitoredTrainingSession(
        #    checkpoint_dir="./results/",
        #    ) as sess:
        with tf.train.SingularMonitoredSession(hooks=hooks) as sess:
            #sess.run(init_op)
            for i in range(num_iters):
                run_ops = [
                    child_ops["loss"],
                    child_ops["lr"],
                    child_ops["grad_norm"],
                    child_ops["train_acc"],
                    child_ops["train_op"],
                ]
                for j in range(1350):
                    loss, _, _, acc, _ = sess.run(run_ops)
                    step = sess.run(child_ops["global_step"])
                    if j % 200 == 0:
                        print("loss at step {} is {:<6.4f} with accuracy {}".format(step, loss, acc))
                acc = sess.run(child_ops["train_acc"])
                print("Training for round {} completed. Acc for the model is {}".format(i, acc))
                avg_val_acc = 0
                for ct_step in range(self.FLAGS.controller_train_steps * self.FLAGS.controller_num_aggregate):
                    run_ops = [
                      controller_ops["loss"],
                      controller_ops["entropy"],
                      controller_ops["lr"],
                      controller_ops["grad_norm"],
                      controller_ops["valid_acc"],
                      controller_ops["baseline"],
                      controller_ops["skip_rate"],
                      controller_ops["train_op"],
                    ]
                    loss, entropy, lr, gn, val_acc, bl, skip, _ = sess.run(run_ops)
                    avg_val_acc += val_acc
                    #print("step number {}, Controller loss = {}, validation acc = {}".format(ct_step, loss, val_acc))
                    controller_step = sess.run(controller_ops["train_step"])
                    #if ct_step % 50 == 0:
                    #    print("Step number {}".format(controller_step))
                print("Average accuracy in round {} is {}".format(i, avg_val_acc / (self.FLAGS.controller_train_steps * self.FLAGS.controller_num_aggregate)))
                #ops["eval_func"](sess, "test")


    def main(self):
        self.generate()

if __name__ == '__main__':
    base_dir = args.base_dir
    config = args.config
    task = args.task
    g = tf.Graph()
    with g.as_default():
        g = Generate(base_dir, config, task)
        g.run()