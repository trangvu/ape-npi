import pickle

import tensorflow as tf
import os

from translate import utils


class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc, model_path=None):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            ckpt = tf.train.get_checkpoint_state(loc)
            if ckpt and ckpt.model_checkpoint_path:
                if model_path is None:
                    ckpt_name = ckpt.model_checkpoint_path
                else:
                    ckpt_name = loc + "/" + model_path
                self.saver = tf.train.import_meta_graph(ckpt_name + '.meta')
                self.saver.restore(self.sess, ckpt_name)


    def get_variable_value(self, var_name):
        with self.graph.as_default():
            vars = [v for v in tf.trainable_variables()
             if v.name == var_name][0]
            values = self.sess.run(vars)
        return values

    def close_session(self):
        self.sess.close()


def load_checkpoint(sess, checkpoint_dir, filename, variables):
    if filename is not None:
        ckpt_file = checkpoint_dir + "/" + filename
        utils.log('reading model parameters from {}'.format(ckpt_file))
        tf.train.Saver(variables).restore(sess, ckpt_file)

        utils.debug('retrieved parameters ({})'.format(len(variables)))
        for var in sorted(variables, key=lambda var: var.name):
            utils.debug('  {} {}'.format(var.name, var.get_shape()))

