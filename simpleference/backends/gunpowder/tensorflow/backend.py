import numpy as np
import os
from gunpowder.ext import tensorflow as tf

class TensorflowPredict(object):
    '''Tensorflow implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        meta_graph_basename: Basename of a tensorflow meta-graph storing the
            trained tensorflow graph (aka a checkpoint), as created by
            :class:`gunpowder.nodes.Train`, for example.

        input_key (string): Name of the input layer.

        outputs (string): Name of the output layer.
    '''

    def __init__(self,
                 meta_graph_basename,
                 input_key,
                 output_key):
        assert os.path.exists(meta_graph_basename + '.meta')
        self.meta_graph_basename = meta_graph_basename
        self.input_key = input_key
        self.output_key = output_key

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self._read_meta_graph()

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        output = self.session.run(self.output_key, feed_dict={self.input_key: input_data})

        assert isinstance(output, np.ndarray)
        if output.ndim == 5:
            output = output[0]
        assert output.ndim == 4
        return output.astype('float32')

    def _read_meta_graph(self):
        # read the meta-graph
        saver = tf.train.import_meta_graph(self.meta_graph_basename + '.meta',
                                           clear_devices=True)
        # restore variables
        saver.restore(self.session, self.meta_graph_basename)

    # Needs to be called in the end
    def stop(self):
        if self.session is not None:
            self.session.close()
            self.graph = None
            self.session = None
