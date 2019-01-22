import tensorflow as tf
import numpy as np
from cn.models.model_cnn import *
from numpy import float32

class ModelCnnTest(tf.test.TestCase):
    def test_model(self):
        inputs = np.random.uniform(0, 1, [2, 299, 299, 3])
        inputs = inputs.astype(float32)
        net = cnn_base(inputs)
        print(net)
        
if __name__ == '__main__':
    ModelCnnTest().test_model()