import tensorflow as tf
import os
import sys
import numpy as np
sys.path.append('../')

from simulate_crossbar.rram_weights import Rram_weights

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
init = tf.global_variables_initializer()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
checkpoint_path = "./checkpoints/my_checkpoint"
checkpoint_dir = os.path.dirname(checkpoint_path)


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, activation):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    self.activation = activation

  def build(self, input_shape):
    self.kernel = self.add_variable(
        "kernel", shape=[int(input_shape[-1]), self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

def iterate_list(input_array, rram_crossbar):
  if (type(input_array) is np.ndarray):
    for index, x in np.ndenumerate(input_array):
      input_array[index] = rram_crossbar.actual_weight(input_array[index])
  else:
    for idx in range(0, len(input_array)):
      iterate_list(input_array[idx], rram_crossbar)

def create_model():
  layer1 = MyDenseLayer(512, tf.nn.relu)
  layer2 = MyDenseLayer(10, tf.nn.softmax)
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu, name='ly1'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  return model


def main(argv):
  model = create_model()
  model.load_weights(checkpoint_path)
  sess = tf.Session()
  sess.run(init)
  #model.fit(x_train, y_train, epochs=5)
  score = model.evaluate(x_test, y_test)
  print(score)
  #print(tf.global_variables())
  #print(model.layers[1].get_weights())
  #l_weights = model.layers[1].get_weights()
  #l_weights[0][1] = 0
  #print(type(l_weights))
  #model.layers[1].set_weights(l_weights)
  #print(model.layers[1].get_weights())
  rram_crossbar = Rram_weights(16, 400)
  l_weights = model.get_weights()
  iterate_list(l_weights, rram_crossbar)
  model.set_weights(l_weights)
  score = model.evaluate(x_test, y_test)
  print(score)
  #model.save_weights(checkpoint_path)


if __name__ == '__main__':
  main(sys.argv)
