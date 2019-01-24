from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys

def main(argv):
  model = ResNet50(weights=None, input_shape=(32, 32, 3))
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  #model.fit(x_train, y_train, epochs=2)
  model.load_weights('./checkpoint/my_checkpoint')
  score = model.evaluate(x_test, y_test)
  print(score)
  #model.save_weights('./checkpoint/my_checkpoint')


if __name__ == '__main__':
  main(sys.argv)
