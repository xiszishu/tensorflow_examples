from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

from progress.bar import IncrementalBar
from simulate_crossbar.rram_weights import Rram_weights

# This example tests a dog picture
def test_dog_picture():
  test_img = './dog.png'
  img2 = image.load_img(test_img)
  resized_images = img2.resize((224, 224))
  print(type(resized_images))
  resized_images.save("dog_resized.png", "PNG", optimize=True)
  x = image.img_to_array(resized_images)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x)
  result = np.argmax(features)
  print(result)

def iterate_list(input_array, rram_crossbar):
  if (type(input_array) is np.ndarray):
    for index, x in np.ndenumerate(input_array):
      input_array[index] = rram_crossbar.actual_weight(input_array[index])
  else:
    for idx in range(0, len(input_array)):
      iterate_list(input_array[idx], rram_crossbar)

def main(argv):
  image_path = '../ILSVRC2012_devkit_t12/image/'
  model = InceptionV3(weights='imagenet', include_top=True)
  #test_dog_picture()
  result_file = open("val.txt", "r")
  correct_num = 0
  num_images = 1000
  topk = 5
  filename = tf.placeholder(tf.string, name="inputFile")
  fileContent = tf.read_file(filename, name="loadFile")
  image_file = tf.image.decode_jpeg(fileContent, channels=3, name="decodeJpeg")
  resize_nearest_neighbor = tf.image.resize_images(
      image_file,
      size=[224, 224],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  sess = tf.Session()
  suffix = '%(index)d/%(max)d [%(elapsed)d / %(eta)d / %(eta_td)s]'
  bar = IncrementalBar('Processing', max=num_images, suffix=suffix)
  rram_crossbar = Rram_weights(16, 350)
  l_weights = model.get_weights()
  #print(l_weights[0][0][0][0])
  iterate_list(l_weights, rram_crossbar)
  model.set_weights(l_weights)

  for i in range(1, num_images + 1):
    img_file = "{}{}{:0>8d}{}".format(image_path, "ILSVRC2012_val_", i,
                                      ".JPEG")
    feed_dict = {filename: img_file}
    with sess.as_default():
      x = resize_nearest_neighbor.eval(feed_dict)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    result = np.argsort(features)[0][-topk:]
    result_line = result_file.readline()
    correct_result = int(result_line.split()[1])
    if (correct_result in result): correct_num += 1
    bar.next()

  bar.finish()
  print("Accuracy: {0:.2f}%".format(float(correct_num) / num_images * 100))

if __name__ == '__main__':
  main(sys.argv)
