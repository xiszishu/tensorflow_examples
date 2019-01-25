from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import sys
from progress.bar import IncrementalBar


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


def main(argv):
  image_path = '../ILSVRC2012_devkit_t12/image/'
  model = ResNet50(weights='imagenet', include_top=True)
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
  for i in range(1, num_images + 1):
    img_file = "{}{}{:0>8d}{}".format(image_path, "ILSVRC2012_val_", i,
                                      ".JPEG")
    feed_dict = {filename: img_file}
    with sess.as_default():
      x = resize_nearest_neighbor.eval(feed_dict)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    #result = np.argmax(features)
    result = np.argsort(features)[0][-topk:]
    result_line = result_file.readline()
    correct_result = int(result_line.split()[1])
    #print(correct_result, result)
    if (correct_result in result): correct_num += 1
    #else: print(img_file)
    bar.next()

  bar.finish()
  #resized_images.save("picture_resized.jpeg", "JPEG", optimize=True)
  print("Accuracy: {0:.2f}%".format(float(correct_num) / num_images * 100))


if __name__ == '__main__':
  main(sys.argv)
