from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import sys


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
  model = VGG16(weights='imagenet', include_top=True)
  #test_dog_picture()
  result_file = open("val.txt", "r")
  correct_num = 0
  num_images = 50000
  for i in range(1, num_images + 1):
    img_file = "{}{}{:0>8d}{}".format(image_path, "ILSVRC2012_val_", i,
                                      ".JPEG")
    img2 = image.load_img(img_file)
    resized_images = img2.resize((224, 224))
    x = image.img_to_array(resized_images)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    result = np.argmax(features)
    result_line = result_file.readline()
    correct_result = int(result_line.split()[1])
    if (correct_result == result): correct_num += 1
  print("Accuracy: {0:.2f}%".format(float(correct_num) / num_images * 100))


if __name__ == '__main__':
  main(sys.argv)
