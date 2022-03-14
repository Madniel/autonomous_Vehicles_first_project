import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('cnn_models/model_resnet32.h5')

test_img = cv2.imread('/home/legion/PycharmProjects/AV/src/av_04/images/500.png')
test_img2 = cv2.resize(test_img, (200, 42))
test_img2 = test_img2[np.newaxis, ...]

print(model.predict(test_img2))
cv2.imshow('img_to_predict', test_img)
cv2.waitKey()
