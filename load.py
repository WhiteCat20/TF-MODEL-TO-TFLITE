import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import easygui


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model bar.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_path = easygui.fileopenbox(title='Select Image File')
# img = image.load_img(test_image_path)
img = cv2.imread(str(input_path))
img = cv2.resize(img, (200, 200))

# Convert the image to a numpy array
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)

# Prepare the input tensor
interpreter.set_tensor(input_details[0]['index'], X)

# Invoke the interpreter
interpreter.invoke()

# Get the output
val = interpreter.get_tensor(output_details[0]['index'])

# Interpret the output
if val[0][0] >= 0.5:
    print(f'unripe {val[0][0]}')
else:
    print(f'ripe {val[0][0]}')
