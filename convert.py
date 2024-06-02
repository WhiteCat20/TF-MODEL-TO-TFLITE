import tensorflow as tf
from keras.models import load_model

modelName = "02062024-234854.keras"
# Load the Keras model (include custom objects if any)
model = load_model(modelName)
file_name = modelName.split('.')[0]
# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(f'{file_name}.tflite', 'wb') as f:
    f.write(tflite_model)
