import tensorflow as tf

saved_model_dir = "path_to_saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("/tmp/model.tflite", "wb").write(tflite_model)
