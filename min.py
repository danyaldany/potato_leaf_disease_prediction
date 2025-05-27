import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"D:\Code\python\Potato_Leaf_Diseas\saved_model\2")
dummy_input = np.random.rand(1, 255, 255, 3).astype(np.float32)
output = model.predict(dummy_input)
print("🔍 Dummy prediction:", output)
