
import tensorflow as tf

# Load your existing model (the one that works locally)
model = tf.keras.models.load_model(
    "models/segmentation_cloud.keras",
    compile=False
)

model.trainable = False

# Save in HDF5 format (cloud-safe)
model.save("models/segmentation_cloud.h5")

print("segmentation_cloud.h5 saved successfully")
