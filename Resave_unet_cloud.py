import tensorflow as tf

model = tf.keras.models.load_model(
    "models/brain_tumor_segmentation_unet_infer.keras",
    compile=False
)

model.trainable = False

# SAVE AS A NEW FILE (IMPORTANT)
model.save("models/segmentation_cloud.keras")

print("Cloud-safe model saved")
