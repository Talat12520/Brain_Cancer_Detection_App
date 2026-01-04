import tensorflow as tf

model = tf.keras.models.load_model(
    "models/brain_tumor_segmentation_unet.keras",
    compile=False
)

model.trainable = False

# Save clean inference-only model
model.save("models/brain_tumor_segmentation_unet_infer.keras")

print("Model re-saved successfully")
