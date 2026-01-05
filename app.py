import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image
from io import BytesIO

# =========================
# CONFIG
# =========================
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLF_IMG_SIZE = 224
SEG_IMG_SIZE = 224

st.set_page_config(
    page_title="Brain Tumor Classification & Segmentation",
    layout="centered"
)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_classification_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, 4)

    state_dict = torch.load(
        "models/best_mobilenet_model.pth",
        map_location=DEVICE
    )
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_segmentation_model():
    return tf.keras.models.load_model(
        "models/segmentation_cloud.h5",
        compile=False
    )


clf_model = load_classification_model()
seg_model = load_segmentation_model()

# =========================
# TRANSFORMS
# =========================
clf_transform = transforms.Compose([
    transforms.Resize((CLF_IMG_SIZE, CLF_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# UTILS
# =========================
def load_for_segmentation(image: Image.Image):
    img = image.resize((SEG_IMG_SIZE, SEG_IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def overlay_mask(image: Image.Image, mask):
    image = image.resize((SEG_IMG_SIZE, SEG_IMG_SIZE))
    image = np.array(image)

    red_mask = np.zeros_like(image)
    red_mask[..., 0] = mask.squeeze() * 255

    overlay = np.clip(
        0.7 * image + 0.3 * red_mask,
        0, 255
    ).astype(np.uint8)

    return overlay


def convert_image(img_array):
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_tumor_type(image: Image.Image):
    img = clf_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = clf_model(img)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    idx = np.argmax(probs)
    return CLASS_NAMES[idx], float(probs[idx]), probs


def predict_mask(image: Image.Image):
    img = load_for_segmentation(image)
    pred = seg_model.predict(img, verbose=0)[0]
    return (pred > 0.5).astype(np.uint8)

# =========================
# STREAMLIT UI
# =========================
st.title("🧠 Brain Tumor Classification & Segmentation")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI")
    st.image(image, width=400)

    # ===== CLASSIFICATION =====
    label, confidence, all_probs = predict_tumor_type(image)

    st.subheader("Tumor Classification Result")
    st.write(f"**Tumor Type:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Prediction Confidence")
    st.progress(confidence)

    st.subheader("Class Probabilities")
    for cls, prob in zip(CLASS_NAMES, all_probs):
        st.write(f"{cls}: {prob:.2f}")

    # ===== SEGMENTATION =====
    if label != "no_tumor":
        st.subheader("Tumor Segmentation")

        mask = predict_mask(image)
        overlay = overlay_mask(image, mask)

        st.image(
            overlay,
            caption="Tumor Segmentation Overlay",
            width=400
        )

        st.download_button(
            label="Download Result Image",
            data=convert_image(overlay),
            file_name="tumor_segmentation_overlay.png",
            mime="image/png"
        )
    else:
        st.success("No tumor detected. Segmentation skipped.")

