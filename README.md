# Brain Tumor Classification and Segmentation

This project is an end-to-end medical imaging application that detects and localizes brain tumors from MRI scans using deep learning.

## Features
- Tumor classification into:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Pixel-level tumor segmentation using U-Net
- Confidence scores and probability distribution
- Visual overlay of tumor region
- Streamlit web interface
- Supports real MRI image uploads

## Models Used
- **Classification:** MobileNetV2 (PyTorch)
- **Segmentation:** U-Net (TensorFlow/Keras)

## Workflow
1. MRI image uploaded by user
2. Image classified using MobileNetV2
3. If tumor detected → segmentation model runs
4. Tumor mask is overlaid on MRI
5. Result displayed and downloadable

## Tech Stack
- PyTorch
- TensorFlow / Keras
- Streamlit
- NumPy
- PIL

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
