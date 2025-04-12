import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Page setup
st.set_page_config(page_title="🔍 Advanced Image Classifier", layout="centered")
st.title("🧠 Advanced Hugging Face Image Classifier")


# ----------------------
# Model Loader
# ----------------------
model_options = {
    "ViT Base (ImageNet)": "google/vit-base-patch16-224",
    "ViT Large (ImageNet)": "google/vit-large-patch16-224",
    "DeiT Base (ImageNet)": "facebook/deit-base-patch16-224"
}
selected_model = st.selectbox("Select a model", list(model_options.keys()))

@st.cache_resource
def load_model(model_name):
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, processor

model_name = model_options[selected_model]
model, processor = load_model(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ----------------------
# Classify Image
# ----------------------
def classify(image, model, processor, top_k=5):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits[0], dim=0)
    top_probs, top_idxs = torch.topk(probs, top_k)
    labels = [model.config.id2label[i.item()] for i in top_idxs]
    return list(zip(labels, top_probs.tolist())), outputs


# ----------------------
# Grad-CAM
# ----------------------
def generate_gradcam(image, model, processor, outputs):
    inputs = processor(images=image, return_tensors="pt").to(device)
    input_tensor = inputs["pixel_values"]
    target_layer = model.vit.encoder.layer[-1].output
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == "cuda"))
    targets = [ClassifierOutputTarget(outputs.logits.argmax().item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Prepare normalized image for overlay
    image_np = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)


# ----------------------
# Feature Extractor for Similarity
# ----------------------
def extract_features(image, model, processor):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.vit(**inputs).last_hidden_state[:, 0]
    return features.cpu().numpy()


# ----------------------
# Webcam Capture
# ----------------------
st.subheader("📸 Capture from Webcam")
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None
    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

ctx = webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

webcam_image = None
if ctx.video_transformer and ctx.video_transformer.frame is not None:
    st.success("✅ Webcam frame captured! Click below to use it.")
    if st.button("Use Current Frame"):
        webcam_image = Image.fromarray(cv2.cvtColor(ctx.video_transformer.frame, cv2.COLOR_BGR2RGB))


# ----------------------
# Image Uploader
# ----------------------
st.subheader("📁 Or Upload an Image")
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
input_image = None

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
elif webcam_image:
    input_image = webcam_image

if input_image:
    st.image(input_image, caption="🖼️ Input Image", use_column_width=True)

    # Run classification
    with st.spinner("🔍 Classifying..."):
        results, outputs = classify(input_image, model, processor)

    st.subheader("🔎 Top Predictions")
    for label, prob in results:
        st.write(f"**{label}** — {prob*100:.2f}%")

    # Plot bar chart
    labels, probs = zip(*results)
    chart_data = pd.DataFrame({'Label': labels, 'Confidence': [p * 100 for p in probs]})
    st.bar_chart(chart_data.set_index('Label'))

    # Grad-CAM
    if st.checkbox("🧠 Show Grad-CAM"):
        cam_img = generate_gradcam(input_image, model, processor, outputs)
        st.image(cam_img, caption="Grad-CAM Heatmap", use_column_width=True)

    # Similarity Checker
    st.subheader("🔁 Image Similarity Checker")
    sim_file = st.file_uploader("Upload a second image to compare", type=["jpg", "jpeg", "png"], key="sim")
    if sim_file:
        sim_image = Image.open(sim_file).convert("RGB")
        st.image(sim_image, caption="🔁 Comparison Image", width=300)

        vec1 = extract_features(input_image, model, processor)
        vec2 = extract_features(sim_image, model, processor)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        st.success(f"✅ Cosine Similarity: {similarity:.4f}")

