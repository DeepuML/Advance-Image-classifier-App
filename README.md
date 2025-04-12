# 🧠 Advanced Hugging Face Image Classifier 🚀

This Streamlit app allows you to classify images using state-of-the-art Vision Transformers (ViT) from Hugging Face!  
It also supports **Grad-CAM visualization**, **image similarity comparison**, and **real-time webcam capture**.

---

## 💡 Features

✅ **Model Selection**  
Choose from powerful pre-trained models:
- ViT Base (ImageNet)
- ViT Large (ImageNet)
- DeiT Base (ImageNet)

✅ **Image Upload or Webcam Support**  
- Upload JPG/PNG images or use your webcam for real-time classification.

✅ **Top-5 Predictions**  
- View model predictions with confidence scores.

✅ **Grad-CAM Visualization**  
- Interpret model decisions using Grad-CAM heatmaps.

✅ **Image Similarity Checker**  
- Compare two images using feature extraction and cosine similarity.

---

## 🖼️ Screenshots

| Classification | Grad-CAM | Similarity Check |
|-----------------|----------|-------------------|
| ![Classification Screenshot](your_screenshot_1.png) | ![GradCAM Screenshot](your_screenshot_2.png) | ![Similarity Screenshot](your_screenshot_3.png) |

---

## ⚙️ Setup Instructions

1️⃣ **Clone the Repository**
``bash
git clone https://github.com/DeepuML/Advance-Image-classifier-App.git
cd YourRepo

pip install -r requirements.txt
streamlit run your_script_name.py


💻 Dependencies
streamlit

transformers

torch

pillow

numpy

matplotlib

scikit-learn

opencv-python

streamlit-webrtc

pytorch-grad-cam

📸 Webcam Support
This app integrates streamlit-webrtc to allow live image capture directly from your webcam!

🧠 Model Sources
Models are downloaded directly from Hugging Face Model Hub:

google/vit-base-patch16-224

google/vit-large-patch16-224

facebook/deit-base-patch16-224

🤝 Contributing
Contributions are welcome!
Please open issues, pull requests, or suggestions.







