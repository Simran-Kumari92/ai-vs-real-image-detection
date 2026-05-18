import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift

# =========================================================
# Multi-Signal Transform
# =========================================================

class MultiSignalTransform:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def __call__(self, pil_img):

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img_bgr, self.target_size)

        # RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_channels = img_rgb.astype(np.float32) / 255.0
        rgb_stack = np.transpose(rgb_channels, (2, 0, 1))

        # FFT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)

        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        fft_channel = cv2.normalize(
            magnitude_spectrum,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Noise Residual
        denoised = cv2.medianBlur(gray, 3)

        noise_residue = cv2.absdiff(gray, denoised)

        noise_channel = cv2.normalize(
            noise_residue,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # ELA
        _, buffer = cv2.imencode(
            ".jpg",
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        img_low = cv2.imdecode(buffer, 1)

        ela_diff = cv2.absdiff(img, img_low)

        ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)

        ela_channel = cv2.normalize(
            ela_gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Final tensor
        fusion_tensor = np.concatenate([
            rgb_stack,
            fft_channel[np.newaxis, ...] / 255.0,
            noise_channel[np.newaxis, ...] / 255.0,
            ela_channel[np.newaxis, ...] / 255.0
        ], axis=0)

        return (
            torch.from_numpy(fusion_tensor).float(),
            fft_channel,
            noise_channel,
            ela_channel
        )

# =========================================================
# Load Model
# =========================================================

device = torch.device("cpu")

model = models.resnet50()

model.conv1 = nn.Conv2d(
    6,
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

model.fc = nn.Linear(model.fc.in_features, 1)

checkpoint = torch.load(
    "models/ai_vs_real_resnet50.pth",
    map_location=device
)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# =========================================================
# Prediction Function
# =========================================================

def predict_image(image):

    transform = MultiSignalTransform()

    input_tensor, fft_img, noise_img, ela_img = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output).item()

    if probability > 0.5:
        label = "REAL IMAGE"
        confidence = probability * 100
        result = f"""
        <div style="
        padding:20px;
        border-radius:15px;
        background:#0f5132;
        color:white;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        ✅ REAL IMAGE<br>
        Confidence: {confidence:.2f}%
        </div>
        """
    else:
        label = "AI-GENERATED IMAGE"
        confidence = (1 - probability) * 100

        result = f"""
        <div style="
        padding:20px;
        border-radius:15px;
        background:#842029;
        color:white;
        text-align:center;
        font-size:24px;
        font-weight:bold;">
        🚨 AI-GENERATED IMAGE<br>
        Confidence: {confidence:.2f}%
        </div>
        """

    return result, fft_img, noise_img, ela_img

# =========================================================
# Custom CSS
# =========================================================

custom_css = """
body {
    background-color: #050816;
}

.gradio-container {
    background: #050816 !important;
    color: white;
}

h1 {
    text-align: center;
    font-size: 40px !important;
    color: cyan !important;
}

footer {
    visibility: hidden;
}
"""

# =========================================================
# UI
# =========================================================

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔍 AI vs Real Image Detection
        
        ### Multi-Signal Forensic Analysis using FFT, Noise Residual & ELA
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Image"
            )

            submit_btn = gr.Button("Analyze Image")

        with gr.Column():

            prediction_output = gr.HTML(
                label="Prediction"
            )

    gr.Markdown("## Forensic Feature Analysis")

    with gr.Row():

        fft_output = gr.Image(label="FFT Analysis")

        noise_output = gr.Image(label="Noise Residual")

        ela_output = gr.Image(label="ELA Analysis")

    submit_btn.click(
        fn=predict_image,
        inputs=input_image,
        outputs=[
            prediction_output,
            fft_output,
            noise_output,
            ela_output
        ]
    )

demo.launch()