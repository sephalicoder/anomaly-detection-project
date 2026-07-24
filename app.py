"""
Interactive UI application using Gradio to demonstrate real-time out-of-distribution
anomaly detection and spatial defect localization for production deployment.
"""
import sys
import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Force the absolute path of the local src directory into the Python path runtime
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import ConvAutoencoder

# Target operational boundaries calculated from your validation benchmarks
THRESHOLDS = {
    "bottle": 0.000900,
    "screw": 0.001215
}

def predict_anomaly(img, category):
    if img is None:
        return "Please upload an image.", None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load corresponding model architecture weights
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", f"{category}_autoencoder.pt")
    if not os.path.exists(ckpt_path):
        return f"❌ Checkpoint not found for {category}. Please train the model first.", None
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Preprocessing transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Convert incoming PIL image safely to tensor format
    orig_tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        recon_tensor, _ = model(orig_tensor)
        
    # Isolate tensors to generate absolute pixel error metrics
    orig_np = orig_tensor.squeeze(0).cpu().numpy()
    recon_np = recon_tensor.squeeze(0).cpu().numpy()
    
    # Compute the channel-averaged 2D spatial pixel error map inline
    error_map = np.mean((orig_np - recon_np) ** 2, axis=0)
    image_score = float(np.mean(error_map))
    
    # Apply threshold binary sorting boundary condition
    thresh = THRESHOLDS.get(category, 0.001)
    is_anomaly = image_score >= thresh
    status = "🚨 ANOMALY DETECTED" if is_anomaly else "✅ NORMAL PASS"
    verdict_text = f"Result: {status}\nAnomaly Score: {image_score:.6f} (Threshold: {thresh:.6f})"
    
    # Render localized visual error overlay map matching visualize.py
    fig, ax = plt.subplots(figsize=(4, 4))
    display_orig = np.transpose(orig_np, (1, 2, 0))
    ax.imshow(display_orig)
    ax.imshow(error_map, cmap="jet", alpha=0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    # Export structural canvas into application return buffer
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())   # returns (H, W, 4) RGBA array
    output_heatmap = Image.fromarray(buf).convert("RGB")
    plt.close(fig)
    
    return verdict_text, output_heatmap

# --- Build Gradio Interactive Workspace Blocks Interface ---
with gr.Blocks(title="Unsupervised Anomaly Inspection System") as demo:
    gr.Markdown("# 👁️ Unsupervised Industrial Quality Control Engine")
    gr.Markdown("Upload an asset image from the MVTec-AD test split to localize anomalies via structural reconstruction metrics.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Inspection Sample Input")
            category_dropdown = gr.Dropdown(choices=["bottle", "screw"], value="bottle", label="Target Pipeline Category Structure")
            submit_btn = gr.Button("Execute Anomaly Scan", variant="primary")
            
        with gr.Column():
            output_verdict = gr.Textbox(label="Decision Boundary Verdict Analysis", interactive=False)
            output_map = gr.Image(label="Pixel-level Defect Localization Overlay")
            
    submit_btn.click(
        fn=predict_anomaly, 
        inputs=[input_image, category_dropdown], 
        outputs=[output_verdict, output_map]
    )

if __name__ == "__main__":
    demo.launch()