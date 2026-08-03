import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = os.getenv('MODEL_NAME', 'Salesforce/blip-image-captioning-base')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading BLIP model on {DEVICE}...')
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print('Model loaded!')

class GradCAMViT:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = model.vision_model.encoder.layers[-1].layer_norm1
        layer.register_forward_hook(self._sa)
        layer.register_full_backward_hook(self._sg)
    def _sa(self, m, i, o): self.activations = o.detach()
    def _sg(self, m, gi, go): self.gradients = go[0].detach()
    def run(self, pv):
        self.model.zero_grad()
        pv = pv.to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            out = self.model.vision_model(pixel_values=pv)
            out.last_hidden_state[:, 0, :].sum().backward()
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))
        w = self.gradients.mean(dim=-1, keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=-1))[:, 1:]
        gs = int(cam.shape[1] ** 0.5)
        cam = F.interpolate(cam.reshape(1, 1, gs, gs), (224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min(): cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

gcam = GradCAMViT(model)

def predict(img):
    if img is None: return None, None, 'Upload an image'
    image = Image.fromarray(img).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    pv = inputs['pixel_values'].to(DEVICE)
    with torch.inference_mode():
        out = model.generate(pixel_values=pv, max_new_tokens=30, num_beams=5, early_stopping=True)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    heatmap = gcam.run(inputs['pixel_values'])
    img_r = image.resize((224, 224))
    arr = np.array(img_r) / 255.0
    overlay = np.clip(arr * 0.6 + plt.cm.jet(heatmap)[:,:,:3] * 0.4, 0, 1)
    return img_r, Image.fromarray((overlay * 255).astype(np.uint8)), caption

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label='Upload an Image'),
    outputs=[gr.Image(label='Original'), gr.Image(label='Grad-CAM Overlay'), gr.Textbox(label='Caption')],
    title='BLIP Image Captioning with Grad-CAM',
    description='Upload any image to get a caption. Red = important region, Blue = ignored.',
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
