import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from src.models import SimpleCNN
from medmnist import PneumoniaMNIST


def apply_gradcam(model, img_tensor, target_layer):
    # Hooks to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    # Backward pass on the predicted class score
    model.zero_grad()
    loss = output[0, pred_class]
    loss.backward()

    # Compute Grad-CAM
    activ = activations[0].detach()[0]      # [C, H, W]
    grads = gradients[0].detach()[0]        # [C, H, W]
    weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
    cam = (weights * activ).sum(dim=0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam, pred_class.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM on PneumoniaMNIST')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='vis',
                        help='Directory to save Grad-CAM images')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of test samples to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_ds = PneumoniaMNIST(split='test', transform=transform, download=True)

    # Ensure output directory exists
    import os
    os.makedirs(args.output, exist_ok=True)

    # Process samples
    for idx in range(args.num_samples):
        img, label = test_ds[idx]
        img_tensor = img.unsqueeze(0).to(device)
        cam, pred = apply_gradcam(model, img_tensor, model.features[-3])

        # Create heatmap overlay
        heatmap = plt.get_cmap('jet')(cam)[..., :3]
        img_np = img.squeeze().cpu().numpy()
        overlay = (heatmap * 0.4 + img_np[..., None] * 0.6)

        # Save
        out_path = os.path.join(args.output, f'cam_{idx}_pred{pred}_true{label}.png')
        plt.imsave(out_path, overlay)
        print(f'Saved {out_path}')