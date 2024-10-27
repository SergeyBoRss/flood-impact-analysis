import os
import torch
import torch.nn as nn
import rasterio
import tifffile as tiff
from torchvision import models
import numpy as np
import argparse

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return (band - band_min) / (band_max - band_min)

def infer_image(image_path, model):
    with rasterio.open(image_path) as src:
        print(f'Обработка {image_path}')

        existing_bands = [normalize(src.read(i + 1).astype(float)) for i in range(src.count)]
        
        green_band = existing_bands[1]  # Зеленый канал (B03)
        swir_band = existing_bands[8]    # SWIR канал (B11)

        mndwi = (green_band - swir_band) / (green_band + swir_band + 1e-10)

        stacked_bands = np.stack(existing_bands + [normalize(1 + mndwi)], axis=0)

    image = torch.tensor(stacked_bands, dtype=torch.float32)
    
    if len(image.shape) == 3:
        image = image.permute(0, 2, 1)  # (каналы, высота, ширина)

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)['out']

    output = output[0].cpu().detach().numpy()
    output = torch.sigmoid(torch.tensor(output)).numpy()
    output = output.squeeze()
    
    return output

def load_model(model_path):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.backbone.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def save_output(output, output_path, threshold):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    binary_mask = (output > threshold).astype(np.uint8)
    flipped_mask = np.fliplr(binary_mask)
    rotated_mask = np.rot90(flipped_mask, k=1)

    tiff.imsave(output_path, rotated_mask)

def process_image(image_path, output_path, model, threshold=0.08):
    output = infer_image(image_path, model)
    save_output(output, output_path, threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка TIF изображения и сохранение маски.')
    parser.add_argument('image_path', type=str, help='Путь к входному 10 канальному TIF изображению')
    parser.add_argument('output_path', type=str, help='Путь для сохранения бинарной маски')
    parser.add_argument('--threshold', type=float, default=0.08, help='Порог для бинаризации маски (по умолчанию 0.2)')
    parser.add_argument('--pth', type=str, help='Путь до весов модели')
    
    args = parser.parse_args()

    model = load_model("deeplabv3_trained_epoch_30_2.pth")
    process_image(args.image_path, args.output_path, model, args.threshold)
