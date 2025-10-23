from pathlib import Path

import yaml
import torch
import numpy as np
import tifffile

from tqdm import tqdm

from esrgan import RRDBNet


@torch.no_grad()
def predict_image_esrgan(
    input, model, input_stats, target_stats
):
    """Predict using ESRGAN model with proper normalization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_mean, input_std = input_stats
    target_mean, target_std = target_stats

    input = input.astype(np.float32)
    normalized_input = (input - input_mean) / input_std

    # Handle single image vs batch
    if len(input.shape) == 2:  # Single image
        x = torch.from_numpy(normalized_input).unsqueeze(0).unsqueeze(0)
    else:  # Batch of images
        x = torch.from_numpy(normalized_input).unsqueeze(1)
    x = x.to(device)

    y = model(x)
    y = y.cpu().numpy()

    # Denormalize
    y = y * target_std + target_mean
    y = y.squeeze()

    return y


if __name__ == "__main__":
    model_path = Path("/group/jug/Vera/dev/BasicSR/experiments/ESRGAN_biosr_er/")
    model, config = load_esrgan_model_and_config(model_path)

    image_folder = config['datasets']['val']['dataroot']
    images = sorted(Path(image_folder).glob('*.tif'))

    input_stats = config['datasets']['train']['input_mean'], \
        config['datasets']['train']['input_std']
    target_stats = config['datasets']['train']['target_mean'], \
        config['datasets']['train']['target_std']

    batch_size = 8
    for i in tqdm(range(0, len(images), batch_size)):
        batch = []
        batch_images = images[i:i + batch_size]

        for image_path in batch_images:
            image = tifffile.imread(image_path)
            high_res, low_res = image
            batch.append(low_res)
        batch = np.array(batch)

        result = predict_image_esrgan(
            batch,
            model,
            input_stats,
            target_stats
        )
