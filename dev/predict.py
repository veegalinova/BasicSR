from pathlib import Path

import yaml
import torch
import numpy as np
import tifffile

from tqdm import tqdm

from rcan import RCAN_custom


def load_model_and_config(model_path: Path, checkpoint='latest'):
    """Load the model from checkpoint with given network configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(model_path)
    config_path = model_path / 'config.yml'
    checkpoint_path = model_path / "models" / f'net_g_{checkpoint}.pth'

    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)
    network_config = config['network_g']
    network_config.pop('type')

    model = RCAN_custom(**network_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict["params"], strict=True)
    model.eval().to(device)
    return model, config


@torch.no_grad()
def predict_image(
    input, model, input_stats, target_stats
):
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
    model_path = Path("/group/jug/Vera/dev/BasicSR/experiments/RCAN_biosr_Factin/")
    model, config = load_model_and_config(model_path)

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

        result = predict_image(
            batch,
            model,
            input_stats,
            target_stats
        )
