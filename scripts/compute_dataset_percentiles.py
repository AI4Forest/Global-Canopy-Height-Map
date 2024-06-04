import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
import sys
# Assuming PreprocessedSatelliteDataset is defined in your project
from config import PreprocessedSatelliteDataset
from runner import Runner

def update_extremes(values, extremes, num_extremes, largest=True):
    """
    Update the list of extreme values (either largest or smallest) based on the new batch.
    """
    combined = torch.cat((extremes, values))
    sorted_values, _ = torch.sort(combined, descending=largest)
    return sorted_values[:num_extremes]

def compute_percentiles(dataset_name, split, percentiles, num_workers_default=4):
    # Set up dataset and DataLoader
    rootPath = Runner.get_dataset_root(dataset_name=dataset_name)
    dataframe = os.path.join(rootPath, f'{split}.csv')

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PreprocessedSatelliteDataset(data_path=rootPath, dataframe=dataframe,
                                           image_transforms=train_transforms,
                                           use_weighted_sampler=None, use_memmap=True)
    total_data_points = len(dataset)
    num_channels = 14

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = num_workers_default * torch.cuda.device_count()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Initialize percentile tracking
    extremes = {channel: {p: torch.tensor([]).to(device) for p in percentiles} for channel in range(num_channels)}

    # Process each batch
    with torch.no_grad():
        for data, _ in tqdm(dataloader):
            data = data.to(device=device, non_blocking=True)
            # Switch the channel dimension to the first dimension, currently its at dim 1
            data = data.permute(1, 0, 2, 3)
            # Flatten the data
            data = data.flatten(start_dim=1)

            for channel in range(num_channels):
                channel_data = data[channel, :]

                for percentile in percentiles:
                    if percentile < 50:
                        num_extremes = int(total_data_points * percentile / 100)
                        largest = False
                    else:
                        num_extremes = int(total_data_points * (100 - percentile) / 100)    # E.g. if percentile == 95, we look at the 5 percentile from the other side
                        largest = True
                    current_extremes = extremes[channel][percentile]
                    new_extremes = update_extremes(values=channel_data, extremes=current_extremes, num_extremes=num_extremes, largest=largest)
                    extremes[channel][percentile] = new_extremes

    # Compute final percentile values
    percentile_values = {channel: {} for channel in range(num_channels)}
    for channel in range(num_channels):
        for percentile in percentiles:
            if percentile > 50:
                percentile_values[channel][percentile] = extremes[channel][percentile].min().item()
            else:
                percentile_values[channel][percentile] = extremes[channel][percentile].max().item()

    # Save results
    dump_path = os.path.join(os.getcwd(), f'{dataset_name}_{split}_percentiles.txt')
    with open(dump_path, 'w') as f:
        for percentile in percentiles:
            percentile_values_for_all_channels = tuple(percentile_values[channel][percentile] for channel in percentile_values)
            f.write(f'{percentile}: {percentile_values_for_all_channels},\n')


    return percentile_values

# Usage example
percentiles = [1, 2, 5, 95, 98, 99]
dataset_name = 'ai4forest_camera'
split = 'train'
percentile_values = compute_percentiles(dataset_name, split, percentiles)
