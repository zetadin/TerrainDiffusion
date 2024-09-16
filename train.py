#! /usr/bin/env python3

import os
import gc
import torch
from torchvision.transforms import v2
from src.Training.Training import train
from src.Training.Model import UNetMini_blocks, UNetTiny_blocks
from src.DataSet.DataSet import TiledDataset


# entry point for training run
if __name__ == "__main__":

    # find the root directory of the project so we can find the data
    prj_root =  os.path.dirname( os.path.abspath(__file__))

    # load the dataset
    ds = TiledDataset(
                  os.path.join(prj_root , "Data" ),
                  tile_size=256, rotations=[0,15,30,45],
                  transform = v2.Compose([
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                    ]))
    print("Dataset size:", len(ds))
    
    # find the mean and std of the dataset
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_tiles = next(iter(dataloader))
    mean = torch.mean(all_tiles).detach().item()
    std = torch.std(all_tiles).detach().item()

    print(f"{mean=}")
    print(f"{std=}")
    print(f"min={torch.min(all_tiles)}")
    print(f"max={torch.max(all_tiles)}")

    # free memory
    del dataloader, all_tiles
    _ = gc.collect()

    # what should the std be?
    target_std = 0.5

    # Normalize the dataset
    ds.transform = v2.Compose([
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                        v2.Normalize(mean=[mean], std=[std/target_std]),
                    ])
    
    # build a reverse transform that un-normalizes the generated images
    invTrans = v2.Compose([
        v2.Normalize(mean = [0.], std = [target_std/std]),
        v2.Normalize(mean = [-mean], std=[1.])
    ])



    # lr scheduler settings
    ls_scheduler_settings = {"base_lr":5e-5, "max_lr":5e-5, "step_size_up":5}

    # train the model
    cur_log_folder = train(dataset=ds, invTrans=invTrans,
                           nEpochs=30, batch_size=8, nInfer=50,
                           ls_scheduler_settings=ls_scheduler_settings,
                           grad_accumulate_schedule = {0: 1, 10: 2, 20: 4},
                           log_path = os.path.join( prj_root, "Training_logs"),

                           blockSet = UNetTiny_blocks,  norm_num_groups=16,
                           log_name = "Unet_Tiny",
                           # blockSet=UNetMini_blocks,  norm_num_groups=32,
                           # log_name = "Unet_Mini",
                           )
    
    print(f"\n\nDone.")
    exit(0)