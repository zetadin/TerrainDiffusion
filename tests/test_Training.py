import torch
import glob
import os
import shutil as sh
from torchvision.transforms import v2
from src.DataSet.DataSet import TiledDataset
from src.Training.train import train
from src.Training.Model import UNetTiny_blocks

def test_Training():
    """
    Checks if training starts and checkpoint files are produced.
    """
    # create a dataset from the test data
    ds = TiledDataset("test_Data", tile_size=256, rotations=[0,15,30,45],
                       transform = v2.Compose([
                            v2.RandomHorizontalFlip(),
                            v2.RandomVerticalFlip(),
                            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                        ]))
    
    # find the mean and std of the dataset
    dataloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    all_tiles = next(iter(dataloader))
    mean = torch.mean(all_tiles).detach().item()
    std = torch.std(all_tiles).detach().item()
    
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

    # remove prior unit_test_log folder if it is there
    if os.path.exists("tb_logs/unit_test_log"):
        sh.rmtree("tb_logs/unit_test_log")

    # train the model
    cur_log_folder = train(
                           dataset=ds, invTrans=invTrans,
                           nEpochs=20, batch_size=8, nInfer=50,
                           blockSet = UNetTiny_blocks,
                           ls_scheduler_settings={"base_lr":5e-5, "max_lr":5e-5, "step_size_up":5},
                           grad_accumulate_schedule = {0: 4},
                           norm_num_groups = 16,
                           log_name = "unit_test_log",)
    
    # There should be 5 checkpoints now, 4 from saving every 5 epochs, and 1 from copy of last checkpoint
    chpt_files = f"{cur_log_folder}/checkpoints/*.ckpt"
    chpt_files = glob.glob(chpt_files)
    
    assert(len(chpt_files) == 5)