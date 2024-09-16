import torch
import sys
import os
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import v2
from src.Training.Model import TerGenUNet, UNetMini_blocks
from src.DataSet.DataSet import TiledDataset


def train(dataset, invTrans,
          nEpochs=1000, nInfer=50, nTrainIter=1000, ls_scheduler_settings = None,
          grad_accumulate_schedule = {0: 1, 100: 2, 150: 4},
          batch_size=8, blockSet=None, seed=59873):
    """
    Runs the training.

    Args:
        dataset: the dataset to be split into training and validation subsets
        invTrans: the inverse transform that de-normalizes the images
        nEpochs: number of epochs
        nInfer: number of steps during image graneration.
        nTrainIter: number of timesteps during training.
        ls_scheduler_settings: dict of settings for the torch.optim.lr_scheduler.CyclicLR learning rate scheduler
        grad_accumulate_schedule: dict containing epochs and number of batches for gradient accumulation
        batch_size: batch size
        blockSet: specification of UNet layer types and numbers of output channels
        seed: seed for torch's global RNG
    """

    # trainer configs
    checkpoint_callback = ModelCheckpoint(monitor="val_MSE_epoch",
                                          filename='{epoch:06d}',
                                          save_top_k=5, every_n_epochs=5)
    
    logger = TensorBoardLogger("tb_logs", name="UNet_")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accumulator = GradientAccumulationScheduler(scheduling=grad_accumulate_schedule)
    trainer = L.Trainer(limit_train_batches=1.0,
                        max_epochs=nEpochs,
                        logger=logger,
                        log_every_n_steps=10,
                        accelerator="gpu",
                        check_val_every_n_epoch = 1,
                        callbacks=[TQDMProgressBar(refresh_rate=batch_size), lr_monitor, checkpoint_callback, accumulator])
    
    print("batch_size:", batch_size)
    print("Logging to:", trainer.logger.log_dir)

    # reproducibility
    torch.manual_seed(seed)

    # dataloader
    g = torch.Generator()
    g.manual_seed(seed + 791423)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05], generator=g)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, generator=g, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, generator=g, shuffle=False)

    # create model
    if(blockSet is None):
        model = TerGenUNet(tile_size=256,  n_inferences=nInfer, n_iter_train=nTrainIter,
                           ls_scheduler_settings=ls_scheduler_settings,
                           invTrans=invTrans)
    else:
        down_block_types, up_block_types, block_out_channels = blockSet
        model = TerGenUNet(tile_size=256,  n_inferences=nInfer, n_iter_train=nTrainIter,
                           down_block_types = down_block_types,
                           up_block_types=up_block_types,
                           block_out_channels = block_out_channels,
                           ls_scheduler_settings=ls_scheduler_settings,
                           invTrans=invTrans)

    # run trainer
    trainer.fit(model, train_loader, val_loader)
    return(trainer.logger.log_dir)





# entry point for training run
if __name__ == "__main__":

    # find the root directory of the project so we can find the data
    prj_root = os.path.join( os.path.abspath(__file__), '..', '..' )

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
    cur_log_folder = train(nEpochs=30, batch_size=8, nInfer=50,
                           blockSet=UNetMini_blocks,
                           ls_scheduler_settings=ls_scheduler_settings,
                           grad_accumulate_schedule = {0: 1, 10: 2, 20: 4})
    
    print(f"\n\nDone.")
    exit(0)