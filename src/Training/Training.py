import torch
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor
from lightning.pytorch.callbacks import GradientAccumulationScheduler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.Training.Model import TerGenUNet


def train(dataset, invTrans,
          nEpochs=1000, nInfer=50, nTrainIter=1000, ls_scheduler_settings = None,
          grad_accumulate_schedule = {0: 1, 100: 2, 150: 4},
          norm_num_groups=32,
          batch_size=8, blockSet=None, seed=59873,
          log_name="UNet_default", log_path="tb_logs"):
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
        norm_num_groups: number of channels in group normalization
        batch_size: batch size
        blockSet: specification of UNet layer types and numbers of output channels
        seed: seed for torch's global RNG
    """

    # trainer configs
    checkpoint_callback = ModelCheckpoint(monitor="val_MSE_epoch",
                                          save_last=True,
                                          filename='{epoch:06d}-{val_MSE_epoch:.4f}',
                                          save_top_k=5, every_n_epochs=5)
    
    logger = TensorBoardLogger(log_path, name=log_name)
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
                           invTrans=invTrans, norm_num_groups=norm_num_groups)
    else:
        down_block_types, up_block_types, block_out_channels = blockSet
        model = TerGenUNet(tile_size=256,  n_inferences=nInfer, n_iter_train=nTrainIter,
                           down_block_types = down_block_types,
                           up_block_types=up_block_types,
                           block_out_channels = block_out_channels,
                           ls_scheduler_settings=ls_scheduler_settings,
                           invTrans=invTrans, norm_num_groups=norm_num_groups)

    # run trainer
    trainer.fit(model, train_loader, val_loader)
    return(trainer.logger.log_dir)
