import torch
import os
import lightning as L
import torchmetrics
import torchvision
from torchvision.transforms import v2
from diffusers import UNet2DModel
from diffusers.utils import make_image_grid
from torch import nn
from src.Training.NoiseScheduler import MyDDPMSCheduler

class TerGenUNet(L.LightningModule):
    def __init__(self, tile_size=256,  n_inferences=25, n_iter_train=1000,
                 down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                    ),
                 up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    ),
                 block_out_channels=(64, 64, 128, 128, 256, 256), # the number of output channels for each UNet block
                 norm_num_groups=32,
                 attention_head_dim=8,
                 ls_scheduler_settings = {"base_lr":5e-5, "max_lr":1e-3,
                                          "step_size_up":20},
                 invTrans = v2.Compose([v2.Normalize(mean = [0.], std = [1.]), v2.Normalize(mean = [-0.], std=[1.])])
                ):
        """
        Args:
            tile_size: the size of the tiles
            n_inferences: number of inference steps
            n_iter_train: number of iterations in training
            down_block_types, up_block_types: names of UNet layer types passed to diffusers.UNet2DModel
            block_out_channels: number of output channels for each UNet block
            ls_scheduler_settings: dict of settings for the torch.optim.lr_scheduler.CyclicLR learning rate scheduler
            attention_head_dim: number of attention head dimensions
            norm_num_groups: number of channels in group normalization in the UNet
            """
        super().__init__()
        self.save_hyperparameters(ignore=['invTrans'])

        self.n_inferences = n_inferences
        self.n_iter_train = n_iter_train
        self.ls_scheduler_settings = ls_scheduler_settings
        self.tile_size = tile_size
        self.invTrans = invTrans


        # metrics
        self.train_MSE = torchmetrics.regression.MeanSquaredError()
        self.val_MSE = torchmetrics.regression.MeanSquaredError()

        # UNet
        self.unet = UNet2DModel(
            sample_size=tile_size,
            in_channels=1, out_channels=1,  # the number of I/O channels: monochrome
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels = block_out_channels, 
            down_block_types = down_block_types,
            up_block_types = up_block_types,
            norm_num_groups = norm_num_groups,
            attention_head_dim = attention_head_dim
        )

        # noise scheduler & pipelines
        self.noise_scheduler = MyDDPMSCheduler(num_train_timesteps=self.n_iter_train)


    def forward(self, noised, inference_amounts):
        """
        Perform a single iference on the noised data.
        """
        o = self.unet(noised, inference_amounts).sample
        return o
           

    def training_step(self, inp, batch_idx):
        """
        Execute a step of training.
        """
        with torch.no_grad(): # no need for gradients on input data
            bs = inp.shape[0]
            noise = torch.randn(inp.shape, device = inp.device)
    
            # random number of inferences to perform on each image
            inference_amounts = torch.randint(0, self.n_iter_train, (bs,),
                                     device=inp.device, dtype=torch.int64)
            
            noised_inp = self.noise_scheduler.add_noise(inp, noise, inference_amounts)

        out = self.forward(noised_inp, inference_amounts)
        
        # compute and log loss
        loss = nn.functional.mse_loss(noise,out)
        self.log('loss', loss)

        # compute metrics
        self.train_MSE(noise,out)

        return loss
        
        
    def on_train_epoch_end(self):
        """
        Runs at the end of a training epoch.
        Logs the training metrics to TensorBoard.
        """
        # log epoch metrics
        self.log('train_MSE_epoch', self.train_MSE.compute())
        self.train_MSE.reset()
        
    
    def sample(self, N, seed = None, n_inference_iter=None, use_default_gen=False):
        """
        Generate some images.
        Args:
            N: number of images to generate
            seed: seed for the random number generator
            n_inference_iter: number of inference steps to execute
            use_default_gen: if True, use the default generator, otherwise use the provided seed
        """
        with torch.no_grad():
            if(not use_default_gen):
                gen = torch.Generator(device = self.device)
                if(seed is not None):
                    gen.manual_seed(seed)
            else:
                gen=None

            if(n_inference_iter is None):
                n_inference_iter = self.n_inferences

            # next part adapted from DDPMPipeline to output raw tensors, not normalized PILs or ndarrays
            # Sample gaussian noise to begin loop
            images = torch.randn((N, 1, self.tile_size, self.tile_size),
                                 generator=gen, device = self.device)    
            # set step values
            self.noise_scheduler.set_timesteps(n_inference_iter)
    
            for t in self.noise_scheduler.timesteps:
                # 1. predict noise model_output
                model_output = self.unet(images, t).sample
    
                # 2. compute previous image: x_t -> t_t-1
                images = self.noise_scheduler.step(model_output, t, images, generator=gen).prev_sample
        return(images)
        

    def validation_step(self, inp):
        """
        Execute a step of validation.
        """
        # this is the validation loop
        with torch.no_grad(): # no need for gradients on input data
            bs = inp.shape[0]
            noise = torch.randn(inp.shape, device = inp.device)
    
            # random number of inferences to perform on each image
            inference_amounts = torch.randint(0, self.n_iter_train, (bs,),
                                     device=inp.device, dtype=torch.int64)
            
            noised_inp = self.noise_scheduler.add_noise(inp, noise, inference_amounts)

            out = self.forward(noised_inp, inference_amounts)
        
            # compute metrics
            self.val_MSE(noise,out)
            
    
    def on_validation_epoch_end (self):
        """
        Runs at the end of a validation epoch.
        Logs the validation metrics and sample images to TensorBoard.
        """
        # log epoch metrics
        self.log('val_MSE_epoch', self.val_MSE.compute())
        self.val_MSE.reset()

        
        if(self.trainer.current_epoch%5 == 4 # will match checkpointing frequency in training
          ):
            outs = self.invTrans(self.sample(4, seed=268711))
            images = [v2.functional.to_pil_image(outs[i,:,:]) for i in range(outs.shape[0])]
            image_grid = make_image_grid(images, rows=2, cols=2)

            # Save a sample of generated images
            test_dir = os.path.join(self.trainer.logger.log_dir, "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save(os.path.join(test_dir, f"gen_{self.trainer.current_epoch:06d}.png"))
            
            # also draw validation set examples
            inp = next(iter(self.trainer.val_dataloaders)).to(self.device)
            bs = inp.shape[0]
            noise = torch.randn(inp.shape, device = inp.device)
            # Use random timestep for the single inference to perform on each image
            inference_amounts = torch.randint(0, self.n_iter_train, (bs,),
                                     device=inp.device, dtype=torch.int64)
            noised_inp = self.noise_scheduler.add_noise(inp, noise, inference_amounts)
            out = self.forward(noised_inp, inference_amounts)
            pred_original = self.noise_scheduler.single_iter_predict_original(out, inference_amounts, noised_inp)
            
            # log them to TensorBoard
            imgs_tensor = torch.cat([self.invTrans(noised_inp), self.invTrans(inp), self.invTrans(pred_original)])
            tv_image_grid = torchvision.utils.make_grid(imgs_tensor, nrow=bs)
            self.logger.experiment.add_image('validation_images', tv_image_grid, self.trainer.global_step, dataformats="CHW")
            
            

    
    def configure_optimizers(self):
        """
        Setup the learning rate scheduler and optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                                     optimizer,
                                     base_lr = self.ls_scheduler_settings["base_lr"],
                                     max_lr = self.ls_scheduler_settings["max_lr"],
                                     cycle_momentum=False, 
                                     step_size_up = self.ls_scheduler_settings["step_size_up"]),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_MSE_epoch",
            },
        }
    



### UNet Architecture Hyperparameter Sets ###
UNetTiny_blocks = (
    (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    (
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    (16, 16, 32, 32) # requires norm_num_groups = 16 or less
)

UNetMini_blocks = (
    (
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    (32, 32, 32, 64, 64)
)

UNetSmall_blocks = (
    (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    (32, 64, 64, 64, 128, 128)
)