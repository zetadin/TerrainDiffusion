from diffusers import DDPMScheduler
import torch

class MyDDPMSCheduler(DDPMScheduler):
    def match_shape(self,
                    values: "Union[np.ndarray, torch.Tensor]",
                    broadcast_array: "Union[np.ndarray, torch.Tensor]"):
        """
        Turns a 1-D array into an array or tensor with len(broadcast_array.shape) dims.
        Copied from the main branch of the diffusers library.

        Args:
            values: an array or tensor of values to extract.
            broadcast_array: an array with a larger shape of K dimensions with the batch
                dimension equal to the length of timesteps.
        Returns:
            a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        tensor_format = getattr(self, "tensor_format", "pt")
        values = values.flatten()

        while len(values.shape) < len(broadcast_array.shape):
            values = values[..., None]
        if tensor_format == "pt":
            values = values.to(broadcast_array.device)

        return values
        
    def single_iter_predict_original(self,
        model_output: "Union[torch.FloatTensor, np.ndarray]",
        timesteps: "Union[torch.IntTensor, np.ndarray]",
        sample: "Union[torch.FloatTensor, np.ndarray]",
        predict_epsilon=True,) -> torch.FloatTensor:
        """
        In a single step construct a generated image from inferred noise and the noised input image.
        Adapted from DDPMScheduler.step() version in the main branch of the diffusers library.
        Args:
            model_output: noise tensor predicted by the diffusion model.
            timesteps: timestep tensor indicating how much noise is mixed in.
            sample: the original image(s) with noise added.
            predict_epsilon: if True, model_output is the prediction for the original image instead of the noise.
        Returns:
            Tensor for the reconstructed de-noised original image(s).
        """

        # 1. compute alphas, betas
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = self.match_shape(sqrt_alpha_prod, sample)
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self.match_shape(sqrt_one_minus_alpha_prod, sample)

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if predict_epsilon:
            pred_original_sample = (sample - sqrt_one_minus_alpha_prod * model_output) / sqrt_alpha_prod
        else:
            pred_original_sample = model_output

        return(pred_original_sample)