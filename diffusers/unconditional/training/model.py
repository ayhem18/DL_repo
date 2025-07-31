from typing import Union
from diffusers.models.unets.unet_2d import UNet2DModel


from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim
from training.config import ModelConfig


def set_my_custom_model(config: ModelConfig) -> DiffusionUNetOneDim:

    model = DiffusionUNetOneDim(input_channels=config.input_shape[0],
                          output_channels=config.input_shape[0],
                          cond_dimension=256,                          
                          )
    
    # the default model reaches a training loss of 0.01 in less than 10 epochs
    # let's try to use a model with a similar capacity: it uses attention blocks which aren't currently implemented 
    # however, let's match the number of blocks + the number of channels. 
    model.build_down_block(num_down_layers=4, num_resnet_blocks=3, out_channels=[256, 512, 1024, 1024], downsample_types="conv")
    model.build_middle_block(num_resnet_blocks=3)
    model.build_up_block(num_resnet_blocks=3, upsample_types="transpose_conv")

    return model



def set_diffusers_model(config: ModelConfig) -> UNet2DModel:
    # let's if the model performs well without attention blocks
    model = UNet2DModel(
        sample_size=config.input_shape[1:],  # the target image resolution
        in_channels=config.input_shape[0],  # the number of input channels, 3 for RGB images
        out_channels=config.input_shape[0],  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D", 
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",             
        ), 
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D", 
            "UpBlock2D", 
            ),
        )

    return model



def set_model(config: ModelConfig) -> Union[DiffusionUNetOneDim, UNet2DModel]:
    if config.model_type == "custom_model":
        return set_my_custom_model(config)
    elif config.model_type == "diffusers":
        return set_diffusers_model(config)
    else:
        raise ValueError(f"Invalid model type: {config.model_type}") 

    