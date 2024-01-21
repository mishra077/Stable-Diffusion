# STABLE DIFFUSION FROM SCRATCH 


### Reason for using VAE Attention Block
We have used Convoltution Block for calculating the close relationship between each pixels locally i.e. the kernel size determines which group of pixels will be dependent with each other. But in the case of Attention Block, all pixels are attended with each other globally. This will help the model to understand if there is a corelation between pixels which were not the part of the same group or kernel.