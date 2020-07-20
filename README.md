# Neural Style Transfer for US Imaging

## Repository tree

- img/ ---- input images, datasets, etc...
- output/ ---- gpu or cpu outputs
- models/ ---- output models from the perceptual loss network (.model), dictionaries of the average styles (.pickle)
- nst/ ---- neural style transfer with optimization approach, in tensorflow
- perc_loss/ ---- neural style transfer with learning approach, in pytorch


## Setup

First of all, run setup.sh to build the repository tree and the conda environments.
Three environments should be created:
1. tfenv has Tensorflow 2.1
2. tencu has Tensorflow 2.1 and cudnn 7.6.5
3. pytcu has Pytorch 1.4 

Then, manually copy the simulated and clinical datasets into img directory.

The optimization approach wors with image directly loaded from the data dir and then cropped; instead the learning approach needs to have the images already cropped when loaded.
This operation is performed by build_datasets.py.


## Execution

### Optimization approach

In order to run this approach, an environment with Tensorflow 2.1 must be activated.

To run nst.py, some arguments need to be passed to the script. The default runs the LQ to HQ task with content 34.png and style 645.png, optimizing for 3000 iterations on cpu.

The scripts eval_nst.sh allows to run the various tasks on gpu, while eval_local.sh allows to run local style transfer.
Some variables in these scripts set the input/output paths:
- name = ETH user name (e.g. dmenini)
- machine = name of the working station (e.g. hoss)
- project = name of the directory contianing the project (e.g. nst-for-us-imaging)
- env = conda environment to be activated (e.g. tencu)
Other variables determine how the NST happens:
- task = either 'lq2hq', 'seg2hq', 'hq2clinical'
- content = content image number in an array (e.g. (34))
- style = style image number in an array (e.g. (645))
- dict = dictionary name for average style loss (e.g. us_hq_ft_dict.pickle)
- script (nst.py or other modifed versions)

The bash scripts prepare the environment for the GPU and run the python scripts passing the desired arguments. Anyway, some may still be set manually, like loss, epochs, weights and learning rate.
Here there is an example command to run LQ to HQ nst for 3000 iterations (300x10) with basic style loss:

qsub eval_nst.sh --epochs 300 --lr 0.5 --weights 0.01 100000 --loss 0 --name out --message "example run"

The argument "name" is the prefix of the output images. It is useful when different outputs from the same images are produced (e.g. with different losses or parameters) in order to not overwrite the output.
The argument "message" is printed at the beginning of the log and is helpful to write down the details of the run (e.g loss used, weights, etc)

The script eval_local allows to run local_nst.py, which performs local style transfer. It works similarly as the others.

The average style, in order to be used, has to be created first. This can be done with create_avg_style.py. 

### Learning approach

In order to run this approach, an environment with Pytorch must be activated.

To run training.py, the script eval_training.sh must be used. Again, some environment variables need to be set, like name, machine, etc.
Moreover, some arguments can be passed to control the training, e.g. batch-size, epochs, weights.

Here there is an example command to run the training for 5 epochs:

qsub eval_training.sh --epochs 5 --weights 5000000 100 --loss 0 --model-name lq2hq.model


To run transfer.py, the script eval_transfer.sh must be used. The images to be transfered have to be indicated directly inside the python program (much faster).
The bash script provides only the paths of the content (to be modified accordingly) and output directories.

Here there is an example command to run the training for 5 epochs:

qsub eval_transfer.sh --model-name lq2hq.model

To evaluate SSIM and PSNR (average values) of the images obtained by style transfer on the test set, run eval_score.py.
To evaluate FID of the images obtained by style transfer on the test set, run eval_fid.py.
They work in a tensorflow environment.

The average style, in order to be used, has to be created first. This can be done with create_avg_style.py. The pytorch implementation is noticeably slower than the Tensrflow one. 