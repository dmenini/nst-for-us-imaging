#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/nst-for-us-imaging/img/gpu_result/
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l h_vmem=50G
#$ -l gpu=1
#$ -q gpu.24h.q
#$ -r y

name=dmenini
machine=hoss
project=nst-for-us-imaging
env=pytcu10

images=1
script=style.py

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

models_dir=${home_net}/${project}/perc_loss
image_dir=${home_net}/${project}/img/style_dataset

data_dir=${home_gpu}/submission/${time}
save_dir=${home_net}/${project}/img/gpu_result/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${data_dir}/img/new_att_all
mkdir -p ${save_dir}/opt
cp ${models_dir}/* ${data_dir}/
cp  ${image_dir}/new_att_all/* ${data_dir}/img/new_att_all/

python -u ${data_dir}/${script} --style-dir ${data_dir}/img --save-dir ${save_dir} --image ${images} --gpu 1 train --visualize 0 "$@"

rm -r ${data_dir}