#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/nst-for-us-imaging/output/gpu_result/
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l h_vmem=50G
#$ -q gpu.24h.q
#$ -l gpu=1
#$ -r y

name=dmenini
machine=hoss
project=nst-for-us-imaging
env=pytcu10

script=style.py

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

task_dir=${home_net}/${project}/perc_loss
dataset_dir=${home_net}/${project}/img/content_dataset
save_dir=${home_net}/${project}/output/gpu_result/${time}

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}/img/new_att_all
mkdir -p ${save_dir}/opt
cp -r ${task_dir}/* ${sub_dir}/
cp  ${dataset_dir}/new_att_all/* ${sub_dir}/img/new_att_all/

python -u ${sub_dir}/${script} --dataset ${sub_dir}/img --save-dir ${save_dir} --gpu 1 "$@"

rm -r ${sub_dir}