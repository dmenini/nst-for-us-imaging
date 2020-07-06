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
save_dir=${home_net}/${project}/output/gpu_result/${time}

model_dir=${home_net}/${project}/models/perceptual

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}
mkdir -p ${save_dir}
cp -r ${task_dir}/* ${sub_dir}/
cp -r ${model_dir}/* ${sub_dir}/

content_dir=${home_net}/${project}/img/lq_test
model=us_lq2hq_noisy_v2
mkdir -p ${save_dir}/$model

for i in {1..600}; do
python -u ${sub_dir}/${script} --mode transfer --save-dir ${save_dir} --model-name ${model} --content ${content_dir}/$i.png --gpu 1 "$@"
done
rm -r ${sub_dir}