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
env=tencu

content=(1 18 34)
style=(11 28 44)
script=segmentation_mix.py

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

task_dir=${home_net}/${project}/nst
image_dir=${home_net}/${project}/img/data/new_att_all
save_dir=${home_net}/${project}/output/gpu_result/${time}

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}
mkdir -p ${save_dir}/opt
cp ${task_dir}/* ${sub_dir}/

for i in 0 1 2; do
cp ${image_dir}/${content[${i}]}.png ${sub_dir}
cp ${image_dir}/${style[${i}]}.png ${sub_dir}
python -u ${sub_dir}/${script} --data-dir ${sub_dir} --save-dir ${save_dir} --content ${content[${i}]} --style ${style[${i}]} "$@"
done

rm -r ${sub_dir}