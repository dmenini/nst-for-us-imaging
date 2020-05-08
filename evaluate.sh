#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/nst-for-us-imaging/img/gpu_result/
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

images=(1 18 34)
script=us_nst_seg_old.py
dict_file=avg_style.pickle

time=$(date +"%T_%m-%d-%y")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

models_dir=${home_net}/${project}/models
image_dir=${home_net}/${project}/img/data/new_att_all

data_dir=${home_gpu}/submission/${time}
save_dir=${home_net}/${project}/img/gpu_result/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${data_dir}
mkdir -p ${save_dir}/opt
cp ${models_dir}/* ${data_dir}/

for i in ${images[@]}; do
cp ${image_dir}/${i}.png ${data_dir}
python -u ${data_dir}/${script} --data-dir ${data_dir} --save-dir ${save_dir} --image ${i} "$@"
done

rm -r ${data_dir}