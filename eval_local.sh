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

content=(34)
style=(645)
script=local_transfer.py

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

task_dir=${home_net}/${project}/nst
image_dir=${home_net}/${project}/img
save_dir=${home_net}/${project}/output/gpu_result/${time}

content_dir=${image_dir}/data/new_att_all
style_dir=${image_dir}/data/new_att_all

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}
mkdir -p ${save_dir}/opt
cp ${task_dir}/* ${sub_dir}/

END=$((${#content[@]} - 1))
for i in $(seq 0 $END); do
	content_file=${content[${i}]}.png
	style_file=${style[${i}]}.png
	cp ${content_dir}/${content_file} ${sub_dir}
	cp ${style_dir}/${style_file} ${sub_dir}
	python -u ${sub_dir}/${script} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${style_file} "$@"
done

rm -r ${sub_dir}