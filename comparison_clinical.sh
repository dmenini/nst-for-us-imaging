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
task=hq2clinical

content=(34)
style=(000)
script=nst.py
dict=us_clinical_dict.pickle

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

task_dir=${home_net}/${project}/nst
image_dir=${home_net}/${project}/img
save_dir=${home_net}/${project}/output/gpu_result/${time}
models_dir=${home_net}/${project}/models/nst

content_dir=${image_dir}/data/new_att_all
style_dir=${image_dir}/clinical_us/training_set

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}
mkdir -p ${save_dir}/opt
cp ${task_dir}/* ${sub_dir}/
cp ${models_dir}/${dict} ${sub_dir}/

for i in 0 1 2; do
	content_file=${content[${i}]}.png
	style_file=${style[${i}]}_HC.png
	cp ${content_dir}/${content_file} ${sub_dir}
	cp ${style_dir}/${style_file} ${sub_dir}
	python -u ${sub_dir}/${script} --task ${task} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${style_file} --dict-path ${sub_dir}/${dict} --lr $lr --loss 0 --name mix0_ "$@"
	python -u ${sub_dir}/${script} --task ${task} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${style_file} --dict-path ${sub_dir}/${dict} --lr $lr --loss 1 --name mix1_ "$@"
	python -u ${sub_dir}/${script} --task ${task} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${style_file} --dict-path ${sub_dir}/${dict} --lr $lr --loss 2 --name mix2_ "$@"
	python -u ${sub_dir}/${script} --task ${task} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${content_file} --dict-path ${sub_dir}/${dict} --lr $lr --loss 0 --name old0_ "$@"
	python -u ${sub_dir}/${script} --task ${task} --save-dir ${save_dir} --content ${sub_dir}/${content_file} --style ${sub_dir}/${content_file} --dict-path ${sub_dir}/${dict} --lr $lr --loss 1 --name old1_ "$@"
done
done

rm -r ${sub_dir}