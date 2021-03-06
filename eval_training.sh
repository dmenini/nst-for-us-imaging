#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/nst-for-us-imaging/output/gpu_result/
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l h_vmem=50G
#$ -q gpu.24h.q
#$ -l gpu=1
#$ -r y

task=hq2clinical

name=dmenini
machine=hoss
project=nst-for-us-imaging
env=pytcu10

script=training.py

time=$(date +"%d-%m-%y_%T")

home_net=/scratch_net/${machine}/${name}
home_gpu=/scratch/${name}

task_dir=${home_net}/${project}/perc_loss
save_dir=${home_net}/${project}/output/gpu_result/${time}

if [ ${task} = 'lq2hq' ]; then
	echo lq2hq
	dataset_dir=${home_net}/${project}/img/lq_dataset
	test_dir=${home_net}/${project}/img/lq_test/1.png
	style_dir=${home_net}/${project}/img/hq_dataset/new_att_all
	style=645.png

elif [ ${task} = 'seg2hq' ]; then
	echo seg2hq
	dataset_dir=${home_net}/${project}/img/seg_dataset
	test_dir=${home_net}/${project}/img/seg_test/1.png
	style_dir=${home_net}/${project}/img/hq_dataset/new_att_all
	style=645.png
	
elif [ ${task} = 'hq2clinical' ]; then
	echo hq2clinical
	dataset_dir=${home_net}/${project}/img/hq_dataset
	test_dir=${home_net}/${project}/img/hq_test/1.png
	style_dir=${home_net}/${project}/img/clinical_us/training_set
	style=000_HC.png
fi	

sub_dir=${home_gpu}/submission/${time}

source ${home_net}/.bashrc

source ${home_net}/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

mkdir -p ${sub_dir}/img/new_att_all
mkdir -p ${save_dir}/opt
cp -r ${task_dir}/* ${sub_dir}/
cp ${dataset_dir}/new_att_all/* ${sub_dir}/img/new_att_all/
cp ${style_dir}/${style} ${sub_dir}/
echo cp ${style_dir}/${style} ${sub_dir}/
python -u ${sub_dir}/${script} --dataset ${sub_dir}/img --save-dir ${save_dir} --style ${sub_dir}/${style} --content ${test_dir} --gpu 1 "$@"

rm -r ${sub_dir}