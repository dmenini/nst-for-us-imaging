#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/cluster_out/
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l h_vmem=50G
#$ -q gpu.24h.q
#$ -l gpu=1
#$ -r y

NAME=dmenini
MACHINE=hoss
PROJECT=nst-for-us-imaging
ENV=tencu

IMAGES=(1 18 34)
SCRIPT=us_nst_segmentation.py

HOME_NET=/scratch_net/${MACHINE}/${NAME}
HOME_GPU=/scratch/${NAME}

DATA_DIR=img/data/new_att_all/
SAVE_DIR=img/result/
DICT_FILE=avg_style.pickle

source ${HOME_NET}/.bashrc

source ${HOME_NET}/anaconda3/etc/profile.d/conda.sh
conda activate ${ENV}

mkdir -p ${HOME_NET}/${PROJECT}/models/
mkdir -p ${HOME_NET}/cluster_sub/models/
cp ${HOME_NET}/${PROJECT}/models/* ${HOME_NET}/cluster_sub/models/
mkdir -p ${HOME_GPU}/submission
for i in ${IMAGES[@]}; do cp ${HOME_NET}/${PROJECT}/${DATA_DIR}/${i}.png ${HOME_GPU}/submission/; done

nvcc --version

python -u ${HOME_NET}/cluster_sub/models/${SCRIPT} --data-dir ${HOME_GPU}/submission/ --save-dir ${HOME_NET}/${PROJECT}/${SAVE_DIR}

rm -r ${HOME_GPU}/submission