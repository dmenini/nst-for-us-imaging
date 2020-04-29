#!/bin/bash
#$ -o /scratch_net/hoss/dmenini/cluster_out/
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=50G
#$ -q gpu.24h.q

NAME=dmenini
MACHINE=hoss
PROJECT=nst-for-us-imaging
ENV=tfenv

IMAGES=(1 18 34)
SCRIPT=us_nst_segmentation.py

HOME_NET=/scratch_net/${MACHINE}/${NAME}
HOME=/scratch/${NAME}

DATA_DIR=img/data/new_att_all/
SAVE_DIR=img/result/
DICT_FILE=avg_style.pickle

echo $HOME_NET

source ~/.bashrc
source ${HOME_NET}/anaconda3/etc/profile.d/conda.sh
conda activate ${ENV}

mkdir -p ${HOME_NET}/${PROJECT}/models/
mkdir -p ${HOME_NET}/cluster_sub/models/
cp ${HOME_NET}/${PROJECT}/models/* ${HOME_NET}/cluster_sub/models/
mkdir -p ${HOME}/submission
for i in ${IMAGES[@]}; do cp ${HOME_NET}/${PROJECT}/${DATA_DIR}/${i}.png ${HOME}/submission/; done

python -u ${HOME_NET}/cluster_sub/models/us_nst_segmentation.py --data-dir ${HOME}/submission/ --save-dir ${HOME_NET}/${PROJECT}/${SAVE_DIR}

rm -r ${HOME}/submission