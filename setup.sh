mkdir img
mkdir models
mkdir models/nst
mkdir models/perceptual
mkdir output
mkdir output/masks
mkdir output/gpu_result
mkdir output/result

conda create --name tfenv --file tf_spec.txt
conda create --name tencu --file tfcuda_spec.txt
conda create --name pytcu10 --file torch_spec.txt

