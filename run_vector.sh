export PATH="$PATH:/h/$(whoami)/miniconda3/bin"
source activate fastmri
cd /h/$(whoami)/fastMRIPrivate
pip install -r requirements.txt
PYTHONPATH=$PWD python models/main.py --data-path /localssd/fastMRI/ --challenge singlecoil --batch-size 4 --model unet_image_fcn_consistent --exp-name unet_image_fcn_overfit --batches-per-volume 2
