wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.4/files/ngccli_linux.zip -O ngccli_linux.zip && unzip ngccli_linux.zip
find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
sha256sum ngccli_linux.zip
chmod u+x ngc-cli/ngc
echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
ngc config set  # Enter: no-apikey

ngc registry resource download-version "nvidia/modulus/modulus_datasets-hrrr_mini:1"
ngc registry resource download-version "nvidia/modulus/modulus_datasets_cwa:v1"

mkdir -p data/corrdiff-mini/
cp modulus_datasets-hrrr_mini_v1/modulus_datasets-hrrr_mini_v1/hrrr_mini/* data/corrdiff-mini/

# pip install nvidia-physicsnemo
cd ~/physicsnemo & pip install -e .
pip install hydra-core wandb nvtx opencv-python dask

# HRRR-Mini example
python train.py --config-name=config_training_hrrr_mini_regression.yaml  # Duration: A few hours on a single A100 GPU (2m images)

python train.py --config-name=config_training_hrrr_mini_diffusion.yaml \
  ++training.io.regression_checkpoint_path=./checkpoints_regression/UNet.0.20224.mdlus

HYDRA_FULL_ERROR=1 python generate.py --config-name="config_generate_hrrr_mini.yaml" \
  ++generation.io.res_ckpt_filename=./checkpoints_diffusion/EDMPrecondSuperResolution.0.80128.mdlus \
  ++generation.io.reg_ckpt_filename=./checkpoints_regression/UNet.0.20224.mdlus

