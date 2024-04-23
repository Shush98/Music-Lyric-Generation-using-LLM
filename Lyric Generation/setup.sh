wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get -y install cuda
sudo apt-get -y install nvidia-gds

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install torch==2.0.0

pip install transformers==4.37.2

pip install datasets

pip install huggingface_hub

pip install peft==0.8.2

pip install trl

pip install accelerate==0.26.1

pip install bitsandbytes==0.42.0