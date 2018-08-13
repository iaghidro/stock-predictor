# This script assumes that the file cudnn-8.0-linux-x64-v7.tgz is already present in the same directory,
# the file can be found at https://developer.nvidia.com/rdp/cudnn-download after registering with Nvidia.
# Also it is assumed that the requirements for CUDA 8 installations are already met.

# ensure system is updated and has basic build tools
sudo apt update
sudo apt --assume-yes upgrade
sudo apt --assume-yes install build-essential gcc g++ make binutils
sudo apt --assume-yes install software-properties-common git


# last release of cuda 8 with update
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb" -O "cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt update
sudo apt -y install cuda
wget "https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb" -O "cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb"
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb
sudo apt update
#sudo modprobe nvidia
nvidia-smi
#rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
#rm cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb

# cudnn
echo "installing cudnn ..."
tar -zxf cudnn-8.0-linux-x64-v7.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/* /usr/local/cuda/include/
#rm cudnn-8.0-linux-x64-v7.tgz
