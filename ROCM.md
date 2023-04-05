building rocm from source:

# install rocm
# https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/How_to_Install_ROCm.html#d23e6376
sudo amdgpu-install --usecase=rocmdev,graphics,opencl,hiplibsdk --no-32

# compile cupy against rocm (if there's no compatible prebuilt binary (e.g cupy-rocm-5.0)

git clone https://github.com/cupy/cupy
git submodule update --init
pip install -U setuptools pip
# supplemetary notes: https://docs.cupy.dev/en/v7.8.0/install_rocm.html 
export ROCM_HOME=/opt/rocm-5.4.3/
# get target from the rocminfo 'name' output
/opt/rocm/bin/rocminfo
export HCC_AMDGPU_TARGET=gfx1030
export __HIP_PLATFORM_HCC__
export CUPY_INSTALL_USE_HIP=1
sudo apt-get install libstdc++-12-dev
pip3 install -e .

