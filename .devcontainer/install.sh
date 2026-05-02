apt-get update

apt-get install -y wget unzip git
DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 libglib2.0-0
pip install autopep8 pylint gdown ipykernel
pip install timm
pip install matplotlib
pip install albumentations