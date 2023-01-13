##

Download [PTH File](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth) : 

    mkdir checkpoints
    cd checkpoints
    wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth

### Conda

    conda env create -f environment.yml
    conda activate blip
    python annotate.py --config configs/med_config.json --model checkpoint/model_base_caption_capfilt_large.pth --image /image-path/image.jpg

### Pants

You need to have 2 libs in your system :
- [conda](https://docs.vmware.com/en/VMware-vSphere-Bitfusion/3.0/Example-Guide/GUID-ABB4A0B1-F26E-422E-85C5-BA9F2454363A.html)
- [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)


    ./pants run annotate.py -- --config configs/med_config.json --model checkpoint/model_base_caption_capfilt_large.pth --image /image-path/image.jpg

Package a pex file :

    ./pants package ::

Create tar.gz

    cd scripts
    ./package.sh