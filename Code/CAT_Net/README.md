**Here is how to setup CAT_Net in the SureBoTv2 environment**

### Make sure the dependencies as required in the requirements.txt is installed

### Install jpegio
git clone https://github.com/dwgoon/jpegio.git
cd jpegio
python setup.py install

### Download and create weights
Download Weights: https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq
Store the weights as shown in the folders below:

File Structure
CAT-Net
├── pretrained_models  (pretrained weights for each stream)
│   ├── DCT_djpeg.pth.tar
│   └── hrnetv2_w48_imagenet_pretrained.pth
├── output  (trained weights for CAT-Net)
│   └── splicing_dataset
│       ├── CAT_DCT_only
│       │   └── DCT_only_v2.pth.tar
│       └── CAT_full
│           └── CAT_full_v1.pth.tar
│           └── CAT_full_v2.pth.tar


### Usage

**Step 1: Upload images into the images folder**

**Step 2: Enter the file_name in the command prompt "Filename:"**

**Step 3: Wait for the codes to finish processing**

**Step 4: Review the results**

```
Review results in Command Terminal
Review the output heatmap in the "output_pred" folder
```