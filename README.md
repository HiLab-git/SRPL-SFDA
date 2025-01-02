## SRPL-SFDA: SAM-Guided Reliable Pseudo-Labels for Source-Free Domain Adaptation in Medical Image Segmentation
This repository provides the code for "SRPL-SFDA: SAM-Guided Reliable Pseudo-Labels for Source-Free Domain Adaptation in Medical Image Segmentation".

## Requirements
Before you can use this package for image segmentation. You should:

- PyTorch version >=1.0.1
- Some common python packages such as Numpy, Pandas, SimpleITK,OpenCV, pyqt5, scipy......
- Install the [SAM](https://github.com/facebookresearch/segment-anything) for Segment Anything.

## Usage
1. Clone this repository
```bash
git clone
```
2. Download the dataset and put them in the corresponding folder
3. Run the following command to train the model
```bash
sh train.sh
```
4. Run the following command to test the model
```bash
sh test.sh
```
<!-- 5. Run the following command to evaluate the model
```bash
python evaluate.py
``` -->


