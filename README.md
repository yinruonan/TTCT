# Cross Teaching between 3-D and 2-D Networks for 3-D Seismic Fault Detection  
This repository contains the code and trained models for the manuscript "**3-D Seismic Fault Detection From Biased Sparse Annotations via Dynamic Cross-Teaching**".  

**ABSTRACT:**
Data-driven approaches to 3-D seismic fault detection typically rely on large-scale, accurately annotated datasets. However, voxel-wise fault annotation is labor-intensive and highly subjective, which makes high-quality 3-D labels difficult to obtain in practice. Training survey-specific models from sparse slice annotations provides a practical alternative, yet learning a 3-D network directly from such supervision remains challenging because the supervision is limited and the annotations are often biased. To address this issue, we propose TTCT, a cross-teaching framework between 2-D and 3-D networks for 3-D seismic fault detection from sparse slice annotations with annotation bias. TTCT exploits the complementary strengths of 2-D and 3-D networks through dynamic cross-teaching, thereby enabling robust learning under sparse supervision while reducing dependence on annotated slices. In addition, we introduce an annotation bias-aware loss to model sparse annotation confidence by approximating expert attention, thereby improving robustness to annotation bias. Experiments on three public datasets show that TTCT outperforms strong semi-supervised and weakly supervised baselines with fewer annotated slices and can recover some true but unlabeled faults. Notably, TTCT achieves a Dice similarity coefficient (DSC) of 82.37\% with only 6.25\% (1/16) annotated slices per sample, close to the 83.72\% achieved by full supervision. Moreover, the DSC decreases by only 0.73\% when annotation bias affects 64.4\% of the annotated slices.  

##  Dataset  
The training and test datasets used in this work can be downloaded [here](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8).

After downloading, please organize the dataset in the following structure:

```text
root_dir
├── test
│   ├── fault
│   │   ├── 0.dat
│   │   └── ...
│   └── seis
│       ├── 0.dat
│       └── ...
└── train
    ├── fault
    │   ├── 1.dat
    │   └── ...
    └── seis
        ├── 1.dat
        └── ...
```  
## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```
## Training
To train TTCT with different numbers of labeled slices, run the following command:

```bash
python train.py \
    --root-dir your_dataset_root \
    --alg ttct \
    --n-slice 8 \
    --save-freq 20 \
    --eval-freq 20
```
To train TTCT with simulated biased annotations, run:
```bash
python train.py \
    --root-dir your_dataset_root \
    --alg ttct \
    --n-slice 8 \
    --save-freq 20 \
    --eval-freq 20 \
    --sim-expt-ann
```
Here, `--n-slice` specifies the number of labeled slices used for training.

## Quick test

We provide pretrained checkpoints [here](https://drive.google.com/drive/folders/1luf_PZJ73oRP6eJUZTHXvV7FDDC-bZR6?usp=sharing). To evaluate the model and visualize the results, download the pretrained checkpoints and place them in `./ckpts`, then run:

```bash
python test.py
```

## License  
This project is licensed under the Apache-2.0 license - see LICENSE.md file for details.

## Contact us  
b22070022@s.upc.edu.cn 