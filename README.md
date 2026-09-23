
# This repository contains the source code and trained models for TTCT.  

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
This project is licensed under the Apache-2.0 license - see **LICENSE** for details.

## Contact us  
b22070022@s.upc.edu.cn 
