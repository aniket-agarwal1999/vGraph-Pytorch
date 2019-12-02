# vGraph: A Generative Model For Joint Community Detection and Node Representational Learning

This is a Pytorch implementation of the paper [vGraph: A Generative Model For Joint Community Detection and Node Representational Learning](https://arxiv.org/abs/1906.07159) and is done under the **NeurIPS Reproducibility Challenge 2019**. The original implementation by author can be found [here](https://github.com/fanyun-sun/vGraph).

## Summary of the paper

<img src='https://github.com/aniket-agarwal1999/vGraph-Pytorch/blob/master/images/model.png'>

This paper proposes a novel technique for learning node representations and at the same time perform community detection task for the graphical data by creating a generative model using the variational inference concepts. **The full paper summary along with its main contributions can be found [here](https://github.com/vlgiitr/papers_we_read/blob/master/summaries/vgraph.md)**

## Setup Instructions and Dependancies

The code has been written in *Python 3.6* and *Pytorch v1.1*. Also Pytorch Geometric has been used for training procedures, along with the usage of TensorboardX for logging loss curves.

For training/testing the model, you must first download `Facebook social circles` dataset. It can be found [here](https://snap.stanford.edu/data/ego-Facebook.html). After downloading the dataset, all the files must be placed inside `./dataset/Facebook/`.

## Repository Overview

The following is the information regarding the various important files in the directory and their functions:

- `model.py`: File containing the network architecture
- `utils.py`: File containing helper functions and losses
- `data.py`: File containing functions to call dataset in an operable format
- `train_nonoverlapping.py`: File containing the training procedure for non-overlapping dataset
- `train_overlapping.py`: File containing the training procedure for overlapping dataset

## Running the model

For training the model, use the following commands:

```
python train_nonoverlapping.py            ### For training non-overlapping dataset
python train_overlapping.py               ### For training overlapping dataset
```

## Current Status of the Project

Currently the directory contains dataloader and training procedure for 2 non-overlapping datasets(`Cora` and `Citeseer`) and 10 overlapping datasets(`facebook0`, `facebook107`, `facebook1684`, `facebook1912`, `facebook3437`, `facebook348`, `facebook3980`, `facebook414`, `facebook686`, `facebook698`). I plan to add more dataloaders in the directory. Also the various accuracy measures as specified in the paper will also soon be added in the repository.

```
If you found the codebase useful in your research work, consider citing the original paper
```

## License

This repository is licensed under MIT License