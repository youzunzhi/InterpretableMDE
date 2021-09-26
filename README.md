# InterpretableMDE
A PyTorch implementation for "Towards Interpretable Deep Networks for Monocular Depth Estimation" paper.

arXiv link: https://arxiv.org/abs/2108.05312



## Data and Model

For [MFF](https://github.com/JunjH/Revisiting_Single_Depth_Estimation) models, we use the dataset they released [here](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing), and you can download their models as the baselines [here](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing). For [BTS](https://github.com/cogaplex-bts/bts) models, they use a different set of NYUv2 training images (24,231 instead of 50,688), and you download it [here](https://drive.google.com/file/d/1vh5KsqpgiFWEBzWCImoIEsQANb6djgFC/view?usp=sharing). We put all of our models [here](https://drive.google.com/drive/folders/1zvTpuE00-thzyjUaYR5vK1LJnQgigF6E?usp=sharing).



## Evaluation

In this project we use [yacs](https://github.com/rbgirshick/yacs) to manage the configurations. To evaluate the performance of a model, for example, the MFF model with SENet backbone using our assigning method, simply run

```sh
python eval.py MODEL_WEIGHTS_FILE [PATH_TO_MODEL/mff_senet_asn]
```

from the root directory.

To evaluate the depth selectivity, run

```sh
python dissect.py MODEL_WEIGHTS_FILE [PATH_TO_MODEL/mff_senet_asn] LAYERS D_MFF ON_TRAINING_DATA True
```

then get the depth selectivity and the dissection result of each unit. Layers' names are seperated by `_`.



## Training

To train a model from scratch, run

```sh
python train.py MODEL_NAME MFF_resnet
```

We currently provide four options for `MODEL_NAME`, and the training scheme will automatically be switched to align with the original ones when using BTS models.



## Acknowledgement

The model part of our code is adapted from [Revisiting_Single_Depth_Estimation](https://github.com/JunjH/Revisiting_Single_Depth_Estimation) and [bts](https://github.com/cogaplex-bts/bts). Some snippets are adapted from [monodepth2](https://github.com/nianticlabs/monodepth2).



## Bibtex

```
@article{you2021interpretable,
	title		= {Towards Interpretable Deep Networks for Monocular Depth Estimation}, 
  	author		= {Zunzhi You and Yi-Hsuan Tsai and Wei-Chen Chiu and Guanbin Li},
  	journal 	= {arXiv preprint arXiv:2108.05312},
  	year		= {2021},
}
```

