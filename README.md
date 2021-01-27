# FactorVAE
Pytorch implementation of FactorVAE proposed in Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies
```
python 3.8.5
pytorch 1.7.0
torchvision 0.8.1
opencv 4.5.0
tqdm
```
<br>
You can easily install al dependencies with anaconda using
```
conda env create -f environment.yml
```

<br>

### 2D Shapes(dsprites) Dataset
```
sh scripts/prepare_data.sh dsprites
```
the data directory structure should be like below<br>
```
.
└── data
    └── dsprites-dataset
        └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

### Usage
you can reproduce results below as follows
```
e.g.
sh scripts/run_celeba.sh $RUN_NAME
sh scripts/run_dsprites_gamma6p4.sh $RUN_NAME
sh scripts/run_dsprites_gamma10.sh $RUN_NAME
sh scripts/run_3dchairs.sh $RUN_NAME
```
or you can run your own experiments by setting parameters manually
```
e.g.
python main.py --name run_celeba --dataset celeba --gamma 6.4 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 ...
```

### Reference
1. Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])


[http://arxiv.org/abs/1802.05983]: http://arxiv.org/abs/1802.05983
[download]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
