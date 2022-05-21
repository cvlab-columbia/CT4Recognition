# CVPR Causal Transportability for Visual Recognition


## ColorMNIST

Run `python fd_classifier_cmnist.py`.

We provide a pretrained model here: `cvpr_cmnist_s8_z32_fdc_l_256_model_best_508.pth`


## Waterbird
Download waterbird dataset, we provide a version [here](https://cv.cs.columbia.edu/mcz/CausalTrans/waterbird_dataset.tar.gz) 

If you want to run our pretrained model, here are two models by running two times: [Model 1](https://cv.cs.columbia.edu/mcz/CausalTrans/waterbird_model_checkpoint_78.pth), [Models 2](https://cv.cs.columbia.edu/mcz/CausalTrans/waterbird_model_papermodel.pth)

Run `python fd_waterbird.py` for our experiment, you can choose to retrain your own or evaluating our downloaded checkpoints in Line 253 and L467.


## 
