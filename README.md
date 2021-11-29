# onestage_julia
Julia implementation of the CVPR 2021 paper Training Generative Adversarial Networks in One Stage

### Baseline Model Updates

- `src` includes the layers and networks
- `utils` includes some utility functions
- `main.ipynb` includes the main training pipeline. Currently, I am able to initialize the networks, read the data, pass it through the network and output losses, and update the paremeters. However, the loss functions get stuck in some value, and the output of the Generator is not meaningful. 

### TODO Items
- Batch Normalization layer gives a weird CUDA error, fix that. ✔
- Complete the training loop by implementing the optimizers ✔
- Fix the bug in the training, so that the model is outputting meanningful results.
- Complete the dataloader for CelebA too.
- Implement the onestage version.

```
@InProceedings{shen2021training,
    author    = {Shen, Chengchao and Yin, Youtan and Wang, Xinchao and Li, Xubin and Song, Jie and Song, Mingli},
    title     = {Training Generative Adversarial Networks in One Stage},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3350-3360}
}
```
