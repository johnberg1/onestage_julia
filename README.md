# onestage_julia
Julia implementation of the CVPR 2021 paper Training Generative Adversarial Networks in One Stage

### Baseline Model Updates

- `src` includes the layers and networks
- `utils` includes some utility functions
- `main.ipynb` includes the main training pipeline.
- `trained_model` include a generator and a discriminator checkpoint, trained on MNIST for 20 epochs for the two stage setup.

### Current Status

 Currently, I am able to train the two stage model on MNIST, and it produces good outputs. These outputs can be seen on `main.ipynb`. I will continue with the implementation of the one stage version soon.

### TODO Items
- Batch Normalization layer gives a weird CUDA error, fix that. ✔
- Complete the training loop by implementing the optimizers. ✔
- Fix the bug in the training, so that the model is outputting meanningful results. ✔
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
