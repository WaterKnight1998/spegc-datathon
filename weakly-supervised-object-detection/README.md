# Weakly Supervised Object Detection


We made use of GradCam to obtain the location of the underwater elements by just having the class of the image (no bounding boxes).

```
Grad-CAM uses the gradients of any target concept (say logits for “dog” or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
```

Grad-CAM technique allows us to interpretate our CNN models
without the need of retraining our model or updating it's architecture. In addition, it is class-discriminative and as it is applied to last convolutional layer we get high accuracy and detail.

This technique is usually used to explain our CNN models:
* It allows you to understand the cases on which our model was wrong.

* It helps the user gain confidence on the model.

* In cases where AI is better than humans we can learn from it.

As suggested in the paper it can be used for Weakly Supervised Object Detection too by just applying a Treshold to binarize the Heat Map. 

By now this treshold needs to be selected manually for each image.

## Training

Check `Train Elements Detection.ipynb` notebook to see how we trained the classification model with FastAI.

## Inference (Obtain location of the different elements)

Check `Inference Elements Detection.ipynb` notebook to see how to apply GradCam.
## [OPTIONAL] Instructions for testing locally

In the moment of writing this documentation, notebooks can be executed on Google Colab without any special configuration.

If at some point Google Colab is not working you could follow these steps and get the same environment that was used for training. 

**Please note**: These instructions assume that you have an Nvidia RTX 30XX, because the conda environment is installing CUDA 11.1.
### Creating the environment

The first step is to create the environment used to train the models.

```bash
conda env create -f conda.yaml
```

### Activating the environment

To activate the conda environment please execute:

```bash
conda activate datathon
```

### Start Jupyter Lab

For the development of the models we used Jupyter Lab.

To start it, please execute `jupyter lab` in terminal.

## References

[GradCam](https://arxiv.org/abs/1610.02391)

[FastAI GradCam Implementation](https://github.com/fastai/fastbook/blob/master/18_CAM.ipynb)