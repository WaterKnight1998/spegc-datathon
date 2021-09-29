# Seafloor classification

To solve this challenge, we took advantage of transfer learning and fine tuning a state of the art model. The chosen model was a ResNet152v2 architecture with a custom central crop layer and some heavy data augmentation to avoid overfitting and solve some of the secondary tasks given by the organization.

## Training

The training process is available in the `seafloor_train.ipynb` jupyter notebook. It mainly uses TF/Keras and Albumentations.

## Inference

Similar to training, the evaluation notebook is available as `seafloor_test.ipynb`.

## [OPTIONAL] Instructions for testing

In the moment of writing this documentation, notebooks can be executed on Google Colab without any special configuration.

## References

[ResNet152v2](https://arxiv.org/abs/1603.05027)
