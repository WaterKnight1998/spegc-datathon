# Underwater elements detection challenge

In this challenge it was proposed to carry out the classification of different elements present in the underwater images,
specifically these elements are "algas", "basura", "fauna", "ripples" and "roca".


Our solution consists of a ResNet 152 based classifier, trained by transfer learning and fine tuning, some preprocessing
tricks and an agressive data augmentation strategy. You can find the training code in
`elementos_train.ipynb` notebook.


For inference and evalutation purposes another notebook (`elementos_test.ipynb`) is provided.


Both notebooks are ready to be used in Google Colab 

