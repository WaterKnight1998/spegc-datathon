# Team 36: SPEGC Datathon 2021

## 📩 Contact

💼 [David Lacalle Castillo](https://es.linkedin.com/in/david-lacalle-castillo-5b6280173)

💼 [Adrián Pascual Bernal](https://es.linkedin.com/in/adri%C3%A1n-pascual-bernal-536a12176)

💼 [Guillermo Sánchez Brizuela](https://es.linkedin.com/in/guillermo-sanchez-brizuela)

This README covers our team's solution for the SPEGC Datathon that took place in september 2021. This solution was chosen as the winning approach and therefore we won the first prize!

The team was randomly formed and we didn't know each other before the competition, final team members were: [Adrián Pascual Bernal](https://www.linkedin.com/in/adri%C3%A1n-pascual-bernal-536a12176/), [David Lacalle Castillo](https://www.linkedin.com/in/david-lacalle-castillo-5b6280173/), Cesar Muñoz Araya and me, [Guillermo Sánchez Brizuela](https://www.linkedin.com/in/guillermo-sanchez-brizuela/). Our tech stack was primarily composed of Tensorflow/Keras, Pytorch/FastAI, Albumentations, Jupyter Notebooks and Anaconda.

***

# The Datathon

This Datathon was organized by the [SPEGC (Sociedad de Promoción Económica de Gran Canaria - Gran Canaria's Economic Promotion Society)](https://www.spegc.org/). Participation was open to the whole world but due to the linguistic scope of the competition, participants were mainly from Spain and Latin American countries. The format of the competition was a data science oriented Hackaton, this is, a set of problems were given along a dataset, and every team had to submit a solution developed in two and a half days.

***

# The dataset

As previously mentioned, at the start of the competition, all teams accessed a curated dataset of images. The dataset was provided by [PLOCAN (Plataforma Oceánica de Canarias - The Oceanic Platform of the Canary Islands)](https://www.plocan.eu/) and contained around 5.3k underwater images of size 3840 x 2160 pixels (w x h), taken by underwater ROVs (Remotely Operated Vehicles). These images, were divided in two subsets, one part (~2.8k images) was tagged based on the type of submarine floor, and the other one (~2.5k images) was tagged based on the presence of a series of elements. Both datasets are closely related to the main challenges, explained below.

Some samples of these images are shown in Figure 1.

***

# Challenges

The competition posed two main challenges and a series of "minor" challenges.

### Main challenge #1

Using the first subset of the dataset, a classification model had to be trained to classify the floor in a given image. The floor type options were sandy, muddy, muddy-sand and reef.

### Main challenge #2

Now using the second part of the dataset, a second classification task was proposed, this time classifying the presence of different elements in the image. Annotated classes were ripples, fauna, algae, waste and rocks. It is important to note that this was a **classification** task, as the dataset was annotated only with the presence of the main object. **No location annotations** were provided to train an object detection model. 

### Secondary challenges

1. Solving both problems at the same time: This is, making a classifier capable of classifying both the floor type and the element presence.
2. Detecting the elements in the image: As no location tags were given, predicting the position of the detected element in the image was a challenging task.
3. Making the model robust to resolution changes: All the given images were the same size, but videos with different resolution and compressions are also recorded by PLOCAN and the model would ideally be resilient to input size changes.
4. Finding subcategories of the proposed element's classes: For example, detect different types of thras using unsupervised learning.
5. Making the model resistant to the absence of the ROV structure: In a great percentaje of the images, the metallic structure of the ROV appears in the sides of the photo, the idea behind this task is to make the model able to perform equally well if no structure is present.
6. Making the model robust to different optical conditions: Underwater lighting can vary a lot due to the cloudyness of the water and external light sources. Furthermore, the ROV camera orientation can change, changing the perspective of the photo. Taking into account these factors would make a better model.

Apart from this, results reproducibility and model inference scripts were positively valuated.

***

# Core ideas

There were a few ideas that we all agree with, due to the reduced time frame we had, transfer learning and fine tuning models were a must, as there was no time to test randomly initializated architectures. Furthermore, as the dataset was limited and the secondary challenges were demanding, data augmentation was also something everybody considered neccesary. We needed a way to compare results between architectures and tests, so we divided the datasets in train/validation splits (80/20), we made this split stratified to ensure no class was underrepresented. Once the dataset was splitted, the validation subset was used to evaluate the model using weighted accuracy as the metric.  

***

# Data augmentation

Although secondary challenges were not a priority, we considered them from the beggining of the competition. One outcome of this, is the inclusion of a quite aggresive data augmentation process during training, randomly parametrizing its application on each epoch. We used data augmentation to deal with the lack of very different scenarios and avoid overfitting. Guided by a subset of the secondary challenges, augmentations used were:

- Blur (Secondary challenge #6)
- Brightness and contrast (Secondary challenge #6)
- JPG compression (Secondary challenge #3)
- Saturation (Secondary challenge #6)
- Affine transformations: Translation, Rotation, Mirroring, Scale (Secondary challenge #3 and #6)
- Proyective transformations (Secondary challenge #6)

These augmentations were implemented using [Albumentations](https://albumentations.ai/docs/).

***

# Preprocessing

Another one of the secondary challenges (#5) was dealing with the metallic structure of the ROV. To solve this, we integrated two layers inmediatly after the Input layer, a central crop of 2/3 of the image, and a resizing of this crop to 512x512. These two operations help with multiple issues:

- No metallic structure makes it to the actual network, so no overfitting to the structure.
- The input image size is now trivial, as the crop is relative to the original size and the cropped region of interest is scaled to a fixed size of 512x512. Thanks to this, altough all images were the same size, we could leave the input shape unspecified in the network. Making the input size trivial helps significantly with the secondary challenge #3.
- As the cropping operation reduces the size of the image, less information is lost when scaling down, due to the original image being smaller.

***

# First challenge

As mentioned before, due to the number of images and the temporal constraints, using transfer learning was the better option. The first day we tried multiple architectures, including ResNet34, ResNet50, ResNet152, a Darknet backbone, Xception, and even a Vision Transformer (ViT). We removed the head and added a Gloabl Average Pooling layer and a fully connected layer of 4 neurons. During model compilation, we also specified class weights to balance the effect of each class in the loss function based on the distribution of tags. First we freezed all the weights except the output layer and trained until validation loss stopped decreasing (Early Stopping). Then, we unfreezed the whole model and fine tuned the weights with a smaller learning rate. 

After comparig the validation results of the different architectures, we choose a ResNet152v2 model which got a weighted accuracy of 0.961.

After choosing the model and the hyperparameters, we trained the model again using the entire dataset of floor images to ensure that the model had been trained on as much data as possible before sending it to final evaluation.

***

# Second challenge

Due to the time limitations of the competition, once we had the model for the first challenge working we decided to avoid looking for new architectures and reused all the code for training and inference we had for the floor type classification. The only change we made was in the output layer of the network to change the four output neurons to five (now there were five classes). After changing this last layer, we proceed to repeat the process of doing transfer learning with the ResNet152v2 trained on Imagenet, ending up fine tuning the whole network.

One thing that went over our heads during the competition was using the model trained with the first dataset as the base for transfer learning, it would had been a good experiment to try.

The final model was trained on the entire elements dataset (train + val) once the architecture and hyperparameters were chosen, similarly to the previous model.

***

# Object detection

An additional challenge was to detect the elements of the second challenge without any localization labeling. This is called Weakly Supervised Object Detection in the academic literature, however, none of the team members knew anything about this field.

Despite all the available research in this area, due the lack of time to search for a good suited method we ended up improvising our own solution in a creative way. We used a method of explainability called Grad-CAM to perform element localization. 

In Grad-CAM the gradient of the output of a deep network is used to generate a heat map of the influence of 
a feature map of the last convolutional layer in the output. 

As we had a good classifier already trained, we took advantage of it by applying Grad-CAM in order to do element detection. 
First, we used Grad-CAM to generate a heatmap of the last convolutional layer influence over the output. Then, we generated a binary mask by thresholding the heatmap, which can be used for the element localization.

***

Related links
+ [Results announcement](https://www.spegc.org/2021/09/23/el-cabildo-de-gran-canaria-valora-los-resultados-del-primer-datathon-de-datos-turisticos/)
