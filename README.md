# Identification-system-for-white-blood-cells-Using-Transfer-Learning
Identification system for white blood cells (WBCs) in blood. Using transfer learning to create a CNN model to identify WBCs in smear images and discussing Generalization Power

---
# Introduction
The generalization power is one of the main challenges in the world of artificial intelligence.
In this project, we intend to investigate the generalizability of convolutional networks in the problem of white blood cell classification.
As we all know, white blood cells play an essential role in the immune system. The first signs of any disease appear quickly in white blood cells. In other words, when a person becomes ill, changes in the number of white blood cells in the patient's blood samples become abnormal. Therefore, differential white blood cell counts can be helpful in early diagnosis of the disease. White blood cells in normal peripheral blood consist of 5 general categories (5 classes)
They are: neutrophils, lymphocytes, monocytes, eosinophils and basophils. In this project, we want to separate these 5 categories from each other with the help of convolutional network. 

![A](https://s20.picofile.com/file/8442487984/Types_of_White_Blood_Cells.png)

---
# CNN Model

---
### CNN Structure Summary
<img width="500" alt="2" src="https://user-images.githubusercontent.com/73002780/138605523-79eb4b7b-a485-432d-9df7-663d3ae10900.jpg">
<img width="1000" alt="2" src="https://user-images.githubusercontent.com/73002780/138605541-4ff7affd-c02b-43a1-997c-b4dbaf853570.png">

---
## CNN Model Results
---
### loss and accuracy for Validation and Train data:
| Validation  | Validation | Train  | Train |
| ------------- | ------------- | ------------- | ------------- |
| Loss  | 0.0364  | Loss  | 0.0760  |
| Accuracy  | 0.9900  | Accuracy  | 0.9755  |

### Accuracy for Train, Test1 and Test2 datas:
| Train Accuracy | Test1 Accuracy  | Test2 Accuracy |
| ------------- | ------------- | ------------- |
| 0.9935  | 0.9610  | 0.0911  |

### Training History Plot
<img width="500" alt="2" src="https://user-images.githubusercontent.com/73002780/138611354-223421ca-69f9-498a-a8ae-716d43e7fbba.jpg">

### Classification Report For Test1 Datas
<img width="500" alt="2" src="https://user-images.githubusercontent.com/73002780/138611478-3aafb0cd-023f-46f0-9733-b6b088f0d6ce.jpg">

### Classification Report For Test2 Datas
<img width="500" alt="2" src="https://user-images.githubusercontent.com/73002780/138611484-752fbb6b-54a9-4807-aabe-7a9f43104870.jpg">

### Confusion Matrix For Train, Test1 and Test2 Datas
![6](https://user-images.githubusercontent.com/73002780/138611544-62a28c05-754c-4a38-963c-ea884ccf338b.jpg)

---
# VGG16 Model
### vgg16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

![vgg16-1-e1542731207177](https://user-images.githubusercontent.com/73002780/138611801-d7db489e-babb-4a2b-9913-0c039f5cd7ed.png)

### dataset
ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Amazon’s Mechanical Turk crowd-sourcing tool. Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held. ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. At all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images. ImageNet consists of variable-resolution images. Therefore, the images have been down-sampled to a fixed resolution of 256×256. Given a rectangular image, the image is rescaled and cropped out the central 256×256 patch from the resulting image.

---
### VGG16 Structure Summary
<img width="1300" alt="2" src="https://user-images.githubusercontent.com/73002780/138612261-623264b0-e083-42da-b5fb-8da43c560860.jpg">


---
## VGG16 Model Results
---
### loss and accuracy for Validation and Train data:
| Validation  | Validation | Train  | Train |
| ------------- | ------------- | ------------- | ------------- |
| Loss  | 0.0330  | Loss  | 0.0006  |
| Accuracy  | 0.9934  | Accuracy  | 1.0  |

### Accuracy for Train, Test1 and Test2 datas:
| Train Accuracy | Test1 Accuracy  | Test2 Accuracy |
| ------------- | ------------- | ------------- |
| 0.9975  | 0.9221  | 0.0679  |

### Training History Plot
<img width="500" alt="2" src="https://user-images.githubusercontent.com/73002780/138612317-f88eaeb7-9a43-44f3-abe6-656b52fcc051.jpg">

### Classification Report For Test1 Datas
<img width="500" alt="4" src="https://user-images.githubusercontent.com/73002780/138612336-818cca47-af16-4385-a8c7-5c8dcc10d0e4.png">

### Classification Report For Test2 Datas
<img width="500" alt="5" src="https://user-images.githubusercontent.com/73002780/138612353-f9daa717-5167-464d-9cf3-3ba42e86c6d7.png">

### Confusion Matrix For Train, Test1 and Test2 Datas
![6](https://user-images.githubusercontent.com/73002780/138612360-2cb76597-a0ed-4917-b6af-d57fa00bac27.jpg)


---
### License Information
This product is open source!

So, Feel free to download, use or make any changes you like to this file
Also, you can contact me with email, instagram and telegram to talk about it.
+ Email: Saeedtajikhk@gmail.com
+ Instagram: saeedthk
+ Telegram: SaeedThk

Your friend Saeed Tajik Hesarkuchak.
