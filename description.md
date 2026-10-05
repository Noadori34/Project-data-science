# Meeting Report – Ultrasound Image Classification

**Next meeting:** 16 oct ?

## Main topic

The main topic of the meeting was our project about using **deep learning to classify ultrasound images**.

Ultrasound is relatively cheap and portable, so it can be useful in primary care. However, one of the problems is that primary care professionals may be able to obtain the ultrasound images but may not have enough experience to interpret them.

Because of this, the images are usually sent to specialists, such as the Nephrology Department, for interpretation. This can increase their workload. The idea of the project is to explore whether a deep learning model could help classify the images and identify possible pathology.

## Dataset

For our project, we currently have around **2,000 ultrasound images**. This is quite a small dataset for training a deep learning model from scratch.

There is also another dataset with around **5,000 kidney ultrasound images**, which has already been used for **pre-training**. We can investigate how to use this pre-trained model with our own dataset instead of training everything from the beginning.

We should also look for other available ultrasound or kidney datasets that could be useful for the project.

## Possible approach

The first idea is to start with a **binary classification problem**, for example, classifying the ultrasound images according to whether there is a pathology or not.

Since we only have around 2,000 images, using the model that has already been pre-trained with the additional kidney dataset could be a good starting point.

We also need to understand the characteristics of the pathology, especially whether the lesions are **localized in a specific area** of the image. Depending on this, we may need to consider **segmentation** as part of the project.

The main techniques to investigate are therefore:

- Deep learning for image classification
- Pre-trained models / transfer learning
- Image segmentation

## Other points

We will need access to a **GPU** for training and testing the deep learning models in the lab.

There is also a document/paper that needs to be **signed**.

## Before the next meeting

Before the next meeting, we should review the reports and dataset.

The next meeting is planned for **16 October 2026**. We need to ask for the exact time and confirm the meeting. It is not necessary for the whole team to attend.

The idea is to have a follow-up approximately **15 days after this meeting** and review our progress.