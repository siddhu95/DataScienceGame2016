# DataScienceGame2016
solution to the competition https://inclass.kaggle.com/c/data-science-game-2016-online-selection

 Data Science Game 2016

								----   Team devenish-IIT Kharagpur


Problem Statement and Competition details
For the second year in a row the Data Science Game, a data science and machine learning international student contest, was held in Paris, France in September, 2016. 143 teams, three times as much as in 2015, representing more than 50 universities from 28 different countries (view the map), faced real-life business challenges. To stand out in this competition, students were asked to conceive and implement predictive models in order to solve Big Data-related issues.

The 143 teams were first cut down to 20 through an online qualification phase which lasted for about a month. During this stage, candidates were given a challenge based on solar energy production optimization.

In order to map such production potential in France, the Data Science Game partnered with Etalab - the French public agency in charge of Open Data and Data use in the French administration. The State Agency created the OpenSolarMap project which provides satellite images of about 80,000 building roofs. Automated classification of roof orientation is a true challenge for Etalab.

For the 2016 Data Science Game participants, the challenge was to develop an algorithm which would recognise the orientation of a roof based on a satellite photograph by building on more than 10,000 roof photographs categorized using crowdsourcing.

OUR SOLUTION

Preprocessing

Augment dataset by rotating all images by three discrete angles apart from original (90,180,-90,0-orig) and change their labels accordingly
1->1,2,1
2->2,1,2
3->3,3,3
4->4,4,4

Rescaled image’s shorter side to 32 and took central crop of size 32 on the longer side resulting in 32 by 32 RGB images
Contrast Norm, RGB to YUV followed by normalization
Horizontal Flipping, RandomCropping used for on-the-fly data augmentation in code
Dataset made uniform across classes by using repetitive sampling on classes with poor strength

CODE: In file image_preproc_util.py, we have some of these implemented.Others are in the lua code of our Torch implementations.

image_preproc_util.py also has a lot of utility functions for arranging images into folders, converting them into numpy format etc.

We also used provider.lua from https://github.com/szagoruyko/cifar.torch/blob/master/ to carry out the rgb to yuv conversions and normalization.

Architecture

Based on Wide-Residual Networks by Sergey Zagoruykosergey and Nikos Komodakisnikos (https://arxiv.org/pdf/1605.07146v1.pdf)
Used a  28-10 WRN
Trained from scratch 
Implemented pseudo learning by labelling unlabelled data and include 30 % of that along with 70% of supervised data once the model trained to around 81 % accuracy.


CODE: Architecture of Resnet : https://github.com/szagoruyko/wide-residual-networks

Apart from WRN, we also tried VGG and ResNet. However, this(WRN) was our best effort at solving the problem.




