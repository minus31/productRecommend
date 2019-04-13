# productRecommend
- Content-based fashion items recommendation

This repo is a framework implemented with Keras for image retreval and content-based recommendation task. 

#### Project inspiration

dfdf

#### Core Function

1. Input a snapshot image and output nine of items which 

[example]
<img src="">

2. Input an item image and output nine of snapshots which 

[example]
<img src="">

#### Data source 

From Musinsa.com(https://www.musinsa.com/)

Musinsa posts street snapshots and provides their related items' information like below.

<img src="https://www.dropbox.com/s/6di2thhbxwx9yjq/site_example.png?raw=1">

I extract **600** posts for training **300** posts for test. I regard images in a same post as images in a same class. 

#### Model details 

* Backbone model : ResNet50(pretrained on the ImageNet dataset)

* Use Combined Global Descriptor inspired by this [paper](), [summary]()
  - In this case, two diffrent descriptors used : SPoC and GeM(implement them with reference of this [paper](https://arxiv.org/pdf/1711.02512.pdf))

- Label smoothing and temperature scaling for classification loss.
- Use ArcFace([paper](https://arxiv.org/abs/1801.07698), [summary](https://minus31.github.io/2019/04/08/ArcFace/)) loss function to get fully discriminative global descriptor.

#### Things to improve

1. I couldn't design a proper metric for my dataset to estimate the performance.

2. Recommendation from the origin site is not actually recommendation. It's more like commercial to promote same brand(like the figure above). In many of cases, images in a same class are not actually related in a manner of visual. 

3. I gathered extra information such as '이름(나이)', '직업', '태그', '관련브랜드', '스타일' and etc. In '스타일', there are 8 kinds of style for total. And it seems like Musinsa.com has its major style that the site promote the most. Thus, I think that it could infulence on the result somehow.  

   <img src="https://www.dropbox.com/s/heqdgpoxgdcyv2n/style_frequency.png?raw=1">

   