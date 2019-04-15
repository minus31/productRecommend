## Content-based fashion items recommendation
This repo is a framework implemented with Keras for image retreval and content-based recommendation task. I call images of people with **snapshot images** and images of items with **item images**

* Demo site link : http://n31.closedclothes.tk/

#### Core Function

**CASE 1** : Input a snapshot image and output six of items which 

*[example 'part']*

<img src="https://www.dropbox.com/s/k4935w5a3y4hphk/ex_part.png?raw=1">

**CASE 2** : Input an item image and output six of snapshots which 

*[example 'snap']*

<img src="https://www.dropbox.com/s/wsty63ebn1inbr8/ex_snap.png?raw=1">

#### Data source 

From Musinsa.com(https://www.musinsa.com/)

Musinsa posts street snapshots and provides their related items' information like below.

<img src="https://www.dropbox.com/s/6di2thhbxwx9yjq/site_example.png?raw=1">

I have extracted **600** posts for training, **300** posts for test. I regard images in a same post as images in a same class. 

#### Model details 

* Backbone model : ResNet50(pretrained on the ImageNet dataset)

* Use Combined Global Descriptor inspired by this [paper](https://arxiv.org/pdf/1903.10663.pdf), [summary](https://minus31.github.io/2019/04/08/CGD/)
  - In this case, two diffrent descriptors used : SPoC and GeM(implement them with reference of this [paper](https://arxiv.org/pdf/1711.02512.pdf)

- Label smoothing and temperature scaling for classification loss.
- Use ArcFace([paper](https://arxiv.org/abs/1801.07698), [summary](https://minus31.github.io/2019/04/08/ArcFace/)) loss function to get fully discriminative global descriptor.

#### Troubles 

1. In inference database, there are several duplicated item images.

   - I am getting rid of the images every time I find the one. 

   <img src="https://www.dropbox.com/s/2bmfu3f2lytswdx/Screenshot%202019-04-14%2023.36.39.png?raw=1">

2. I gathered extra information such as '이름(나이)', '직업', '태그', '관련브랜드', '스타일' and etc. In '스타일', there are 8 kinds of style for total. And it seems like Musinsa.com has its major style that the site promote the most. Thus, I think that it could infulence on the result somehow.  

   <img src="https://www.dropbox.com/s/heqdgpoxgdcyv2n/style_frequency.png?raw=1">

3. Recommendation from the origin site is not actually recommendation. It's more like commercial to promote some brands(like the figure below). In many of cases, images in a same class are not actually related in a manner of visual. 

<img src="https://www.dropbox.com/s/zg0i5k6xvtebqyo/Screenshot%202019-04-15%2012.28.19.png?raw=1"> 

#### Things to improve

1. I want to put style related information, such as '스타일', '상의', into the network in the form of BoW.
2. I couldn't design a satisfactory metric.
3. Post processing  
   - Average query expansion 
   - Database-side feature augmentation

#### Usage

Process of training and updating db are implemented in `main.py`.

**Train**

```python main.py --train True --batch_size 64 --epoch 1000 --dataset_path {path where data located}```

**Update DB**

- for part images

```python main.py --updateDB True --DB_path ./data/db/db/ --model_path ./checkpoint/0 --reference_path ./reference_part.p```

- for snapshot images

```python main.py --updateDB True --DB_path ./data/db/snap/ --model_path ./checkpoint/0 --reference_path ./reference_snap.p```