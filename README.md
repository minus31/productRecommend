### [Eng](#eng) [Kor](#kor)

<a id="kor"></a>

#### Contents

- [Project motivation](#motivation)
- [Core Function](#func)
- [Model details](#detail)
- [Self-feedback](#disscusion)
- [Things to improve](#improve)
- [Troubles](#trouble)

## Content-based fashion items recommendation

이 레포는 케라스를 이용해 content-based 추천시스템을 구현한 것입니다. 사용한 기술은 이미지검색 기술을 이용했습니다. 다만 제가 구성한 데이터셋은 각 클래스들이 시각적으로 유사한 것이 아니라 전문가들의 추천으로 구성되어 있기 때문에, 이미지 검색 시스템이라기 보다 추천시스템이라는 용어를 사용했습니다.  그리고 content-based의 의미는 이미지 간의 관련도가 높은 것을 추천한다는 의미에서 사용했습니다. 이 글에서 사람들의 사진을 **Snapshot**, 각 패션 아이템의 사진을 **Item**이라고 하겠습니다. 밑에서 자세히 설명하겠지만 이 시스템은 Snapshot을 입력으로 받고 그 snapshot과 관련이 있는 fashion item을 추천하는 기능을 합니다.

- Demo site link : http://n31.closedclothes.tk/

<a id="motivation"></a>

#### Project motivation

박진영, 마크 주커버그, 아만시오 오르테가 등 성공하고 유명한 사람들이 매일 같은 옷을 입는 다는 것을 여러 매체를 통해 접했습니다. 그 이유로는 옷을 고를 때 자신 모르게 사용되는 에너지와 시간을 줄이고 자신의 삶을 더 효율적으로 사용하기 위해서라고 합니다. 저 또한 삶의 효율성에 관심이 있기 때문에 실천 해봤습니다만, 저에게는 맞지 않는 방법이라 느꼈습니다. 옷을 잘입고 TPO(Time Place Occasion)에 맞는 착장을 하는 것이 저에게 있어 그 날 하루의 기분과 태도를 제어하는 방법  중 하나이기 때문입니다. 세상에는 저처럼 매일 옷을 잘 입음으로써 하루의 컨디션을 조절하는 사람이 많습니다. 그러나 실제로 옷을 고르면서, 특히 쇼핑을 하면서 많은 에너지가 소비되는 것은 사실입니다. 어떻게하면 저 같은 사람을 도울 수 있을까 생각했고, 자신이 멋있다고 생각하는 코디의 사진을 입력하면 그 사진에서 입고 있는 아이템을 검색해주는 어플리케이션을 기획했습니다.

무신사 닷컴(https://www.musinsa.com/) (한국의 큰 온라인 의류 셀렉샵) 에는 다음 그림처럼 길거리 스냅샷을 올리고 그 스냅샷의 아이템과 연관된 아이템을 소개하는 서비스가 있습니다.

<img src="https://www.dropbox.com/s/6di2thhbxwx9yjq/site_example.png?raw=1">

이것들이 시각적으로 유사한 것은 아닌 듯 합니다. 다음 예처럼 시각적으로는 공통점이 아예 없는 것들도 소개 되곤합니다.

<img src="https://www.dropbox.com/s/zg0i5k6xvtebqyo/Screenshot%202019-04-15%2012.28.19.png?raw=1">

다만 이렇게 소개되는 아이템은 무신사 직원 혹은 관련 브랜드의 직원(패션 전문가)들이 선택한 것으로 해당 이미지 스냅샷에 대해 추천된 아이템으로 생각 할 수 있었습니다. 만약 이 추천 이미지들을 검색할 수 있다면, 그것이 또한 좋은 추천시스템이 될 수도 있겠다는 생각을 했습니다. 이 프로젝트를 위해 저는 600개의 포스트를 훈련용으로 300개를 테스트용으로 크롤링했습니다.

<a id="func"></a>

#### Core Function

기능은 두가지 입니다. 처음에는 **스냅샷을 입력하면 관련된 아이템 6개의 이미지를 출력** 하는 기능을 구현했습니다. 추후에 생각을 해보니 다음과 같은 상황이 생각났습니다. "내일 어제 산 핑크 운동화를 신고 싶은데 어떻게 코디 하면 좋을까?" 이런 상황에서 사용할 수 있도록 **반대로 아이템을 입력하면 관련된 스냅샷을 출력** 하는 기능을 추가 했습니다. 두 기능은 사이트 좌측 상단의 태그로 구분됩니다.

**CASE 1** : Input a snapshot image and output six of items

[example]

<img src="https://www.dropbox.com/s/w6jj3giiokcnkcz/Screenshot%202019-04-15%2012.48.20.png?raw=1">

**CASE 2** : Input an item image and output six of snapshots

[example]

<img src="https://www.dropbox.com/s/n6rsf5ad6wlgxf0/Screenshot%202019-04-15%2012.47.25.png?raw=1">

<a id="detail"></a>

#### Model details

CNN을 통과해 이미지의 설명자(Descriptor)를 만들고 그 유사도를 기준으로 관련된 이미지를 출력합니다. 이 설명자의 유사도가 연관된 이미지 간에 높아지도록 지도학습을 진행했습니다.

- Backbone model : ResNet50(pretrained on the ImageNet dataset)
- 이  논문( [paper](https://arxiv.org/pdf/1903.10663.pdf), [summary](https://minus31.github.io/2019/04/08/CGD/))을 참고하여 Combined Global Descriptor를 사용했습니다. 여기서는 다음 두 가지의 풀링 방법을 사용했습니다.
  - SPoC and GeM(구현은 이 [paper](https://arxiv.org/pdf/1711.02512.pdf) 논문을 참고했습니다.)

- Classification loss에 대해서는 Label smoothing 과 temperature scaling을 적용했습니다.
- 더 분별력있는 descriptor를 얻기위해서 Augular margin loss 중, ArcFace([paper](https://arxiv.org/abs/1801.07698), [summary](https://minus31.github.io/2019/04/08/ArcFace/)) loss function을 사용했습니다. 

이 후 시도 할 예정이거나 시도 중인 것은 다음과 같습니다.

<a id="disscusion"></a>

#### Self-feedback

* 프로젝트에 대해서

  이 프로젝트의 본래 목적은 의류 기존의 이미지 검색 기술을 응용해서 기존의 시각적인 유사도가 기준이 아닌 전문가의 시각에서의 연관도를 기준으로 패션아이템을 추천하는 것입니다.

  다만 이 기술의 적용을 보여주기 위해서 제가 만든 사이트는 위 목적과는 어울리지 않는다고 생각합니다. 보편적으로 사용자 입장에서 입력하는 이미지를 넣었을 때 궁금한 것은 이 이미지 안에 사람이 어떤 브랜드의 옷을 입었는지, 어디서 저 것과 비슷한 옷을 구할 수 있는지 일 것 입니다. 근데 제가 만든 시스템은 시각적으로는 다소 유사하지 않는 결과를 낼 가능성이 큽니다.

  따라서 이 시스템은 소비자의 능동적인 행동을 요구하는 서비스보다, 이전에 산 옷, 이전에 입었던 옷 등을 토대로 자동으로 추천하는 기능으로 사용되는 것이 더 잘 어울릴 것입니다.

* ArcFace loss 에 대해서

  Classification loss는 괜찮지만, ArcFaceloss는 30epochs 부터 validation loss가 증가하기 시작합니다. 아주 빨리 overfitting이 시작된다고 느꼈고 제가 현재 metric으로 쓰고 있는 mAP 기준으로 해도 overfitting문제를 보였습니다. 더많은 Photometric distortion을 적용해도 비슷한 결과를 출력합니다. 근본적인 문제는 데이터셋인 것 같습니다. 데이터셋에 클래스가 600개이고 하나의 클래스별로 많게는 20개 ~ 적게는 3개의 이미지가 있습니다. 이 데이터셋이 너무 작은 것이 문제는 데이터를 좀 더 확보하는 것으로 해결 될 수 있다고 생각하고, 클래스별 개체수 차이는 Oversampling이나 batch를 구성하는 방법에 대해서 고민을 해봐야합니다. 그리고 ArcFaceloss 외의 다른 angular margin loss를 사용하는 것도 시도 해볼만 하다고 생각합니다.(다만 GPU를 사용하는데 드는 비용이 점점 부담 스러워 집니다...)

<a id="improve"></a>

#### Things to improve

1. Post processing 기법을 inference시에 적용할 예정입니다.
   - Average query expansion
   - Database-side feature augmentation
2. 스냅샷에 대한 출력으로 나오는 아이템이 현재는 사용자의 의도와는 관계없이 그저 descriptor의 유사도를 기준으로만 선정되었습니다. 이에 대해서 사용자가 스타일, 의류의 카테고리등을 선택할 수 있게하는 기능을 추가 할 예정입니다.

<a id="trouble"></a>

#### Troubles

1. 아직 이와 관련된 뚜렷한 문제점을 발견하지는 않았습니다만, 무신사에서 다루는 의류의 스타일을 보면 다음 그래프에서 확인 할 수 있듯이 "스트리트/힙합", "심플/캐주얼" 이 나머지에 비해 상당히 많습니다. 따라서 모델이 저 두 스타일에 다소 편향됬을 수 있습니다.

   <img src="https://www.dropbox.com/s/heqdgpoxgdcyv2n/style_frequency.png?raw=1">

#### Usage

학습과 평가 그리고 사이트에 사용하기 위한 reference 파일은 모두 `main.py`에 구현되어 있습니다.

**Train**

```python main.py --train True --batch_size 64 --epoch 1000 --dataset_path {path where data located}```

**Evaluation**

```python main.py --eval True --model_path ./checkpoint/model_weight```

**Update DB**

```python main.py --updateDB True --model_path ./checkpoint/model_weight```

---

<a id="eng"></a>

## Content-based fashion items recommendation

This repo is a framework implemented with Keras for image retreval and content-based recommendation task. In the system, photos of people are called **Snapshots** and photos of clothing are called **Items**. When someone inputs an image of snapshot(or items), this system find items(or snapshot) related to what the one entered  from database. Although it just outputs six images for now, It can be utilized to recommend products to sell and shorten time for customers to reach to what they are looking for.

* Demo site link : http://n31.closedclothes.tk/

#### Project motivation

Many celebrities such as JYP, Mark Zuckerberg, do wearing the same clothes everyday to reduce unnecessary time and energy and improve their efficiency of life. I've practiced it too and I realize it isn't suitable for me. There are many people like me who like to be well dressed and control their condition of a day. I wanted to help somehow that people.

Musinsa.com(https://www.musinsa.com/) is one of the biggest online select shop in Korea. Musinsa posts street snapshots and provides their related items' information like below.

<img src="https://www.dropbox.com/s/6di2thhbxwx9yjq/site_example.png?raw=1">



I cannot say that these proposed items are related in a manner of visual. As shown in an image below, some posts seem very different.

<img src="https://www.dropbox.com/s/zg0i5k6xvtebqyo/Screenshot%202019-04-15%2012.28.19.png?raw=1">

But that item suggestion could be regarded as a recommendation. Thus, if I make a system to suggest them automatically, I thought it would be a good recommendation system. For this project I extracted **600** posts for training, **300** posts for test.

#### Core Function

**CASE 1** : Input a snapshot image and output six of items

[example]

<img src="https://www.dropbox.com/s/w6jj3giiokcnkcz/Screenshot%202019-04-15%2012.48.20.png?raw=1">

**CASE 2** : Input an item image and output six of snapshots

[example]

<img src="https://www.dropbox.com/s/n6rsf5ad6wlgxf0/Screenshot%202019-04-15%2012.47.25.png?raw=1">

#### Model details

* Backbone model : ResNet50(pretrained on the ImageNet dataset)

* Use Combined Global Descriptor inspired by this [paper](https://arxiv.org/pdf/1903.10663.pdf), [summary](https://minus31.github.io/2019/04/08/CGD/)
  - In this case, two diffrent descriptors used : SPoC and GeM(implement them with reference of this [paper](https://arxiv.org/pdf/1711.02512.pdf)

- Label smoothing and temperature scaling for classification loss.
- Use ArcFace([paper](https://arxiv.org/abs/1801.07698), [summary](https://minus31.github.io/2019/04/08/ArcFace/)) loss function to get fully discriminative global descriptor.

#### Troubles

1. I gathered extra information such as '이름(나이)', '직업', '태그', '관련브랜드', '스타일' and etc. In '스타일', there are 8 kinds of style for total. And it seems like Musinsa.com has its major style that the site promote the most. Thus, I think that it could infulence on the result somehow.

   <img src="https://www.dropbox.com/s/heqdgpoxgdcyv2n/style_frequency.png?raw=1">


#### Things to improve

1. Input style related information, such as '스타일', '상의', into the network in the form of BoW.
2. Post processing
   - Average query expansion
   - Database-side feature augmentation

#### Usage

Process of training, evaluation and updating db are implemented in `main.py`.

**Train**

```python main.py --train True --batch_size 64 --epoch 1000 --dataset_path {path where data located}```

**Evaluation**

```python main.py --eval True --model_path ./checkpoint/model_weight```

**Update DB**

```python main.py --updateDB True --model_path ./checkpoint/model_weight```
