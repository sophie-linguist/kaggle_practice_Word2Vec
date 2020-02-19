## kaggle_practice_Word2Vec : Bag of Words Meets Bag of Popcorn

이 repository는 kaggle의 Bag of Words Meets Bag of Popcorn를 연습하기 위해 만든 것입니다. 꼼꼼하게 살펴보려는 목적으로 튜토리얼의 내용을 번역하였으며, 의역 및 생략이 있습니다. part1과 part2-3은 주피터 노트북으로 올려 두었습니다.


----------

Google의 Word2Vec은 딥러닝에서 영감을 얻은 방법론으로, 단어들의 의미에 초점을 두고 있습니다. Word2Vec은 단어 간의 의미와 의미 관계를 이해하려고 시도합니다. 이 방법론은 어떤 면에서는 recurrent neural net 또는 deep neural net과 같은 딥 러닝의 접근법과 유사하지만, 계산 방식은 더 효율적입니다. 

이 튜토리얼 대회는 감성 분석(sentiment analysis)을 위한 Word2Vec이 중심이 됩니다. 사람들은 언어로 감정을 표현하는데, 종종 그 표현들은 비꼬기(sarcasm), 한 가지 이상의 의미로 이해되도록 하기(ambiguity), 말놀이하기(plays on words) 등에 의해 모호해지며,이는 사람과 컴퓨터 둘 다에게 오해를 초래합니다. 이 튜토리얼에서는 어떻게 Word2Vec이 이와 유사한 문제에 적용될 수 있는지 탐색합니다. 

딥러닝 기법은, 사람의 뇌 구조에서 영감을 얻었으며, 연산 능력의 발전에 의해 가능해졌습니다. 딥러닝 기법은 영상 인식, 음성 처리, 자연 언어 과제에서의 획기적인 결과를 바탕으로 일종의 흐름이 되었습니다.


### Tutorical Overview

이 튜토리얼은 자연 언어 처리를 위한 Word2Vec을 시작하는 데 도움을 줄 것입니다. 이 튜토리얼의 목표는 다음과 같습니다. 

1) 기초적인 자연언어 처리
    - 이 튜토리얼의 Part 1은 초심자를 위해 만들어졌습니다. 기초적인 자연 언어 처리 기법을 다루며, Part 1은 이후 튜토리얼에서도 필요합니다.  
2) 텍스트 이해를 위한 딥러닝
    - Part 2와 Part 3에서, 우리는 Word2Vec을 이용한 모델을 학습하는 방법과 학습 결과인 단어 벡터들을 감성 분석에서 사용하는 방법에 대해 보다 깊이 들어갑니다. 
    
    - 딥러닝은 매우 빠르게 진화하는 분야로, 많은 업적들이 아직 출판되지 않았거나, 학술 논문으로만 존재합니다. 튜토리얼의 Part 3은 규범이기보다는 더 탐색적인 것입니다. 즉, Word2Vec 사용법을 제공하는 것이기보다는 오히려 Word2Vec을 이용하여 몇 가지 실험을 하는 것입니다. 


우리는 이 목표를 달성하기 위해 IMDB sentiment analysis data set을 사용합니다. 이 Data set은 100,000개의 영화 리뷰로, 긍정과 부정 리뷰를 포함하며, 각 리뷰는 여러 개의 문단으로 이루어집니다. 

### Acknowledgements

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). 
"Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### Metric

Area under the ROC curve

### Submission Instructions

25,000개의 열과 헤더가 있는 쉼표로 구분된 파일로 제출합니다. 이 파일에는 id, sentiment 열이 있어야 합니다. sentiment 열에는 예측 결과가 들어갑니다: 1은 긍정 리뷰이고, 0은 부정 리뷰입니다. 

### What Is Deep learning

"Deep learning"이라는 용어는 2006년에 만들어졌으며, 여러 개의 비선형적 층위를 포함하고 특성들(features)의 위계를 학습할 수 있는 학습 알고리즘을 가리킵니다 [1].  

현대의 대부분의 기계 학습은 좋은 결과를 내기 위해서 feature engineering 또는 어느 수준의 Domain knowlege를 필요로 합니다. 이는 딥러닝 체계에서는 해당되지 않습니다. 대신에, 딥러닝 알고리즘들은 특성의 위계를 자동적으로 학습하며, 이는 추상화의 증가 수준에서의 대상들을 나타냅니다. 비록 딥러닝 알고리즘의 기본적인 구성 요소들은 오랜 시간 동안 주변에 있었지만 딥러닝은 현재 시점에서 인기가 높아지고 있습니다. 이는 딥러닝 알고리즘의 기본적 구성 요소들 대부분은 오랜 시간 동안 주변에 있었지만, 연산 능력(computing power)의 발전, 하드웨어 가격의 하락, 기계 학습 연구의 발전 등 많은 이유 덕분입니다. 

딥러닝 알고리즘은 그들의 구조(feed-foward, feed-back, or bi-directional)와 학습 프로토콜(순수한 지도 학습, hybrid, 또는 비지도학습)에 따라 범주화됩니다 [2]. 

[1] "Deep Learning for Signal and Information Processing", by Li Deng and Dong Yu (out of Microsoft)

[2] "Deep Learning Tutorial" (2013 Presentation by Yann LeCun and Marc'Aurelio Ranzato)

### Data Set

주석 데이터 셋은 50,000개의 IMDB movie review로 구성되며, 감성 분석을 목적으로 선택된 것입니다. 리뷰의 감성은 두 개로 분류되는데, IMDB 점수가 5 이하인 것의 감성 코드는 0이고, 7 이상인 리뷰의 감성 코드는 1입니다. 각 영화별 리뷰는 30개 이하입니다. 이 중 훈련 데이터 셋에 있는 25,000개, test set에 있는 25,000개가 리뷰한 영화는 서로 (완전히) 다릅니다. 이외에, 50,000개의 IMDB 리뷰는 점수 라벨을 제공하지 않습니다. 

### File Description


* labeledTrainData 
    - 점수 라벨이 붙어 있는 훈련 셋. 이 파일은 탭으로 구분되어 있으며, 헤더 행이 있습니다. 25,000개의 행은 id, 감성 라벨, 각 리뷰에 대한 텍스트 열로 구성됩니다. 

* testData 
    - 테스트 셋. 탭으로 구분된 파일이며, 헤더 행이 있습니다. 25,000개의 행은 id 와 각 리뷰에 해당하는 텍스트를 포함합니다. 이 과제는 테스트 셋의 각 리뷰에 대한 감성을 예측하는 것입니다. 
    
* unlabeledTrainData 
    - 추가된 훈련 셋이며, 라벨이 없습니다. 탭으로 구분된 파일이고, 헤더가 있으며, 각 리뷰의 id와 텍스트를 포함합니다. 
* sampleSubmission 
    - 쉼표로 구분된 제출 파일 샘플입니다.
    
### Data Fields

* **id** - 각 리뷰의 고유 ID
* **sentiment** -  리뷰의 감성, 1은 긍정 리뷰이고, 0은 부정 리뷰
* **review** -  리뷰의 텍스트

* 전체 코드는 https://github.com/wendykan/DeepLearningMovies 에 있습니다. 


### Setting Up Your System

* pandas
* numpy
* scipy
* scikit-learn 
* Beautiful Soup
* NLTK
* Cython
* gensim
* Word2Vec

Kaggle 페이지의 코드는 python 2.7을 기준으로 작성되었기 때문에, 정리하는 과정에서 python 3 이상에서 실행되도록 약간 수정하였습니다. 


### Part 4 Comparing Deep-And-Non-Deep-Learning-Methods

왜 Bag of Words가 더 나은가?
이 튜토리얼에서 가장 큰 이유는 벡터의 평균을 구하고 centroid를 사용하면서 단어의 순서 정보를 잃어버리고 Bag of words 개념과 유사해지기 때문입니다. 즉, 세 가지 방법론이 실질적으로 같습니다.  

시도해 볼 만한 몇 가지 방법들이 있습니다. 
우선, 더 많은 텍스트를 훈련하면 성능을 크게 향상시킬 수 있습니다. 구글의 결과는 십억 단어 이상의 코퍼스를 학습한 단어 벡터에 기초합니다. 우리가 사용한 라벨이 있거나 없는 데이터 셋은 1800만 단어에 불과합니다. C를 이용하여 만들어진 Word2Vecdms pre-trained 모델이 있으며, python으로 불러들일 수 있습니다.  

다음으로, 출판된 논문을 보면, 분포적 단어 벡터 기법은 단어 주머니 모델보다 더 성능이 우수합니다. 이 논문에서 paragraph vector라고 불리는 알고리즘이 IMDB 데이터셋에 사용되었고, 가장 좋은 결과를 산출하였습니다. 여기서 시도한 방법론들이 단어의 순서를 잃어버리는 반면, Paragraph Vectors는 단어 순서 정보를 보존합니다. 
