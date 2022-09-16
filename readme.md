### ㄴ01. Language Identification with linear and non-linear model
* dataset 
  * tweetter messages with 69 different languages ([download here](https://docs.google.com/spreadsheets/d/e/2PACX-1vTOZ2rC82rhNsJduoyKYTsVeH6ukd7Bpxvxn_afOibn3R-eadZGXu82eCU9IRpl4CK_gefEGsYrA_oM/pub?gid=1863430984&single=true&output=tsv))
* package 
  * pandas and sklearn 
* description  
  * Implement the language identifier which can detect the language of tweets 
  * Use simple one-hot encoding and feature engineering that can improve model performance
  * Train data with linear model: [SGD Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) and non-linear model: [MLP classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
  * Find the best hyperparameter combination (loss function, Regularisation, early stopping for SGD classifier / different layer sizes, activation functions, solvers, early stopping, vectoriser parameters for MLP classifier)
  * Compute a confusion matrix for error analysis

### ㄴ02. CBOW Word Embeddings
* dataset 
  * tripadvisor_hotel_reviews.csv :  ([download here](https://drive.google.com/file/d/1ihP1HZ8YHVGGIEp1RHxXdt3PPIi12xvL/view?usp=sharing))
  * scifi.txt: ([download here](https://drive.google.com/file/d/10ehW4jZND3QA29v9aNboYUett5-swuNe/view?usp=sharing)) 
* package 
  * pandas, nltk and pytorch
* description 
  * Train CBOW2 embedding (context width of 2 in both directions) for both datasets
  * Use embedding size of 50 ([This paper](https://aclanthology.org/I17-2006/) provides an insight on how to choose a minimum embedding size while still obtaining
useful representations)
  * Evaluate embedding by computing nearest neighbour distance

### ㄴ03. Language Identification with CNN
* description
  * try out different hyperparameter combinations (e.g. optimizer, learning rate, dropout ration, # of filters, strides, kernel sizes, different pooling strategies, batch sizes) 



### ㄴ04. Named Entity Recognition using BERT


### ㄴ05. Topic modeling 

