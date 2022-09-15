### ㄴ01. Language Identifier 
* Implement the language identifier of twitter datasets using linear model: [SGD Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), [Multinomial Naïve Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) and non-linear model: [MLP classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
* Find the best hyperparameter combination (loss function, Regularisation, early stopping for SGD classifier / different layer sizes, activation functions, solvers, early stopping, vectoriser parameters for MLP classifier)
* Use a confusion matrix for error analysis

### ㄴ02. Word Embeddings
* dataset 
  * tripadvisor_hotel_reviews.csv : https://drive.google.com/file/d/1ihP1HZ8YHVGGIEp1RHxXdt3PPIi12xvL/view?usp=sharing
  * scifi.txt: https://drive.google.com/file/d/10ehW4jZND3QA29v9aNboYUett5-swuNe/view?usp=sharing
* package 
  * pytorch
* description 
  * train CBOW2 embedding (context width of 2 in both directions) for both datasets
  * use embedding size of 50 ([This paper](https://aclanthology.org/I17-2006/) provides an insight on how to choose a minimum embedding size while still obtaining
useful representations)


