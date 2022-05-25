## classify the language of Tweets
- build a language classification model works on the character level

### with linear classification and Multilayer Perceptron
* use sklearn’s classifier : `SGDClassifier`, `Multinomial Naïve Bayes` and `MLPclassifier` and find best model 
* use `GridSearchCV` to optimize hyperparameter

### with CNN
* find the optimal model by adjusting : optimizer, learning rate, dropout, # of filters, strides, kernel size, pooling and batch size
* keep track of the loss to interrupt early if a model does not converge 
* evaluate model by accuracy and F1-macro
