## classify the language of Tweets with with linear classification and Multilayer Perceptron

* train_dev_set.tsv:
https://docs.google.com/spreadsheets/d/e/2PACX-1vTOZ2rC82rhNsJduoyKYTsVeH6ukd7Bpxvxn_afOi
bn3R-eadZGXu82eCU9IRpl4CK_gefEGsYrA_oM/pub?gid=1863430984&single=true&output=tsv
* test_set.tsv:
https://docs.google.com/spreadsheets/d/e/2PACX-1vT-KNR9nuYatLkSbzSRgpz6Ku1n4TN4w6kKmFLk
A6QJHTfQzmX0puBsLF7PAAQJQAxUpgruDd_RRgK7/pub?gid=417546901&single=true&output=tsv

* use sklearn’s classifier : `SGDClassifier`, `Multinomial Naïve Bayes` and `MLPclassifier` and find best model 

* use `GridSearchCV` to optimize hyperparameter