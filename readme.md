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
  * Implement CBOW2 embedding (context width of 2 in both directions) with pytorch
  * Use embedding size of 50 ([This paper](https://aclanthology.org/I17-2006/) provides an insight on how to choose a minimum embedding size while still obtaining
useful representations)
  * Evaluate embedding by computing nearest neighbour distance

### ㄴ03. Language Identification with CNN
* description
  * try out different hyperparameter combinations (e.g. optimizer, learning rate, dropout ration, # of filters, strides, kernel sizes, different pooling strategies, batch sizes) 



### ㄴ04. Named Entity Recognition using Transformers
* dataset    
   polyglot-ner dataset ([learn more](https://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html)) - import directly with python [`dataset` library](https://huggingface.co/docs/datasets/quickstart)
* desciption 

  * Use HuggingFace’s BertForTokenClassification-class and initialize it with a pretrained Hugging Face BERT-base model of your chosen language.
  * create 3 fine-tuned versions of the system:
     1) Fine-tuned with 1’000 sentences
     2) Fine-tuned with 3’000 sentences
     3) Fine-tuned with 3’000 sentences and frozen embeddings
  * compute f1-micro and f1-macro scores for each fine-tuned model 

* useful link 
  * fine-tuning pre-trained model : https://huggingface.co/docs/transformers/training
  * Transformers docs: https://huggingface.co/transformers/index.html
  * Datasets docs: https://huggingface.co/docs/datasets/
  * BertTokenizer: https://huggingface.co/transformers/model_doc/bert.html?highlight=berttokenizer#transformers.BertTokenizer 
  * BertModel: https://huggingface.co/transformers/model_doc/bert.html?highlight=bertmodel#transformers.BertModel
  * BertForSequenceClassification: https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforsequenceclassification#transformers.BertForSequenceClassification 
  * BertForTokenClassification: https://huggingface.co/transformers/model_doc/bert.html?highlight=bertfortokenclassification#transformers.BertForTokenClassification 
  * On the model outputs from different transformers-versions: https://huggingface.co/transformers/migration.html


### ㄴ05. Topic modeling with LDA and CTMs 
* dataset 
  * titles of publications in three time-periods; before 1990, from 1990 to 2009, and 2010 onwards ([download here](https://dblp.uni-trier.de/xml/dblp.xml.gz))
* description 
  * Experiment with different numbers of topics and with different ways of preprocessing
  * CTM is Combined Topic Models which is a pre-trained language model ([paper](https://aclanthology.org/2021.acl-short.96/) / [colab tutorial](https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing))
  * Comparing the coherence of the topics produced by CTM with the topics produced by LDA
  
* useful link 
  * [A 6min presentation of the paper by one of the authors.](https://underline.io/lecture/25716-pre-training-is-a-hot-topic-contextualized-document-embeddings-improve-topic-coherence)
  * Code: [https://github.com/MilaNLProc/contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)


### ㄴ06. Paper review : Attention is all you need 

