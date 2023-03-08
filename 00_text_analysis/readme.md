## This is my cheat-sheet of the most frequently used basic text analysis technique 💫
last updated : 28.Feb.2023
### Data load 
* load the zip file from local to colab
  *  https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92
  ```
  
  ```
* parse raw html 
  ```python
  import urllib.request as urllib # Python's module for accessing web pages
  url = 'https://goo.gl/VRF8Xs' # shortened URL for court case
  page = urllib.urlopen(url) # open the web page

  html = page.read() # read web page contents as a string
  
  from bs4 import BeautifulSoup # package for parsing HTML
  soup = BeautifulSoup(html, 'lxml') # parse html of web page
  text = soup.get_text() # get text (remove HTML markup)
  ```


### tokenizer 
* spacy 
```
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(text) # tokens
sentences = list(doc.sents)
```
more detail information available with spacy such as token.text, token.pos_, token.dep_

* nltk
```
from nltk.tokenize import word_tokenize
```

* gensim
```
from gensim.utils import simple_preprocess
```

### text cleaning 
#### fast punctuation removal
```python
from string import punctuation

punc_remover = str.maketrans('','',punctuation) 
text_nopunc = text_lower.translate(punc_remover)
```

#### normalize number
```python
no_numbers = [t for t in tokens if not t.isdigit()]
# keep if not a digit, else replace with "#"
norm_numbers = [t if not t.isdigit() else '#' 
                for t in tokens ]
```

#### stopwords
```python
nltk.download('stopwords')
from nltk.corpus import stopwords

stoplist = stopwords.words('english') 

# keep if not a stopword
nostop = [t for t in norm_numbers if t not in stoplist]
```

#### stemming 
```
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english') # snowball stemmer, english
# remake list of tokens, replace with stemmed versions
tokens_stemmed = [stemmer.stem(t) for t in tokens]
```

#### lemmatizing
```
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
wnl.lemmatize('corporation'), wnl.lemmatize('corporations')
```

#### one-shot
```python
from string import punctuation
translator = str.maketrans('','',punctuation) 
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def normalize_text(doc):
    "Input doc and return clean list of tokens"
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower() # all lower case
    nopunc = lower.translate(translator) # remove punctuation
    words = nopunc.split() # split into tokens
    nostop = [w for w in words if w not in stoplist] # remove stopwords
    no_numbers = [w if not w.isdigit() else '#' for w in nostop] # normalize numbers
    stemmed = [stemmer.stem(w) for w in no_numbers] # stem each word
    return stemmed
    
df['tokens_cleaned'] = df['text'].apply(normalize_text)
```

#### with gensim 
```python
from gensim.utils import simple_preprocess # lowercase, tokenized, punctuations/numbers removed

df['tokens_simple'] = df['text'].apply(simple_preprocess)
```

### POS tagging 
```python
nltk.download('averaged_perceptron_tagger')
from nltk.tag import perceptron 
from nltk import word_tokenize
tagger = perceptron.PerceptronTagger()
tokens = word_tokenize(text)
tagged_sentence = tagger.tag(tokens)
```

### Text data summary (word counts and frequency distribution)
* frequency distribution
```python
from collections import Counter

freqs = Counter()
for i, row in df.iterrows():
    freqs.update(row['opinion_text'].lower().split())

freqs.most_common()[:20] # can use most frequent words as style/function words
```

### simple sentiment analysis 
* spacy
* hugging face
```python
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

pipe = pipeline("sentiment-analysis")

class OpinionDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
    def __len__(self):
        return len(df)

    def __getitem__(self, i):
        return df.iloc[i]["opinion_text"][:512] # BERT max seq length


dataset = OpinionDataset(df)
sentiments = []

for out in tqdm(pipe(dataset, batch_size=16), total=len(dataset)):
        if out['label'] == "NEGATIVE":
            sentiments.append(-1*out['score'])
        else:
            sentiments.append(out['score'])

```


### wordnet 
will be updated 
