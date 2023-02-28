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

document = nlp(text)
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
