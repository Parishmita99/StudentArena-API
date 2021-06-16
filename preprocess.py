import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import nltk

df = pd.read_csv("dataset/dice_com-job_us_sample.csv.zip")


def identify_tokens(row):
    jobdescription = row['jobdescription']
    tokens = nltk.word_tokenize(jobdescription)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df['description'] = df.apply(identify_tokens, axis=1)
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))                  

def remove_stops(row):
    my_list = row['description']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

df['meaningful'] = df.apply(remove_stops, axis=1)
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

df['meaningfulstring']=df['meaningful'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))

df['meaningfulstring']=df['meaningfulstring'].str.lower()
df['skills']=df['skills'].str.lower()
df['jobtitle']=df['jobtitle'].str.lower()
df['skills']=df['skills'].fillna(' ')

# see below and see job description strings are removed from skills
for i in df.index:
  if(df['skills'].iloc[i]=="see below" or df['skills'].iloc[i]=="(see job description)" ):
    df['skills'].iloc[i]=" "

#skills and meaningfulstring is combined into one column named combined    
def parameter(row):    
    return row['skills']+" "+row['meaningfulstring']

df['combined']=df.apply(parameter,axis=1) 
df.insert(0,'Job id',range(1,1+len(df)))
df=df.drop(['jobid','uniq_id','description', 'meaningful','meaningfulstring'], axis = 1)
df.to_csv('Preprocessed.csv')
