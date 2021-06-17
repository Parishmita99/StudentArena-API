import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("dataset/Preprocess.csv.zip")
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined'])
cosine_sim = cosine_similarity(count_matrix)

def get_index_from_job_type(title):
    return df[df['jobtitle']==title]['Job id'].values[0]

def recommend(job):
  jobs_user_likes =job
  Job_id = get_index_from_job_type(jobs_user_likes)
  similar_jobs = list(enumerate(cosine_sim[Job_id]))
  sorted_similar_jobs = sorted(similar_jobs,key=lambda x:x[1],reverse=True)[1:]
  print("Top 10 similar jobs to "+job+" are:\n")
  return sorted_similar_jobs
k=recommend("java architect - denver, co - fulltime")
i=0

y_pred=[]
for element in k:
    if i==0:
      y_pred=df.iloc[(element[0])]
    #print(df.iloc[(element[0])])
    i=i+1
    if i>0:
      y_pred=y_pred.append(df.iloc[(element[0])])
    if i>10:
        break
for i in y_pred:
  print(i)
