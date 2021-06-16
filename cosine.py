import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("dataset/Preprocessed.csv.rar")
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined'])
cosine_sim = cosine_similarity(count_matrix)

def get_index_from_job_type(title):
    return df[df['jobtitle']==title]['Job id'].values[0]

jobs_user_likes ="java architect - denver, co - fulltime"
Job_id = get_index_from_job_type(jobs_user_likes)
similar_jobs = list(enumerate(cosine_sim[Job_id]))
sorted_similar_jobs = sorted(similar_jobs,key=lambda x:x[1],reverse=True)[1:]


i=0
print("Top 10 similar jobs to "+jobs_user_likes+" are:\n")
for element in sorted_similar_jobs:
    print(df.iloc[(element[0])])
    i=i+1
    if i>10:
        break
