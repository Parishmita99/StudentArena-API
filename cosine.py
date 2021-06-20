import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("dataset/Preprocessed.csv.zip",nrows=6000)
cv = CountVectorizer()
df.fillna(" ")

count_matrix = cv.fit_transform(df['combined'])
cosine_sim = cosine_similarity(count_matrix)

def get_index_from_job_type(title):
    return df[df['jobtitle']==title]['Job id'].values[0]

def recommend(jobs_user_likes):
    if jobs_user_likes not in df['jobtitle'].unique():
        return 'Job not in Database'
    else:
        Job_id = get_index_from_job_type(jobs_user_likes)
        similar_jobs = list(enumerate(cosine_sim[Job_id]))
        sorted_similar_jobs = sorted(similar_jobs,key=lambda x:x[1],reverse=True)[1:]
        sorted_similar_jobs = sorted_similar_jobs[1:21]
    
        job_indices = [i[0] for i in sorted_similar_jobs]
        job_id = df['Job id'].iloc[job_indices]
        job_title = df['jobtitle'].iloc[job_indices]
        job_desc = df['jobdescription'].iloc[job_indices]
        job_company = df['company'].iloc[job_indices]
        recommendation_data = pd.DataFrame(columns=['Job_Id','Job title','Job Description','Company'])

        recommendation_data['Job_Id'] = job_id
        recommendation_data['Job title'] = job_title
        recommendation_data['Job Description'] = job_desc
        recommendation_data['Company'] = job_company
        return recommendation_data.to_dict('records')

jobs_user_likes ="information security engineer"
recommendation_data= recommend(jobs_user_likes)
print(recommendation_data)
