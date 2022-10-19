"""
This aim of this project is to be able to give recommendations (finding best matching CV on bench)
on open staffing positions.

TODO:

0. Import and clean text data
1. Process text data
2. Create Synthetic data model
3. Generate Synthetic data
4. Use Synthetic data to train recommendation system for staffing

"""

from src.data_preprocessing import process_data
from src.data_preprocessing import remove_stopwords

# get job and applicant text
final_jobs, final_experience = process_data()


#print(final_experience.head())
print(final_jobs.head())

#final_experience = remove_stopwords(final_experience, "poi_pos_desc", use_stemmer=True)
#print(final_experience.head())
final_jobs = remove_stopwords(final_jobs, "pos_com_city_empType_jobDesc_title", use_stemmer=True)



print(final_jobs.head())


