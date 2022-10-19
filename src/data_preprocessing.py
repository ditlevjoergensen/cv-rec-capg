
import pandas as pd
from functools import reduce
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys  
sys.path.insert(0, 'C:\Projects\Staffing')

def process_data():
    """
    Process raw text data into a-zA-z lowercase text
    No NLP processes have been used yet. 

    returns two corpuses one for job descriptions and one for applicant experiences
    """




    jf_df = pd.read_csv("data/Combined_Jobs_Final.csv")

    #print(jf_df.head())
    #print(list(jf_df))

    #print(jf_df['Requirements'].head())
    #print(jf_df.isnull().sum())

    #print(jf_df.notnull().sum())
    cols = ['Job.ID', 'Title', 'Position', 'Company', 'City', 'State.Name', 'Job.Description', 'Employment.Type', 'Education.Required']
    final_jobs = jf_df[cols]
    final_jobs.columns = ['Job.ID', 'Title', 'Position', 'Company', 'City', 'State Name', 'Job_Description', 'Empl_type', 'Edu_req']

    #replacing nan with thier headquarters location
    final_jobs['Company'] = final_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
    final_jobs.loc[final_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
    final_jobs.loc[final_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
    final_jobs.loc[final_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
    final_jobs.loc[final_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
    final_jobs.loc[final_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
    final_jobs.loc[final_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'
    final_jobs.loc[final_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
    final_jobs.loc[final_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
    final_jobs.loc[final_jobs.Company == 'Volvo Group', 'City'] = 'Washington'
    final_jobs.loc[final_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'

    # add data into last NAN's
    final_jobs['Empl_type']=final_jobs['Empl_type'].fillna('Full-Time/Part-Time')

    # Creating the corpus of the jobs
    final_jobs["pos_com_city_empType_jobDesc_title"] = final_jobs["Position"].map(str) + " " + final_jobs["Company"] +" "+ final_jobs["City"]+ " "+final_jobs['Empl_type']+" "+final_jobs['Job_Description']+" " + final_jobs['Title']

    #removing unnecessary characters and convert to lowercase
    final_jobs['pos_com_city_empType_jobDesc_title'] = final_jobs['pos_com_city_empType_jobDesc_title'].str.replace('[^a-zA-Z \n\.]'," ").str.lower() 

    #print(final_jobs['pos_com_city_empType_jobDesc_title'].head(1000))
    final_jobs = final_jobs[['Job.ID', 'pos_com_city_empType_jobDesc_title']]

    # Creating the corpus of the applications


    #Position of interest
    poi = pd.read_csv("data/Positions_Of_Interest.csv", sep=',')
    poi = poi.sort_values(by='Applicant.ID')
    #print(poi.head())
    # There is no need of application and updation becuase there is no deadline mentioned in the website ( assumption) hence we are droping unimportant attributes
    poi = poi.drop('Updated.At', 1)
    poi = poi.drop('Created.At', 1)

    #cleaning the text
    poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")
    poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.lower()
    poi = poi.fillna(" ")
    #print(poi.head(20))

    poi = poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()
    #print(poi.head(20))

    # Experiences
    exper_applicant = pd.read_csv("data/Experience.csv")
    #print(exper_applicant.head())
    #taking only Position
    exper_applicant = exper_applicant[['Applicant.ID','Position.Name', 'Job.Description']]

    #cleaning the text
    exper_applicant['Position.Name'] = exper_applicant['Position.Name'].str.replace('[^a-zA-Z \n\.]',"").str.lower()
    exper_applicant['Job.Description'] = exper_applicant['Job.Description'].str.replace('[^a-zA-Z \n\.]',"").str.lower()

    #print(exper_applicant.head())

    exper_applicant = exper_applicant.sort_values(by='Applicant.ID')
    exper_applicant = exper_applicant.fillna(" ")

    #print(exper_applicant.head())

    #adding same rows to a single row
    exper_applicant_position = exper_applicant.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()
    exper_applicant_desc = exper_applicant.groupby('Applicant.ID', sort=False)['Job.Description'].apply(' '.join).reset_index()
    #print(exper_applicant_position.head(20))
    #print(exper_applicant_desc.head(20))
    data_frames = [poi, exper_applicant_position, exper_applicant_desc]
    final_experience = reduce(lambda  left, right: pd.merge(left,right,on=['Applicant.ID'],
                                                how='outer'), data_frames).fillna(" ")



    final_experience["poi_pos_desc"] = final_experience["Position.Of.Interest"].map(str) + " " + final_experience["Position.Name"] +" "+ final_experience["Job.Description"]
    final_experience = final_experience[['Applicant.ID', 'poi_pos_desc']]
    #print(final_experience.head())
    #print(final_jobs.head())

    return final_jobs, final_experience




def remove_stopwords(df, column_name, use_stemmer=True):
    """
    remove stopwords from a corpus
    """

    
    stop = stopwords.words('english')
    df[column_name] = df[column_name].astype(str)
    only_text = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    print(only_text.head())
    only_text = only_text.apply(lambda x : filter(None,x.split(" ")))
    print(only_text.head())
    if use_stemmer:
        stemmer =  PorterStemmer()
        only_text = only_text.apply(lambda x : [stemmer.stem(y) for y in x])
    #print(only_text.head())

    df["text"] = only_text
    return df
