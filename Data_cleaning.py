#%%
import pdftotext
import pandas as pd
import numpy as np
import re
import os
# ToDo: 
# from cleaning_text import cleaning_text
#wtie a function to import from different file, which is the cleaning_text

def cleaning_text(contents):
    ### Cleaning all the unwanted rows in the transcript
    df = pd.DataFrame(contents)

    # remove the unnessary string
    df[0] = df[0].str.replace('\n','')
    df[0] = df[0].str.replace('Bloomberg Transcript','')
    df[0] = df[0].str.replace('\x0c\n','')
    df[0] = df[0].str.replace('FINAL','')
    df[0] = df[0].str.replace('A - ','')
    df[0] = df[0].str.replace('Q - ','')

    # using re to remove the unnessary string
    def drop_unnessary(x):
        page = re.findall(r'Page \d+ of \d+', x) # 'page ... of ... '
        BIO = re.findall(r'{BIO', x) # '{BIO 18731996 <GO>}'
        Company_Name = re.findall(r'Company N ame:', x) # 'Company N ame: H annover Rueck SE'
        Company_Ticker = re.findall(r'Company Ticker:', x) # 'Company Ticker: H N R1 GR Equity'
        Date = re.findall(r'Date:', x) # Date: 2015-03-10
        if page == [] and BIO == [] and Company_Name == [] and Company_Ticker == [] and Date == []:
            return True
        else:
            return False

    true_false = df[0].apply(lambda x: drop_unnessary(x))
    df = df[true_false]

    # drop the final page declaration
    df = df[df[0] != 'This transcript may not be 100 percent accurate and may contain misspellings and other']
    df = df[df[0] != 'inaccuracies. This transcript is provided "as is", without express or implied warranties of']
    df = df[df[0] != 'any kind. Bloomberg retains all rights to this transcript and provides it solely for your']
    df = df[df[0] != 'personal, non-commercial use. Bloomberg, its suppliers and third-party agents shall']
    df = df[df[0] != 'have no liability for errors in this transcript or for lost profits, losses, or direct, indirect,']
    df = df[df[0] != 'incidental, consequential, special or punitive damages in connection with the']
    df = df[df[0] != 'furnishing, performance or use of such transcript. Neither the information nor any']
    df = df[df[0] != 'opinion expressed in this transcript constitutes a solicitation of the purchase or sale of']
    df = df[df[0] != 'securities or commodities. Any opinion expressed in the transcript does not necessarily']
    # df = df[df[0] != 'reflect the views of Bloomberg LP. ¬© COPYRIGHT 2022, BLOOMBERG LP. All rights']  
    df = df[df[0] != 'reserved. Any reproduction, redistribution or retransmission is expressly prohibited.']
    # ¬© could not be identified, would apply re
    def drop_Bloomberg_mark(x):
        Bloomberg_mark = re.findall(r'reflect the views of Bloomberg LP', x) # 'reflect the views of Bloomberg LP. ¬© COPYRIGHT 2022, BLOOMBERG LP. All rights'
        if Bloomberg_mark == []:
            return True
        else:
            return False

    true_false = df[0].apply(lambda x: drop_Bloomberg_mark(x))
    df = df[true_false]

    # drop the empthy row
    df = df[df[0] != '']
    df = df[df[0] != '']

    return df

def participants_list(df):
    # reset the index to make sure the index is continuous for better processing
    df = df.reset_index(drop=True)

    #  'Company Participants' index
    # df.loc[df[0] == 'Company Participants']
    Participant_start_index = df.index[df.iloc[:,0] == 'Company Participants'].tolist()
    #  'Other Participants' index
    # df.loc[df[0] == 'Other Participants']
    Participant_middle_index = df.index[df.iloc[:,0] == 'Other Participants'].tolist()
    #  'MANAGEMENT DISCUSSION SECTION' index, is the beginning of the management discussion, would stop before this row
    # df.loc[df[0] == 'MANAGEMENT DISCUSSION SECTION']
    Participant_end_index = df.index[df.iloc[:,0] == 'MANAGEMENT DISCUSSION SECTION' ].tolist()
    # try to find the 'MANAGEMENT DISCUSSION SECTION' or 'Presentation' index
    if Participant_end_index == []:
        Participant_end_index = df.index[df.iloc[:,0] == 'Presentation'].tolist()

    print(Participant_start_index, Participant_middle_index, Participant_end_index)

    # make the list of company_paticipants and other_participants
    company_paticipants = df.loc[Participant_start_index[0]+1:Participant_middle_index[0]-1]
    company_paticipants.drop(company_paticipants.index[company_paticipants.iloc[:,0] == ''].tolist(), inplace=True)
    company_paticipants = company_paticipants.values.tolist()

    other_paticipants = df.loc[Participant_middle_index[0]+1:Participant_end_index[0]-1]
    other_paticipants.drop(other_paticipants.index[other_paticipants.iloc[:,0] == ''].tolist(), inplace=True)
    other_paticipants = other_paticipants.values.tolist()

    # print("==========================")
    # print("the company paticipants is: ", company_paticipants)
    # print("==========================")
    # print("the other paticipants is: ", other_paticipants)

    #%%
    # after extract the paticipants, we can drop those information to make the transcript more clear
    df = df.reset_index(drop=True)
    df = df.drop(range(df.index[df.iloc[:,0] == 'Company Participants'].tolist()[0],df.index[df.iloc[:,0].isin(['MANAGEMENT DISCUSSION SECTION','Presentation'])].tolist()[0]+1))

    # drop the first row of the df
    df = df.reset_index(drop=True)
    df = df.iloc[1: , :]


    # reset the index again to make sure the index is continuous for better processing
    df = df.reset_index(drop=True)
    # # save to csv
    # df.to_csv('/Users/timliu/Desktop/output/df.csv')
    return df, company_paticipants, other_paticipants


#%%
# apply the function to processsing all the file
path = "/Users/timliu/Documents/GitHub/data_collecting/BBG_original_file/European (Re)Insurers/HNR1 GY" #資料夾目錄
save_path = "/Users/timliu/Documents/GitHub/data_collecting/output/HNR1 GY_text"
df = pd.DataFrame()
# create a dataframe with 2500 rows
df_clean_na = pd.DataFrame(np.zeros((2500,1)), columns=['index'])

all_participants = []

files= os.listdir(path) #得到資料夾下的所有檔名稱
for file in files:
    if file.endswith(".pdf"):
        print(file)
        # Load PDF
        with open(path+"/"+file, "rb") as f:
            pdf = pdftotext.PDF(f)
        # Save all text to a txt file.
        with open(save_path+"/"+file.replace(".pdf", ".txt"), "w") as f:
            f.write("\n\n".join(pdf))
        # open the text file
        with open(save_path+"/"+file.replace(".pdf", ".txt")) as f:
            contents = f.readlines()
            df_clean = cleaning_text(contents)
            # extract all the participants
            df_pure_text,company_paticipants,other_paticipants = participants_list(df_clean)
            all_participants.append(company_paticipants)
            all_participants.append(other_paticipants)
            # using the file name to set as the dataframe's column name
            # df[f"{files.index(file)}"] = df_clean
            df[f"{files[files.index(file)]}"] = df_pure_text
            df_clean_na[f"{files[files.index(file)]}"] = df[f"{files[files.index(file)]}"].dropna(inplace=False).reset_index(drop=True)

# drop the first column of the df
df_clean_na = df_clean_na.iloc[:,1:]

# save the dataframe
df_clean_na.to_csv('/Users/timliu/Documents/GitHub/data_collecting/output/test/df_test.csv')

#%%
# get the value inside the all_participants 
all_participants = [item for sublist in all_participants for item in sublist]
all_participants = [i[0] for i in all_participants]
print(all_participants)
# %%
# exclude the title of the participants, i.e.'Roland Vogel, CFO' to 'Roland Vogel" by using re
all_participants = [re.sub(r'\,.*', '', participant) for participant in all_participants]
# exclude the 'Property & Casualty Reinsurance'
all_participants = [re.sub(r'Property & Casualty Reinsurance', '', participant) for participant in all_participants]
# exclude the '[0682QB-E Ulrich Wallin]'
all_participants = [re.sub(r'\[0682QB-E Ulrich Wallin\]', '', participant) for participant in all_participants]
# drop the empty string
all_participants = [participant for participant in all_participants if participant != '']
# remove the sapce in the string
all_participants = [participant.strip() for participant in all_participants]
# add the 'Operator' to the list
all_participants.append('Operator')


# %%
new_df = pd.DataFrame()
# identify the len before NaN of each column
for column in df_clean_na.columns:
    # end_index = len(df_clean_na[column])-df_clean_na.isnull().sum(axis = 0)[column]-1
    # # 這邊要注意775是NaN 所以774還是有值的
    # identify all the rows in df with all_participants in it
    both_participants_row_index = df_clean_na[df_clean_na[column].isin(all_participants)].index.tolist()
    # # append the end_index to the end of both_participants_row_index
    # both_participants_row_index.append(end_index)
    # apply the both_participants_row_index to the df_clean_na['participants']
    new_df[column] = df_clean_na[column]
    new_df[f"participants_{column}"] = df_clean_na[column].apply(lambda x: x if x in all_participants else np.nan)
    # fill the NaN with the value of the previous row
    new_df[f"participants_{column}"] = new_df[f"participants_{column}"].fillna(method='ffill')
    # # exclude the row if pure_df[column]==pure_df[f"participants_{column}"]
    # pure_df = pure_df[pure_df[column] != pure_df[f"participants_{column}"]]


# %%
pure_df = pd.DataFrame()
# identify the len before NaN of each column
for column in df_clean_na.columns:
    # exclude the row if pure_df[column]==pure_df[f"participants_{column}"]
    pure_df = new_df[new_df[column] != new_df[f"participants_{column}"]]
# drop the column if the column start with participants
pure_df = pure_df.drop(pure_df.columns[pure_df.columns.str.startswith('participants_')], axis=1).T

# append the text of each roll into one string by using s.str.cat(sep='. ')
pure_df = pure_df.apply(lambda x: x.str.cat(sep='. '), axis=1)
# change the pure_df to dataframe
pure_df = pd.DataFrame(pure_df)
# rename the column
pure_df.columns = ['meeting_text']
# extract the index as column from the text
pure_df['file_name'] = pure_df.index
# extract the date from the index column
pure_df['date'] = pure_df['file_name'].apply(lambda x: x.split('_')[0])
# change the date column to datetime
pure_df['date'] = pd.to_datetime(pure_df['date'])
# reset the index
pure_df = pure_df.reset_index(drop=True)
pure_df

#save the dataframe
pure_df.to_csv('/Users/timliu/Documents/GitHub/data_collecting/output/test/pure_df.csv')



# %%
# take two of the first column as test df
single_meeting_df = new_df.iloc[:,:2]
# rename the second column to 'participants'
single_meeting_df.columns = ['text', 'participants']
# drop the NaN
single_meeting_df = single_meeting_df.dropna(inplace=False)
# if the text is empty, drop it
single_meeting_df = single_meeting_df[single_meeting_df['text'] != '']
# drop if 'text' == 'participants'
single_meeting_df = single_meeting_df[single_meeting_df['text'] != single_meeting_df['participants']]

list_participants = [] 
list_participants.append(single_meeting_df['participants'].unique().tolist())
list_participants = list_participants[0]

# create participants_split_df with column of 'participants' and 'text'
participants_split_df = pd.DataFrame()

for i in range(len(list_participants)):
    processing_df = single_meeting_df.copy()
    # 萃取出所有的participantsj中Operator說的話
    processing_df['participants'] = processing_df['participants'].apply(lambda x: x if x == f"{list_participants[i]}" else np.nan)
    # drop the NaN
    processing_df = processing_df.dropna(inplace=False)
    processing_df = processing_df.apply(''.join).to_frame().T
    processing_df['participants'] =  f"{list_participants[i]}"
    participants_split_df = participants_split_df.append(processing_df, ignore_index=True)


participants_split_df

################################################################################################################################################################
# %%
# write a function to split the text by '.' in participants_split_df
def split_text(text):
    text = text.split(". ")
    return text
# apply the function to the participants_split_df
participants_split_df['text'] = participants_split_df['text'].apply(lambda x: split_text(x))
participants_split_df

sentence_split_df = pd.DataFrame()
for i in range(len(participants_split_df)):
    sentence_list = participants_split_df['text'].iloc[i]
    sentence_split_single_df = pd.DataFrame (sentence_list, columns = ['sentence'])
    sentence_split_single_df['participants'] = participants_split_df['participants'].iloc[i]
    sentence_split_df = sentence_split_df.append(sentence_split_single_df, ignore_index=True)
# drop if the 'sentence' is empty
sentence_split_df = sentence_split_df.dropna(inplace=False)

sentence_split_df

################################################################################################################################################################
#%%
# count the frequency of participants in the sentence_split_df
participants_count_df = sentence_split_df.groupby(['participants']).count()
participants_count_df
################################################################################################################################################################

# %%
paragrapg_df = new_df.iloc[:,:2]
# drop the NaN
paragrapg_df = paragrapg_df.dropna(inplace=False)
#rename the second column to 'participants', first column to 'text'
paragrapg_df.columns = ['text', 'participants']
# if the text is empty, drop it
paragrapg_df = paragrapg_df[paragrapg_df['text'] != '']
# reset the index
paragrapg_df = paragrapg_df.reset_index(drop=True)
# if 'text' == 'participants', get the index of the row
paragrapg_index = paragrapg_df[paragrapg_df['text'] == paragrapg_df['participants']].index.tolist()

# +1 for every value in paragrapg_index
start_paragrapg_index = []
for i in range(len(paragrapg_index)):
    start_paragrapg_index.append(paragrapg_index[i]+1)
# disregard the last value in the list
start_paragrapg_index = start_paragrapg_index[:-1]
print(len(start_paragrapg_index))
# -1 for every value in paragrapg_index
end_paragrapg_index = []
for i in range(len(paragrapg_index)):
    end_paragrapg_index.append(paragrapg_index[i])
# disregard the first value in the list
end_paragrapg_index = end_paragrapg_index[1:]
print(len(end_paragrapg_index))

# extracct the text of the paragrapg_df between end_paragrapg_index and start_paragrapg_index
paragraph_split_df = pd.DataFrame()
for i in range(len(start_paragrapg_index)):
    paragraph = paragrapg_df.iloc[start_paragrapg_index[i]:end_paragrapg_index[i]]
    # merge the paragraph to one cell 
    paragraph_text = paragraph.apply(''.join).to_frame().T
    # appemd the paragraph_text['text'].iloc[:,0] to the paragraph_split_df
    paragraph_split_df = paragraph_split_df.append(paragraph_text, ignore_index=True)

# look up for paragrapg_df['participants'] from the start_paragrapg_index
participants = paragrapg_df.iloc[start_paragrapg_index,1].to_frame()
# reset the index
participants = participants.reset_index(drop=True)

paragraph_split_df['participants'] = participants
paragraph_split_df
################################################################################################################################################################



# %%
# export the four dataframes to csv
# incliding paragraph_split_df, participants_split_df, sentence_split_df, and participants_count_df
final_df_for_NLP_path = '/Users/timliu/Documents/GitHub/data_collecting/output/final_df'
participants_split_df.to_csv(f'{final_df_for_NLP_path}/participants_split_df.csv', index=False)
sentence_split_df.to_csv(f'{final_df_for_NLP_path}/sentence_split_df.csv', index=False)
participants_count_df.to_csv(f'{final_df_for_NLP_path}/participants_count_df.csv', index=False)
paragraph_split_df.to_csv(f'{final_df_for_NLP_path}/paragraph_split_df.csv', index=False)

# %%
from nltk import word_tokenize, pos_tag
# import nltk
# nltk.download()
def determine_tense_input(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)

    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    return(tense)

# %%
# apply the function to the sentence_split_df
sentence_split_df['tense'] = sentence_split_df['sentence'].apply(lambda x: determine_tense_input(x))
# decode the sentence_split_df['tense'] to different column
sentence_split_df['tense_future'] = sentence_split_df['tense'].apply(lambda x: x['future'])
sentence_split_df['tense_present'] = sentence_split_df['tense'].apply(lambda x: x['present'])
sentence_split_df['tense_past'] = sentence_split_df['tense'].apply(lambda x: x['past'])

# if the tense_future > tense_present > tense_past, then the tense is future
# else if the tense_present > tense_future > tense_past, then the tense is present
# else if the tense_past > tense_future > tense_present, then the tense is past
sentence_split_df['tense'] = sentence_split_df.apply(lambda x: 'future' if x['tense_future'] > x['tense_present'] > x['tense_past'] else 'present' if x['tense_present'] > x['tense_future'] > x['tense_past'] else 'past', axis=1)
# else if the tense_past = tense_future = tense_present, then the tense is unknown
sentence_split_df['tense'] = sentence_split_df.apply(lambda x: 'unknown' if x['tense_past'] == x['tense_future'] == x['tense_present'] else x['tense'], axis=1)
# disregard the tense_future, tense_present, and tense_past
sentence_split_df = sentence_split_df.drop(columns=['tense_future', 'tense_present', 'tense_past'])


# %%
# count the amount of rowa of sentence_split_df['tense']=='unknown'
unknown_sentence_count = sentence_split_df[sentence_split_df['tense'] == 'unknown'].shape[0]
# count the amount of rowa of sentence_split_df['tense']=='future'
future_sentence_count = sentence_split_df[sentence_split_df['tense'] == 'future'].shape[0]
# count the amount of rowa of sentence_split_df['tense']=='present'
present_sentence_count = sentence_split_df[sentence_split_df['tense'] == 'present'].shape[0]
# count the amount of rowa of sentence_split_df['tense']=='past'
past_sentence_count = sentence_split_df[sentence_split_df['tense'] == 'past'].shape[0]

# print the result
print(f'unknown_sentence_count: {unknown_sentence_count}')
print(f'future_sentence_count: {future_sentence_count}')
print(f'present_sentence_count: {present_sentence_count}')
print(f'past_sentence_count: {past_sentence_count}')

#%% # snetiment analysis
import numpy as np
import pandas as pd

import re
import string 

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

import nltk 
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords          # module for stop words that come with NLTK
nltk.download('stopwords')
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

# twitter_samples.fileids()
# documents
docs_negative = [(t, "neg") for t in twitter_samples.strings("negative_tweets.json")]
docs_positive = [(t, "pos") for t in twitter_samples.strings("positive_tweets.json")]
print("==========================================================")
print(f'There are {len(docs_negative)} negative sentences.')
print(f'There are {len(docs_positive)} positive sentences.')

# spliting dataset 
train_set = docs_negative[:3500] + docs_positive[:3500]
test_set = docs_negative[3500:4250] + docs_positive[3500:4250]
valid_set = docs_negative[4250:] + docs_positive[4250:]

# clean text
def process_text(text):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    #text = text.str
    text = str(text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'#', '', text)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    text_tokens = tokenizer.tokenize(text)

    text_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  
                word not in string.punctuation): 
            stem_word = stemmer.stem(word)  # stemming word
            text_clean.append(stem_word)
            
    sentence = ' '.join(text_clean)
    
    return sentence

# categorical label
def cat_label(label):
    if label == 'neg':
        value = -1
    elif label == 'pos':
        value = 1
    return value 

# split for x and y 
def xy(dataset):
    df = pd.DataFrame(dataset, columns = ['text', 'label'])
    df['text_clean'] = df['text'].apply(lambda r: process_text(r))
    #df['categorical_label'] = df.label.factorize()[0]
    df['categorical_label'] = df['label'].apply(lambda r: cat_label(r))

    x = df.text_clean
    y = df.categorical_label

    return x, y

# dataframe
x_train, y_train = xy(train_set)
x_test, y_test = xy(test_set)
x_valid, y_valid = xy(valid_set)

## using the naive bayes classifier
model = Pipeline([
    ('bow',CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("==========================================================")
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))
print(accuracy_score(y_pred,y_test))

# Apply into earnings call sentence
# import dataset
path = '/Users/timliu/Documents/GitHub/data_collecting/df_for_NLP/sentence_split_df.csv'
df_sentence = pd.read_csv(path)
# df_sentence.head()

# drop participant columns as we dont need it
# df_sentence = df_sentence.drop(['participants'], axis=1)

# check NaN values
print("==========================================================")
print(df_sentence.isnull().sum())

# delete NaN rows
df_sentence = df_sentence.dropna()  

# clean text for sentiment analysis
df_sentence['text_clean'] = df_sentence['sentence'].apply(lambda r: process_text(r))
# df_sentence.head(5)

# making prediction
prediction = model.predict(df_sentence.text_clean)
prediction_label = np.array(['positive' if p==1 else 'negative' for p in prediction])
df_sentence['prediction_label'] = prediction_label
df_sentence['sentiment_score'] = prediction
# df_sentence.head()

print("==========================================================")
print(Counter(df_sentence['prediction_label']))

# %%
