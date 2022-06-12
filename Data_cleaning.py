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
path = "/Users/timliu/Documents/on the desktop/MSc BA/論文/ARP/European (Re)Insurers/HNR1 GY" #資料夾目錄
save_path = "/Users/timliu/Desktop/output"
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
df_clean_na.to_csv(save_path+'/df.csv')

#%%
# if error of unhashable type: 'list' pop up, 
# run it twice but I don't know why, but no more then three times
# get the value inside the all_participants 
all_participants = [item for sublist in all_participants for item in sublist]
all_participants = list(set(all_participants))
print(all_participants)
# %%
# exclude the title of the participants, i.e.'Roland Vogel, CFO' to 'Roland Vogel" by using re
all_participants = [re.sub(r'\,.*', '', participant) for participant in all_participants]
# exclude the 'Property & Casualty Reinsurance'
all_participants = [re.sub(r'Property & Casualty Reinsurance', '', participant) for participant in all_participants]
# exclude the '[0682QB-E Ulrich Wallin]'
all_participants = [re.sub(r'\[0682QB-E Ulrich Wallin\]', '', participant) for participant in all_participants]
# drop duplicated participants
all_participants = list(set(all_participants))
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

# %%
test = participants_split_df.split(".")
test
# %%
# write a function to split the text by '.' in participants_split_df
def split_text(text):
    text = text.split(".")
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
sentence_split_df


# %%
