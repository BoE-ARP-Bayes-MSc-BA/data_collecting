#%%
import pdftotext
import pandas as pd
import numpy as np
import re
import os
# ToDo: 
# from cleaning_text import cleaning_text

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
    # df = df[df[0] != 'This transcript may not be 100 percent accurate and may contain misspellings and other']
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

    other_paticipants = df.loc[Participant_middle_index[0]+1:Participant_end_index[0]-1]
    other_paticipants.drop(other_paticipants.index[other_paticipants.iloc[:,0] == ''].tolist(), inplace=True)

    print("==========================")
    print("the company paticipants is: ", company_paticipants)
    print("==========================")
    print("the other paticipants is: ", other_paticipants)

    #%%
    # after extract the paticipants, we can drop those information to make the transcript more clear
    df = df.reset_index(drop=True)
    df = df.drop(range(df.index[df.iloc[:,0] == 'Company Participants'].tolist()[0],df.index[df.iloc[:,0].isin(['MANAGEMENT DISCUSSION SECTION','Presentation'])].tolist()[0]+1))

    # drop the first row of the df
    df = df.reset_index(drop=True)
    df = df.iloc[1: , :]
    # 'Q4 2014 Earnings Call'

    # reset the index again to make sure the index is continuous for better processing
    df = df.reset_index(drop=True)
    # # save to csv
    df.to_csv('/Users/timliu/Desktop/output/df.csv')
    return df, company_paticipants, other_paticipants


#%%
# apply the function to processsing all the file
path = "/Users/timliu/Documents/on the desktop/MSc BA/論文/ARP/European (Re)Insurers/HNR1 GY" #資料夾目錄
save_path = "/Users/timliu/Desktop/output"
df = pd.DataFrame()

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
            df[f"{files.index(file)}"] = df_clean

# %%
df_test = df.iloc[:,0].to_frame()
df = df_test
# %%
# if couldn't find the date, we would set the date as the meeting happening
df,company_paticipants,other_paticipants = participants_list(df_per_meeting)
df