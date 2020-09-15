import pandas as pd
import shutil

# 1. Combine Excels to single CSV
def combineExcels(): # Mandatory
    shutil.copy(r'E:\\sem 6\\project\\datasets\\list_attr_celeba.csv', r'E:\\sem 6\\project\\datasets\\merged.csv')
    data = pd.read_csv (r'E:\\sem 6\\project\\datasets\\merged.csv')

    # https://www.datacamp.com/community/tutorials/pandas-to-csv?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034358&utm_targetid=dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9300430

    # read_file.to_csv (r'Path to store the CSV file\File name.csv', index = None, header=True)
    # print(type(data))
    # print(data['image_id'])
    # print(data.head(2)) # Number of rows to select
    print(data.columns)

# 2. Remove blurred images
def deleteBlurredImages():
    pass

# 3. Manual Filtering 
def selectiveDelete():
    pass

combineExcels()
# deleteBlurredImages()
selectiveDelete()