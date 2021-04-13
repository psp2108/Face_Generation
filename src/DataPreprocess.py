import pandas as pd
import os
import json
from tqdm import tqdm
from PIL import Image

with open("config.json", "r") as f:
    jsonFile = json.load(f)
    csvDetails = jsonFile['CSVDetails']
    dataset = jsonFile['ImageDetails']
    deleteRecords = jsonFile['DeleteRecords']
    attributesToMerge = jsonFile['MergeAttributes']
    columnsToDrop = jsonFile['DiscardAttributes']
    
# 1. Change the resolution of images
def resizeImagesTo128x128():
    print("Resizing Images ...")
    picsDirectory = os.path.join(dataset['ImageRootPath'], dataset['ImageImages']).replace("/", "\\")
    picsProcessedDirectory = os.path.join(dataset['ImageRootPath'], dataset['ImageProcessedImages']).replace("/", "\\")

    if not os.path.exists(picsProcessedDirectory):
        os.makedirs(picsProcessedDirectory)

        originalWidth = 178
        originalHeight = 218
        difference = (originalHeight - originalWidth) // 2

        newWidth = 128
        newHeight = 128

        cropRect = (0, difference, originalWidth, originalHeight - difference)

        for eachImage in tqdm(os.listdir(picsDirectory)):
            image = Image.open(os.path.join(picsDirectory, eachImage)).crop(cropRect)
            image.thumbnail((newWidth, newHeight), Image.ANTIALIAS)
            image.save(os.path.join(picsProcessedDirectory, eachImage))
        
        print("Resized Images")
    else:
        print("Output folder '"+ picsProcessedDirectory +"' aready exists. Images are either already resized or delete the folder.")
    print("-" * 100)

# 2. Normalize values
def normalizeAttributesFile():
    print("Normalizing Data ...")
    csvRootPath = csvDetails['CSVRootPath']
    toNormalize = csvDetails['NormalizeData']['AttributesFile']
    df = pd.read_csv(os.path.join(csvRootPath, toNormalize['From']).replace("/", "\\"))

    rawData = df.values
    rawData[:,1:] = (rawData[:,1:]+1)/2
    df = pd.DataFrame(rawData, columns=df.columns)
   
    df.to_csv(os.path.join(csvRootPath, toNormalize['To']).replace("/", "\\"), index = False)

    print(toNormalize['From'], "normalized")
    print("-" * 100)

# 3. Combine Excels to single CSV
def combineExcels(): 
    print("Combining CSV files ...")
    csvRootPath = csvDetails['CSVRootPath']
    csvList = csvDetails['CSVListToCombine']
    combinedCSV = csvDetails['CombinedCSV']
    if len(csvList) == 0:
        print("No files selected to combine")
        return

    df = pd.read_csv(os.path.join(csvRootPath, csvList[0]).replace("/", "\\"))
    print("Reading file '"+ csvList[0] +"'")

    for i in range(1,len(csvList)):
        print("Reading file '"+ csvList[i] +"'")
        csvFile = pd.read_csv(os.path.join(csvRootPath, csvList[i]).replace("/", "\\"))
        df = df.merge(csvFile, how = 'outer', on = 'image_id')

    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)

    print("Files combined")
    print("-" * 100)

# 4. Remove Improper Records and Manual Filtering too 
def deleteImproperRecords():
    print("Deleting Improper Records ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']

    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))    

    tempIndex = df.index
    print(tempIndex)

    for conditions in deleteRecords:
        listOfSets = []
        print("Deleting Records where {}".format(conditions))
        for cols,values in conditions.items():
            tempIndex = df[(df[cols].isin(values))].index
            listOfSets.append(set(list(tempIndex)))

        finalSet = listOfSets[0]
        for i in range(1, len(listOfSets)):
            finalSet = finalSet.intersection(listOfSets[i])
        
        df.drop(list(finalSet), inplace = True)

    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)
    
    print("Records Deleted")
    print("-" * 100)

# 5. Merge Columns
def mergeAttributes():
    print("Merging Attributes ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']

    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))   
    df.name = "df"

    for eachAttribute in attributesToMerge:
        newAttribute = eachAttribute['Name']
        print("Procssing", newAttribute)
        
        df[newAttribute] = [eachAttribute['Default']] * len(df.index)

        for eachCondition in eachAttribute['Conditions']:
            expressionList = []

            for col, val in eachCondition['If'].items():
                expressionList.append("({}['{}']=={})".format(df.name, col, val))
            
            df.loc[eval(" & ".join(expressionList)), [newAttribute]] = eachCondition['Then']

    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)

    print("All Attributes merged")
    print("-" * 100)

# 6. Manual Filtering 
def dropColumns():
    print("Dropping non importatnt Columns ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']

    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))
    
    for eachColumn in columnsToDrop:
        df.drop(eachColumn, axis = 1, inplace = True) 
        print("Dropped", eachColumn)

    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)

    print("All Attributes merged")
    print("-" * 100) 

def balanceGender():
    print("Balancing Gender ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']
    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))
    
    maleIndices = list(df[(df["Male"]==1)].index)
    femaleIndices = list(df[(df["Male"]==0)].index)

    deleteCount = abs(len(maleIndices) - len(femaleIndices))
    deleteFrom = []

    if len(femaleIndices) > len(maleIndices):
        deleteFrom = femaleIndices
    elif len(maleIndices) > len(femaleIndices):
        deleteFrom = maleIndices

    newList = []
    if deleteCount:
        stepSize = len(deleteFrom) / deleteCount
        for i in range(deleteCount):
            newList.append(deleteFrom[int(i * stepSize)])

    df.drop(newList, inplace = True)
    
    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)
    
    print("Gender Balanced")
    print("-" * 100)

resizeImagesTo128x128()
normalizeAttributesFile()
combineExcels()
deleteImproperRecords()
mergeAttributes()
dropColumns()
balanceGender()

csvRootPath = csvDetails['CSVRootPath']
combinedCSV = csvDetails['CombinedCSV']

df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))
print("FINAL CSV DIMENSIONS:", df.shape)