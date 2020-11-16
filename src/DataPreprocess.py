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
    manualCleaning = jsonFile['ManualCleaning']
    
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

# 4. Remove Improper Records
def deleteImproperRecords():
    print("Deleting Improper Records ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']
    deleteRecordList = deleteRecords['RecordList']

    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))    
    columnList = deleteRecordList.keys()

    for i in columnList:
        print("Deleting Records where {}={}".format(i, deleteRecordList[i]))
        df = df[df[i] != deleteRecordList[i]]

    print("Dropping columns {}".format(columnList))
    df.drop(columnList, axis = 1, inplace = True) 
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

    for eachAttribute in attributesToMerge['Features']:
        newAttribute = eachAttribute['Name']
        print("Procssing", newAttribute)
        
        df[newAttribute] = [eachAttribute['Default']] * len(df.index)

        for eachCondition in eachAttribute['Conditions']:
            expressionList = []

            for col, val in eachCondition['If'].items():
                expressionList.append("({}['{}']=={})".format(df.name, col, val))

            df[newAttribute][eval(" & ".join(expressionList))] = eachCondition['Then']['Value']

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

# 7. Manual Filtering 
def selectiveDelete():
    print("Manual deletion of records ...")
    csvRootPath = csvDetails['CSVRootPath']
    combinedCSV = csvDetails['CombinedCSV']

    df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))
    
    for eachColumn, values in manualCleaning.items():
        for eachValue in tqdm(values):
            df = df[df[eachColumn] != eachValue]

    df.to_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"), index = False)
    
    print("Some records deleted")
    print("-" * 100) 

resizeImagesTo128x128()
normalizeAttributesFile()
combineExcels()
deleteImproperRecords()
mergeAttributes()
dropColumns()
selectiveDelete()

csvRootPath = csvDetails['CSVRootPath']
combinedCSV = csvDetails['CombinedCSV']

df = pd.read_csv(os.path.join(csvRootPath, combinedCSV).replace("/", "\\"))
print("FINAL CSV DIMENSIONS:", df.shape)