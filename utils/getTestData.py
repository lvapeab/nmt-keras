import os
import tarfile

# https://pypi.org/project/wget/
import wget

# Download and extract the files

def downloadAndExtractFiles(cachePath,*args):
    for url in args:
        fileName = os.path.basename(os.path.normpath(url))
        checkPath = cachePath +  fileName
        if os.path.exists(checkPath):
            print(checkPath + ' already exists')
        else :
            print('\n Downloading ' + fileName + ' from ' + 'url')
            wget.download(url, cachePath)
    print('\n')

baseCacheDir = 'cache/'
task = 'wmt15/'
rawtask = 'raw-'+task

train = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_training.tar.gz'
dev = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_dev.tar.gz'
test = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/task1_en-es_test.tar.gz'
label = 'http://www.quest.dcs.shef.ac.uk/wmt15_files/gold/Task1_gold.tar.gz'

os.makedirs(baseCacheDir, exist_ok=True)
cachePath = baseCacheDir + rawtask
os.makedirs(cachePath, exist_ok=True)

downloadAndExtractFiles(cachePath,train,dev,test,label)

for file in os.listdir(cachePath):
    if file.endswith(".tar.gz"):
        tar = tarfile.open(cachePath+file, "r:gz")
        print('Extracting: ' + file + ' to ' + cachePath)
        tar.extractall(path=cachePath)
        tar.close()

# make the test data
exampleDir = 'examples/'+task
os.makedirs(exampleDir, exist_ok=True)

totalLines = 500 # total number of lines to take from example data
for f in os.listdir( cachePath ):
    if f.endswith(".hter") or f.endswith(".pe") or f.endswith(".source") or f.endswith(".target"):
        file_in = cachePath+f
        file_out = exampleDir+f
        print('Copying first ' + str(totalLines) + ' lines of ' + file_in + ' to ' + file_out)
        with open(file_in) as file:
            lines = file.readlines()
            lines = [l for i, l in enumerate(lines) if i <= totalLines-1]
            with open(file_out, "w") as f1:
                f1.writelines(lines)
