from zipfile import ZipFile
import os
import glob
import boto3

# Unzip, renaming and moving files.
zips = glob.glob(r'*.zip)
for z in zips:
    with ZipFile(z,'r') as zipObj:
        file_name_list = zipObj.namelist()
        for file_name in file_name_list:
            if file_name == 'YOUR FILE NAME':
                 zipObj.extract(file_name, 'DIRECTORY TO MOVE TO')
                 name = 'FILENAME CHANGE'
                 os.rename('OLD_DIRECTORY','NEW_DIRECTORY'+name)
                 

# Pulling files from Amazon S3 bucket.
                 
session = boto3.session.Session(
                 region_name='REGION'
                 aws_access_key_id='ACCESS_KEY'
                 aws_secret_access_key='SECRET_ACCESS'
                 aws_session_token='SESSION'
)
                 
s3 = session.resource('s3)
my_bucket = s3.Bucket('BUCKET_NAME')

for i in url_list:
    objects = my_bucket.objects.filter(Prefix=i)
    for obj in objects:
        path, filename = os.path.split(obj.key)
        my_bucket.download_file(obj.key,filename)