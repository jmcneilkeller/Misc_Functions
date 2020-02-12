import pandas as pd
from datetime import datetime
import sys
import time
from random import uniform
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process

def military_time(time):
    # Takes in string formatted to clock time and transforms to military time.
    return datetime.strptime(time, '%I:%M:%S%p').strftime('%H:%M:%S')

def typewriter(text, start=None, stop=None):
    # Prints out in a typewriter effect to console.
    # Expects a string
    if start is None:
        start = 0.025
    if stop is None:
        stop = 0.1
    for i in range(len(text)):
        sys.stdout.write(text[i])
        sys.stdout.flush()
        time.sleep(uniform(start, stop))
        
def email_comparer(csv_to_vet,unsubscribe_csv):
    # Compares email lists between two CSVs and suppresses against those emails you don't want to send to.
    v = pd.read_csv(csv_to_vet,index_col=0)
    v['send'] = None
    v_send = []
    s = pd.read_csv(unsubscribe_csv,header=1,index_col=0)
    sp = s['email'].tolist()
    for email in v['email']:
        match = process.extractOne(email, sp,scorer=fuzz.QRatio)
        if match[1] == 100:
            v_send.append('N')
        else:
            v_send.append('Y')
    v['send'] = v_send
    output = v.loc[v['send']=='Y']
    ofn = csv_to_vet.replace('.csv','')
    date = time.strftime('%Y%m%d')
    f_name = date + '_' + ofn + '__CLEAN.csv'
    output.to_csv(f_name)