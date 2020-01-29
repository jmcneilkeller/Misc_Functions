from datetime import datetime
import sys
import time
from random import uniform

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