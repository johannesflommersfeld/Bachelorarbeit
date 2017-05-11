"""
    converter.py
    ~~~~~~~~~~

    A small utility program, which converts the output of the neuronal nets 
    into data, that can be plottet directly. In the end there are only the 
    number of epochs and the corresponding validation accuracy in the file.
    """

import os
import sys

if(len(sys.argv) != 2):
    print("Please give a .txt file to convert!")
elif('.txt' not in sys.argv[1]):
    print("Please give a .txt file to convert!")
else:
    i = open(sys.argv[1], 'r')
    o = open("temp.txt", 'w')
    for line in i.readlines():
        if('Epoch' in line):
            line = line.replace('Epoch','')
            line = line.replace(' ','')
            line = line.replace(':',' ')
            line = line.replace('validation','')
            line = line.replace('accuracy','')
            line = line.replace('%','')

            o.write(line)
    i.close()
    o.close()
    os.rename("temp.txt",sys.argv[1].replace('.txt', '.dat'))
