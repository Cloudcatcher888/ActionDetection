import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3D

from conf import *

# import warnings
# warnings.filterwarnings('error')
Class = ['walk:', 'sitDown:', 'standUp:', 'pickUp:', 'carry:', 'throw:', 'push:', 'pull:', 'waveHands:', 'clapHands:']
fc = open(confFile)

def lower_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = high+1
    while low<high:
        mid= (low+high)/2
        midd = nums[int(mid)]
        if midd<target:
            low = int(mid)+1
        else:
            high =int(mid)
            pos =high
    return pos
for a in range(1,17):
    for s in range(1, 11):
        for e in range(1, 3):
            filename = rootpath+'a{:0>2}_s{:0>2}_e{:0>2}_skeleton.txt'.format(a, s, e)
            f = open(filename)
            length= f.readline().strip(' ').split(' ')
            length=int(length[0])
            saveFilePath = rootpath+'outputUT\\'
            saveFile = 'a{:0>2}_s{:0>2}_e{:0>2}_skeleton3D.txt'.format(a, s, e)
            f2 = open(saveFilePath+saveFile, 'w+')
            for l in range(0,length):
                n = f.readline()
                if int(n)==40:
                    for i in range(0,20):
                        data = f.readline()
                        f2.write(data[0:-2])
                        f.readline()
                        f2.write('\n')
                else:
                    for ll in range(0,int(n)):
                        f.readline()

            f2.close()

