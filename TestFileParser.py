import json
import os.path as osp
import os
import sys
#from boxx import show
import cv2
import time
import numpy as np


ids = list()
jsonfile = []
imgfile = []
pts = []
label = []
prdpts=[]
prdlables=[]
scores=[]

for line in open(osp.join('/home/wynmew/workspace/ssd.pytorch/eval', 'testAllwithoutAug_90000')):
    ids.append((line.strip()))

for index in range(len(ids)):
    lineID = index
    print(lineID)
#    try:
    if ids[lineID] == '':
        print('next img')
        if label !=[]:
            print('show img', imgfile)
            img=cv2.imread(imgfile)
            cv2.rectangle(img,(int(pts[0]),int(pts[1])),(int(pts[2]),int(pts[3])),(0,255,0),3)
            for i in range (len(prdpts)):
                cv2.rectangle(img,
                    (int(prdpts[i][0]), int(prdpts[i][1])), (int(prdpts[i][2]), int(prdpts[i][3])), (0, 0, 255), 3)
            cv2.imshow('pre',img)
            cv2.waitKey(0)
            time.sleep(1)

        jsonfile = []
        imgfile = []
        pts = []
        label = []
        prdpts = []
        prdlables = []
        continue
    elif ids[lineID][0] == 'G':
        jsonfile = ids[lineID][18:]
        imgfile = jsonfile.replace("annotations","images").replace('.json', '.jpg')
        print('get grundtruth for ', imgfile)
    elif ids[lineID][0] == 'l':
        tmp = ids[lineID].split(' ')
        pts = [float(tmp[1]), float(tmp[3]), float(tmp[5]), float(tmp[7])]
        label = int(tmp[9])
        print('get gt label:', pts, '-', label)
    elif ids[lineID][0] == 'P':
        print('get predication')
    elif ids[lineID][0].isdigit():
        tmp = ids[lineID].split(' ')
        prdlables.append([tmp[2]])
        scores.append([tmp[4]])
        prdpts.append([float(tmp[5]), float(tmp[7]), float(tmp[9]), float(tmp[11])])
        print('get pre line: ', tmp[4], "-",[float(tmp[5]), float(tmp[7]), float(tmp[9]), float(tmp[11])], '-', tmp[2] )
    else:
        print('error1 at', lineID, " : ", ids[lineID])
        continue
'''
    except:
        print('error2 at', lineID, " : ", ids[lineID])
        continue
'''
