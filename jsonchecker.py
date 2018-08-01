import json
import os.path as osp
import sys

ids = list()

for line in open(osp.join('/home/wynmew/workspace/Data', 'trainSet')):
    ids.append(('/home/wynmew/workspace/Data', line.strip()))

for index in range(len(ids)-1):
    img_id = ids[index]
    annofile = osp.join(img_id[0], img_id[1]).replace("images", "annotations").replace('.jpg', '.json')
    with open(annofile) as datafile:
        AnnoData = json.load(datafile)
    #print(annofile)
    # print(AnnoData)
    try:
        label=AnnoData["annotations"][0]["name"]
    except:
        print(annofile)