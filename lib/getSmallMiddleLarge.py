import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif'] = 'SimHei'


def getGtAreaAndRatio(label_dir):
    """
    get gt of each scale
    """
    data_dict = {}
    assert Path(label_dir).is_dir(), "label_dir is not exist"

    txts = os.listdir(label_dir)

    for txt in txts:
        print(txt)
        image_dir = os.path.join(label_dir, txt).replace("txt", "jpg")
        image_dir = image_dir.replace("labels", "images")
        img = Image.open(image_dir)
        w, h = img.size

        ratioPic = w * h / (640 * 640)
        with open(os.path.join(label_dir, txt), 'r') as f:
            lines = f.readlines()

        for line in lines:
            temp = line.split()  # str to list{5}
            coor_list = list(map(lambda x: x, temp[1:]))  # [x, y, w, h]
            area = float(coor_list[2]) * float(coor_list[3])
            area = area / ratioPic
            ratio = round(float(coor_list[2]) / float(coor_list[3]), 2)  # w/h

            if temp[0] not in data_dict:
                data_dict[temp[0]] = {}
                data_dict[temp[0]]['area'] = []
                data_dict[temp[0]]['ratio'] = []

            data_dict[temp[0]]['area'].append(area)
            data_dict[temp[0]]['ratio'].append(ratio)

    return data_dict



def getSMLGtNumByClass(data_dict, class_num):
    """
    get nums of small,middle and large per class
    """
    s, m, l = 0, 0, 0

    #image size
    h = 640
    w = 640
    for item in data_dict['{}'.format(class_num)]['area']:
        if item * h * w <= h * w * (80*80)/(640*640):
            s += 1
        elif item * h * w <= h * w * (200*200)/(640*640):
            m += 1
        else:
            l += 1
    return s, m, l


def getAllSMLGtNum(data_dict, isEachClass=False):
    """
    distribution of small,middle and large ones's gt
    """
    S, M, L = 0, 0, 0
    classDict = {'0': {'S': 0, 'M': 0, 'L': 0}, '1': {'S': 0, 'M': 0, 'L': 0}}

    print(classDict['0']['S'])
    # range(class_num)
    if isEachClass == False:
        for i in range(2):
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
        return [S, M, L]
    else:
        for i in range(2):
            S = 0
            M = 0
            L = 0
            s, m, l = getSMLGtNumByClass(data_dict, i)
            S += s
            M += m
            L += l
            classDict[str(i)]['S'] = S
            classDict[str(i)]['M'] = M
            classDict[str(i)]['L'] = L
        return classDict


# draw
def plotAllSML(SML):
    x = ['S:[0, 32x32]', 'M:[32x32, 96x96]', 'L:[96*96, 640x640]']
    fig = plt.figure(figsize=(10, 8))
    plt.bar(x, SML, width=0.5, align="center", color=['skyblue', 'orange', 'green'])
    for a, b, i in zip(x, SML, range(len(x))):
        plt.text(a, b + 0.01, "%d" % int(SML[i]), ha='center', fontsize=15, color="r")  # plt.text
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('gt_size', fontsize=16)
    plt.ylabel('nums', fontsize=16)
    plt.title('distribution of small, middle and large images(640x640)', fontsize=16)
    plt.show()



if __name__ == '__main__':
    labeldir = r'VOCdevkit/labels' #path
    data_dict = getGtAreaAndRatio(labeldir)
    isEachClass =False
    SML = getAllSMLGtNum(data_dict, isEachClass)
    print(SML)
    if not isEachClass:
        plotAllSML(SML)

