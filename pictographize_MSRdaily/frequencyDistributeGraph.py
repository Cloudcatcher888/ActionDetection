import pandas as pd
import math
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import utils
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import decomposition

from preproc import *
from getfeature import *
#from test import *
from fv_py3 import *

#warnings.filterwarnings('error')
import warnings
warnings.filterwarnings("ignore")


subGroupId = 0
trainModel = 1
biasPara = -10
stepNumber = 30
tralength = 52
A = [[1, 6, 9], [8, 9, 12], [10, 16, 18], [18, 17, 19], [11, 12, 18], [13, 14, 20], [16, 17, 18], [6, 13, 20], [14, 15, 19], [14, 17, 18], [6, 8, 9]]
A_sub = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
crossValid = [1,2,3,4,5,6,7,8,9,10]
cross = 5
pz,py,px,vz,vy,vx=14,8,6,14,8,6
# pz,py,px,vz,vy,vx=13,7,7,13,7,7
weight = np.ones((16,20))
point = range(0, 20)
point = [1,2,4,5,7,8,9,11,12,13,15,16,17,19]
stageWeight = [1,1,1,1,1,1,1,1,1,1]
stage = len(stageWeight)

#point = []

P = np.zeros(20)
TP = np.zeros(20)
n = np.zeros(20)

def calcEnt():
    Ent = 0
    for i in range(1, 11):
        Ent -= n[i]/99*math.log(TP[i]/P[i])
    return  Ent

def equaled(x, y):
    for a in range(11):
        if A[a].count(x) + A[a].count(y) == 2:
            return True
    return False

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def calAngle1(va, vb):
    assert(va.ndim == 3)
    assert(vb.shape == (1, 3))
    # each vector in va times vb repectively
    vavb = np.sum(va*vb, axis=2)
    absVa = np.linalg.norm(va, axis=2)
    absVa[np.where(absVa == 0.0)] = 1
    if (vb == 0).all():
        vb = np.array([0, 0, 1])
    absVb = np.linalg.norm(vb)
    cosTheta = np.clip(vavb/(absVa*absVb), -1.0, 1.0)
    theta = np.arccos(cosTheta)
    return theta

def getFeature2(traj, bodyPart, featType, node=100):
    if bodyPart == Limb.ONE:
        tr = traj[:, node]
        tr = tr.reshape([-1, 1, 3])
    else:
        tr = traj[:, limb_map[bodyPart]]
    # return the height of the key points from the specific limbs
    if featType == FeatureType.POS:
        return tr[:, :, 2]
    if featType == FeatureType.POS2:
        return tr[:, :, 0]
    if featType == FeatureType.POS3:
        return tr[:, :, 1]
    if featType == FeatureType.V_MAG:
        tr = tr[1:] - tr[:-1]
        # the return data's format is frame-by-point
        return np.linalg.norm(tr, axis=2)
    if featType == FeatureType.V_ANG:
        tr = tr[1:] - tr[:-1]
        vz = np.array([0, 0, 1])
        vz.shape = (1, 3)
        return calAngle1(tr, vz)

#mat = []
#maxlen = 120
#lengh = []
#for a in range(1, 11):
#    lengh1 = []
#    for s in range(1, 11):
#        lengh2 = []
#        for e in range(1, 3):
#            xx = get_trajectory(a, s, e)
#
#            if xx is None:
#                xx = np.zeros([maxlen,20,3])
#                lengh2.append(0)
#            else:
#                print(a, s, e, len(xx))
#                lengh2.append(xx.__len__())
#                xx = np.concatenate((xx, np.zeros([int(maxlen - xx.__len__()), 20, 3])))
#            mat.append(xx)
#        lengh1.append(lengh2)
#    lengh.append(lengh1)
#rang = []
#rang2= []
#rang3= []
#Vrang = []
#Vrang2= []
#Vrang3= []
#mat = np.array(mat).reshape([200,120,20,3])
#Vmat1 = np.concatenate((mat, np.zeros([200, 1, 20, 3])), axis=1)
#Vmat2 = np.concatenate((np.zeros([200, 1, 20, 3]), mat), axis=1)
#Vmat = Vmat2 - Vmat1
#Vmat= Vmat[:, 1:(Vmat.shape[1]-2), :, :]
#for i in range(0, 20):
#    rang.append([mat[:, :, i, 2].min(), mat[:, :, i, 2].max()])
#    rang2.append([mat[:, :, i, 0].min(), mat[:, :, i, 0].max()])
#    rang3.append([mat[:, :, i, 1].min(), mat[:, :, i, 1].max()])
#for i in range(0, 20):
#    Vrang.append([Vmat[:, :, i, 2].min(), Vmat[:, :, i, 2].max()])
#    Vrang2.append([Vmat[:, :, i, 0].min(), Vmat[:, :, i, 0].max()])
#    Vrang3.append([Vmat[:, :, i, 1].min(), Vmat[:, :, i, 1].max()])

#mat2 = np.mean(mat[:, :4, :, :], axis=0)

MAX = -10 * np.ones(20)
MIN = 10 * np.ones(20)
MAX2 = -10 * np.ones(20)
MIN2 = 10 * np.ones(20)
MAX3 = -10 * np.ones(20)
MIN3 = 10 * np.ones(20)
VMAX = -10 * np.ones(20)
VMIN = 10 * np.ones(20)
VMAX2 = -10 * np.ones(20)
VMIN2 = 10 * np.ones(20)
VMAX3 = -10 * np.ones(20)
VMIN3 = 10 * np.ones(20)

for a in range(1, 16):
    if a==12:
        a=a
    for s in range(1, 11):
        for e in range(1, 3):
            if a==13 and s==5 and e==2 or a==13 and s==6 and e==1:
                continue
            xx = get_trajectory(a, s, e)
            Vxx1 = np.concatenate((xx, np.zeros([1, 20, 3])), axis=0)
            Vxx2 = np.concatenate((np.zeros([1, 20, 3]), xx), axis=0)
            Vxx = Vxx2 - Vxx1
            Vxx = Vxx[1:Vxx.shape[0]-1]
            for i in range(1, 20):
                MAX[i] = np.max((np.max(xx[:, i, 2]), MAX[i]))
                MIN[i] = np.min((np.min(xx[:, i, 2]), MIN[i]))
                MAX2[i] = np.max((np.max(xx[:, i, 0]), MAX2[i]))
                MIN2[i] = np.min((np.min(xx[:, i, 0]), MIN2[i]))
                MAX3[i] = np.max((np.max(xx[:, i, 1]), MAX3[i]))
                MIN3[i] = np.min((np.min(xx[:, i, 1]), MIN3[i]))
                VMAX[i] = np.max((np.max(Vxx[:, i, 2]), VMAX[i]))
                VMIN[i] = np.min((np.min(Vxx[:, i, 2]), VMIN[i]))
                VMAX2[i] = np.max((np.max(Vxx[:, i, 0]), VMAX2[i]))
                VMIN2[i] = np.min((np.min(Vxx[:, i, 0]), VMIN2[i]))
                VMAX3[i] = np.max((np.max(Vxx[:, i, 1]), VMAX3[i]))
                VMIN3[i] = np.min((np.min(Vxx[:, i, 1]), VMIN3[i]))
rang = np.column_stack((MIN, MAX))
rang2 = np.column_stack((MIN2, MAX2))
rang3 = np.column_stack((MIN3, MAX3))
Vrang = np.column_stack((VMIN, VMAX))
Vrang2 = np.column_stack((VMIN2, VMAX2))
Vrang3 = np.column_stack((VMIN3, VMAX3))
def detectMotion(matChip, bia, i):
    dist = -10#np.linalg.norm(matChip - mat2[:, :, i])
    if dist < bia:
        return False
    else:
        return True

def detectMotionF(tra, k):
    bia = 5
    for i in range(len(tra)):
        bia = np.exp(-i / 10) + biasPara
        if detectMotion(tra[i], bia, k):
            return i

def detectMotionA(tra, k):
    bia = 5
    for i in range(len(tra)):
        bia = np.exp(-i / 10) + biasPara
        j = len(tra) - i - 1
        if detectMotion(tra[j], bia, k):
            return j

def rescale(tra):
    tralen = len(tra)
    step = max(1, int(tralen / stepNumber))
    r_tra = np.repeat(tra, max(1, int(tralength / tralen + 0.5)), axis=0)
    v_tra = r_tra[::step]
    v_tra = v_tra[:(stepNumber)]
    v_tra = v_tra + 5
    v_tra = reshape(v_tra, [stage, -1, 20, 3])
    v_traTemp = np.repeat(v_tra[0], stageWeight[0], axis=0)
    for v in range(1, stage):
        v_traTemp = np.concatenate((v_traTemp, np.repeat(v_tra[v], stageWeight[v], axis=0)), axis=0)
    return v_traTemp

#for pz in range(6,20):
#    for py in range(2,7):
#        for vz in range(6,20):
#            px=py
#            vx=py
#            vy=py

Range = []
Range2 = []
Range3 = []
for i in range(0, 20):
    Range.append(np.linspace(rang[i][0], rang[i][1], pz))
    Range2.append(np.linspace(rang2[i][0], rang2[i][1], py))
    Range3.append(np.linspace(rang3[i][0], rang3[i][1], px))
VRange = []
VRange2 = []
VRange3 = []
for i in range(0, 20):
    VRange.append(np.linspace(Vrang[i][0], Vrang[i][1], vz))
    VRange2.append(np.linspace(Vrang2[i][0], Vrang2[i][1], vy))
    VRange3.append(np.linspace(Vrang3[i][0], Vrang3[i][1], vx))

'''
gmm
'''
averageSum = 0
for cross in range(1, 11):

    Fes = []
    for a in A_sub[subGroupId]:
        for s in range(1, 11):
            for e in range(1, 3):
                # if s == cross and e ==2:
                #     continue
                if a == 13 and s == 5 and e == 2 or a == 13 and s == 6 and e == 1:
                    continue
                tra = get_trajectory(a, s, e, Limb.ALL)
                if tra is not None:
                    v_tra = rescale(tra)
                    pos1 = getFeature2(tra, Limb.ALL, FeatureType.POS)
                    fore = detectMotionF(pos1, 2)
                    aft = detectMotionA(pos1, 2)
                    fore, aft = 2,len(pos1)-2
                    #print(a,s, pos1.shape[0], fore, aft,   aft - fore)
                    Pos1 = pos1
                    pos2 = getFeature2(tra, Limb.ALL, FeatureType.POS2)
                    Pos2 = pos2
                    pos3 = getFeature2(tra, Limb.ALL, FeatureType.POS3)
                    Pos3 = pos3
                    Vpos1 = np.concatenate((tra, np.zeros([1, 20, 3])))
                    Vpos2 = np.concatenate((np.zeros([1, 20, 3]), tra))
                    Vpos = Vpos2 - Vpos1
                    VPos1 = Vpos[1:tra.shape[0], :, 2]
                    VPos2 = Vpos[1:tra.shape[0], :, 0]
                    VPos3 = Vpos[1:tra.shape[0], :, 1]
                    for i in range(1, 20):
                        if point.count(i)==0:
                            continue
                        pos = Pos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range[i])
                        #if i ==15 and a == 2 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = ans[0]
                        pos = Pos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range2[i])
                        # if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        pos = Pos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range3[i])
                        # if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        pos = VPos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange[i])
                        #if i ==1 and s == 1 and e ==1:
                        #   pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        pos = VPos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange2[i])
                        #if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange2[i])
                        fe = np.concatenate((fe, ans[0]))
                        pos = VPos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange3[i])
                        #if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange3[i])
                        fe = np.concatenate((fe, ans[0]))
                        fe = np.concatenate((fe, v_tra[:, i, :].reshape([-1])))
                        Fes.append(fe)
    Fes = np.array(Fes)
    means, covs, weights = generate_gmm(Fes, len(point))



    mm = []
    Ans = []
    for a in A_sub[subGroupId]:
        for s in range(1, 11):
            for e in range(1, 3):
                if s == cross and e ==2:
                    continue
                if a == 13 and s == 5 and e == 2 or a == 13 and s == 6 and e == 1:
                    continue
                tra = get_trajectory(a, s, e, Limb.ALL)
                if tra is not None:
                    mm.append(a)
                    v_tra = rescale(tra)

                    Fes=[]
                    ans1 = []
                    ans2 = []
                    ans3 = []
                    ans4 = []
                    ans5 = []
                    ans6 = []
                    pos1 = getFeature2(tra, Limb.ALL, FeatureType.POS)
                    fore = detectMotionF(pos1, 2)
                    aft = detectMotionA(pos1, 2)
                    #print(a,s, pos1.shape[0], fore, aft,   aft - fore)
                    Pos1 = pos1
                    pos2 = getFeature2(tra, Limb.ALL, FeatureType.POS2)
                    Pos2 = pos2
                    pos3 = getFeature2(tra, Limb.ALL, FeatureType.POS3)
                    Pos3 = pos3
                    Vpos1 = np.concatenate((tra, np.zeros([1, 20, 3])))
                    Vpos2 = np.concatenate((np.zeros([1, 20, 3]), tra))
                    Vpos = Vpos2 - Vpos1
                    VPos1 = Vpos[1:tra.shape[0], :, 2]
                    VPos2 = Vpos[1:tra.shape[0], :, 0]
                    VPos3 = Vpos[1:tra.shape[0], :, 1]
                    for i in range(1, 20):
                        if point.count(i)==0:
                            continue
                        pos = Pos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range[i])
                        #if i ==15 and a == 2 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = ans[0]
                        ans1.append(ans[0]*weight[a-1][i-1])
                        pos = Pos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range2[i])
                        #if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans2.append(ans[0]*weight[a-1][i-1])
                        pos = Pos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range3[i])
                        # if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans3.append(ans[0]*weight[a-1][i-1])
                        pos = VPos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange[i])
                        #if i ==1 and s == 1 and e ==1:
                        #   pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans4.append(ans[0]*weight[a-1][i-1])
                        pos = VPos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange2[i])
                        #if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange2[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans5.append(ans[0]*weight[a-1][i-1])
                        pos = VPos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange3[i])
                        #if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange3[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans6.append(ans[0]*weight[a-1][i-1])

                        fe = np.concatenate((fe, v_tra[:, i, :].reshape([-1])))
                        Fes.append(fe)
                    Fes=np.array(Fes)
                    ans1 = np.array(ans1).flatten()
                    ans2 = np.array(ans2).flatten()
                    ans3 = np.array(ans3).flatten()
                    ans4 = np.array(ans4).flatten()
                    ans5 = np.array(ans5).flatten()
                    ans6 = np.array(ans6).flatten()
                    fv = fisher_vector(Fes, means, covs, weights)
                        #ans2.append(getAllAngleHist(tra[fore:aft]))
                    #ans2 = np.array(ans2).flatten()
                    v_tra = v_tra.flatten()
                    ans1 = np.concatenate((fv, v_tra))
                    ans1 = fv
                    #plt.bar(range(950), ans1.tolist())
                    #plt.show()
                    #plt.savefig('F:\MSRAction3D\photo\\a{}_s{}_e{}.png'.format(a, s, e))
                    Ans.append(ans1)
    Ans = np.array(Ans)


    mm = np.array(mm)
    # Ans = np.concatenate((Ans, Ans[:20],Ans[70:94]), axis=0)
    # mm =np.concatenate((mm,mm[:20],mm[70:94]))
    if trainModel == 1:
        clf = svm.SVC(C=0.1, kernel="poly", degree=3, probability=True)
        #LDA = decomposition.KernelPCA(n_components=2000)
        #Ans = LDA.fit_transform(Ans, mm)
        clf.fit(Ans, mm)
    precise =0
    precise1 = 0
    total =0

    Tres = []
    As =[]

    for a in A_sub[subGroupId]:
        for s in range(1, 11):
            for e in range(1, 3):
                if s != cross or e!=2:
                    continue
                tra = get_trajectory(a, s, e, Limb.ALL)
                if tra is not None:
                    #if a == 16 and s == 6 and e ==3:
                    #    continue
                    v_tra = rescale(tra)
                    Fes = []
                    ans1 = []
                    ans2 = []
                    ans3 = []
                    ans4 = []
                    ans5 = []
                    ans6 = []
                    pos1 = getFeature2(tra, Limb.ALL, FeatureType.POS)
                    fore = detectMotionF(pos1, 2)
                    aft = detectMotionA(pos1, 2)
                    # print(a,s, pos1.shape[0], fore, aft,   aft - fore)
                    Pos1 = pos1
                    pos2 = getFeature2(tra, Limb.ALL, FeatureType.POS2)
                    Pos2 = pos2
                    pos3 = getFeature2(tra, Limb.ALL, FeatureType.POS3)
                    Pos3 = pos3
                    Vpos1 = np.concatenate((tra, np.zeros([1, 20, 3])))
                    Vpos2 = np.concatenate((np.zeros([1, 20, 3]), tra))
                    Vpos = Vpos2 - Vpos1
                    VPos1 = Vpos[1:tra.shape[0], :, 2]
                    VPos2 = Vpos[1:tra.shape[0], :, 0]
                    VPos3 = Vpos[1:tra.shape[0], :, 1]
                    for i in range(1, 20):
                        if point.count(i) == 0:
                            continue
                        pos = Pos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range[i])
                        # if i ==15 and a == 2 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = ans[0]
                        ans1.append(ans[0]*weight[a-1][i-1])
                        pos = Pos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range2[i])
                        # if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans2.append(ans[0]*weight[a-1][i-1])
                        pos = Pos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=Range3[i])
                        # if i ==12 and s == 1 and e ==1:
                        #    pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans3.append(ans[0]*weight[a-1][i-1])
                        pos = VPos1[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange[i])
                        # if i ==1 and s == 1 and e ==1:
                        #   pos.hist(normed=True, bins=Range[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans4.append(ans[0]*weight[a-1][i-1])
                        pos = VPos2[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange2[i])
                        # if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange2[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans5.append(ans[0]*weight[a-1][i-1])
                        pos = VPos3[:, i]
                        pos = pos.flatten()
                        pos = pd.Series(pos)
                        ans = np.histogram(pos, density=True, bins=VRange3[i])
                        # if i ==7 and a == 7 and e ==1:
                        #    pos.hist(normed=True, bins=VRange3[i])
                        fe = np.concatenate((fe, ans[0]))
                        ans6.append(ans[0]*weight[a-1][i-1])

                        fe = np.concatenate((fe, v_tra[:, i, :].reshape([-1])))
                        Fes.append(fe)
                    Fes = np.array(Fes)
                    ans1 = np.array(ans1).flatten()
                    ans2 = np.array(ans2).flatten()
                    ans3 = np.array(ans3).flatten()
                    ans4 = np.array(ans4).flatten()
                    ans5 = np.array(ans5).flatten()
                    ans6 = np.array(ans6).flatten()
                    fv = fisher_vector(Fes, means, covs, weights)
                    # ans2.append(getAllAngleHist(tra[fore:aft]))
                    # ans2 = np.array(ans2).flatten()
                    v_tra = v_tra.flatten()

                    if fv[0] == fv[0]:
                        ans1 = np.concatenate((fv, v_tra))
                        ans1= fv
                        #plt.bar(range(950), ans1.tolist())
                        #plt.show()
                        #plt.savefig('F:\MSRAction3D\photo\\test_a{}_s{}_e{}.png'.format(a, s, e))
                        #tre = clf.predict(LDA.transform(ans1.reshape([1, -1])))
                        tre = clf.decision_function(ans1.reshape([1, -1]))
                        tre = clf.predict(ans1.reshape([1, -1]))
                    else:
                        tre = 4
                    precise += equaled(tre, a) | (tre == a)
                    precise1 += int(tre == a)

                    print (a, tre)
                    As.append(a)
                    Tres.append(tre)
                    total += 1
                    P[tre] += 1
                    TP[tre] += int(tre == a)
                    n[tre] += 1



    print(precise1 / total)
    averageSum+=precise1/total
    # cm =confusion_matrix(As, Tres)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.matshow(cm)
    # plt.colorbar()
    #
    # plt.ylabel('True label')  # 坐标轴标签
    # plt.xlabel('Predicted label')  # 坐标轴标签
    # plt.show()
print(pz,py,px,vz,vy,vx,averageSum)
Ent = calcEnt()
print (Ent)


