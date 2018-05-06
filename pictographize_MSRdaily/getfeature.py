from preproc import *
# warnings.filterwarnings('error')

BIN_WIDTH = 30 # 角度直方图作为分类依据，直方图中1bin表示30度

def calAngle(va, vb):
    if (va.ndim == 3 or va.ndim == 2) and vb.shape == (1, 3):
        # each vector in va times vb repectively
        vavb = np.sum(va*vb, axis=-1)
        absVa = np.linalg.norm(va, axis=-1)
        absVa[np.where(absVa == 0.0)] = 1
        if (vb == 0).all():
            vb = np.array([0, 0, 1])
        absVb = np.linalg.norm(vb)
        cosTheta = np.clip(vavb/(absVa*absVb), -1.0, 1.0)
        theta = np.arccos(cosTheta)
        return theta/np.pi*180
    elif va.shape == vb.shape and va.ndim == vb.ndim and va.shape[-1] == 3:
        oShape = va.shape[:-1]
        if va.ndim == 2:
            v1 = va[:, np.newaxis, :]
            v2 = vb[:, np.newaxis, :]
        elif va.ndim == 3:
            v1 = va
            v2 = vb
        else:
            raise Exception('Unkown calculation for calAngle function!\n')
        v1v2 = np.sum(v1*v2, axis=2)
        v1Norm = np.linalg.norm(v1, axis=2)
        v2Norm = np.linalg.norm(v2, axis=2)
        cos = np.clip(v1v2/(v1Norm*v2Norm+1e-12), -1.0, 1.0)
        theta = np.arccos(cos)
        return theta/np.pi*180
    else:
        raise Exception('Unkown calculation for calAngle function!\n')

def getAngleHist(traj, stick, start=0, end=0):
    if end == 0:
        end = len(traj)
    limb = traj[start:end, stick[0]-1] - traj[start:end, stick[1]-1]
    anglex = calAngle(limb, np.array([[1, 0, 0]])).squeeze()
    anglez = calAngle(limb, np.array([[0, 0, 1]])).squeeze()
    angle = np.hstack((anglex, anglez))
    ## 直方图 ##
    angleBin = (angle // BIN_WIDTH).astype(np.int)
    angleBin = np.clip(angleBin, 0, 180 // BIN_WIDTH - 1)  # 防止出现刚好夹角为180度的情况, 不加这一步当夹角为180度时下一步会出现IndexError
    l = 180 // BIN_WIDTH
    hist = np.zeros(l * 2)
    for i in range(l):
        hist[i] = np.sum(angleBin[:l] == i)
    for i in range(l):
        hist[i + l] = np.sum(angleBin[l:] == i)
    # 归一化
    hist[: l] /= len(anglex)

    # if arm == 0:
    #     upperarm = traj[start:end, 8] - traj[start:end, 1]
    #     lowerarm = traj[start:end, 10] - traj[start:end, 8]
    # # 右手
    # else:
    #     upperarm = traj[start:end, 7] - traj[start:end, 0]
    #     lowerarm = traj[start:end, 9] - traj[start:end, 7]
    # angle = calAngle(upperarm, np.array([[1, 0, 0]])).squeeze()
    # ## 直方图 ##
    # angleBin = (angle // BIN_WIDTH).astype(np.int)
    # angleBin = np.clip(angleBin, 0, 180 // BIN_WIDTH - 1)  # 防止出现刚好夹角为180度的情况, 不加这一步当夹角为180度时下一步会出现IndexError
    # l = 180 // BIN_WIDTH
    # hist = np.zeros(l * 2)
    # for i in range(l):
    #     hist[i] = np.sum(angleBin == i)
    # # 归一化
    # hist[: l] /= len(angleBin)
    #
    # angle = calAngle(upperarm, np.array([[0, 0, 1]])).squeeze()
    # ## 直方图 ##
    # angleBin = (angle // BIN_WIDTH).astype(np.int)
    # angleBin = np.clip(angleBin, 0, 180 // BIN_WIDTH - 1)  # 防止出现刚好夹角为180度的情况, 不加这一步当夹角为180度时下一步会出现IndexError
    # for i in range(l):
    #     hist[i + l] = np.sum(angleBin == i)
    # # 归一化
    # hist[l: ] /= len(angleBin)
    return hist

def getArmAngleHist(traj, start=0, end=0):
    histLeftUpperArm = getAngleHist(traj, (8, 1), start=start, end=end)
    histLeftLowerArm = getAngleHist(traj, (10, 8), start=start, end=end)
    histRightUpperArm = getAngleHist(traj, (7, 0), start=start, end=end)
    histRightLowerArm = getAngleHist(traj, (9, 7), start=start, end=end)
    return np.hstack((histLeftUpperArm, histLeftLowerArm, histRightUpperArm, histRightLowerArm))

def getAllAngleHist(traj):
    HIST = []
    Edges = [(20, 3), (1, 3), (1, 8), (10, 12), (8, 10), (3, 4), (4, 7), (7, 5), (5, 14), (14, 16), (16, 18), (7, 6),
             (6, 15), (17, 19), (3, 2), (2, 9), (9, 11), (11, 13), (15, 17)]
    for edge in Edges:
        HIST.append(getAngleHist(traj, edge))
    return np.hstack(HIST)

def getFeature(traj, bodyPart, featType):
    if isinstance(bodyPart, Limb):
        tr = traj[:, limb_map[bodyPart]]
    else:
        tr = traj[:, bodyPart]
    # return the height of the key points from the specific limbs
    if featType == FeatureType.POS:
        return tr[:, :, 2]
    if featType == FeatureType.POS:
        return tr[:, :, 0]
    if featType == FeatureType.POS:
        return tr[:, :, 1]
    if featType == FeatureType.V_MAG:
        tr = tr[1:] - tr[:-1]
        # the return data's format is frame-by-point
        return np.linalg.norm(tr, axis=2)
    if featType == FeatureType.V:
        return tr[1:] - tr[:-1]
    if featType == FeatureType.V_ANG:
        tr = tr[1:] - tr[:-1]
        vz = np.array([0, 0, 1])
        vz.shape = (1, 3)
        return calAngle(tr, vz)
    if featType == FeatureType.TRAJ:
        return tr
    raise Exception('Unkown FeatureType:{}\n'.format(FeatureType))

for s in range(0, 10):
    for a in range(0, 3):
        for e in range(0, 3):
            tra = get_trajectory(s, a, e, Limb.ALL)
            if tra is not None:
                feat = getFeature(tra, Limb.ALL, FeatureType.POS)

