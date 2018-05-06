import logging

from getfeature import getFeature, calAngle, getArmAngleHist, BIN_WIDTH
from preproc import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# threshold
thrRatio = 7
thrAng = 135
thrMove = 0.1 # 小于这个门限的认为是不动点 TODO 依据统计调整这个值
thrHeightChange = 0.2

SLIDING_WINDOW = 5
SLIDING_WINDOW_FOR_BEND = 3
SLIDING_WINDOW_FOR_CIRCLE = 15
STEP = 2


def mergeResult(results):
    if len(results) == 0:
        return []
    if len(results) == 1:
        return [(results[0], results[0]+SLIDING_WINDOW)]
    l = []
    start, end = results[0], results[0]+SLIDING_WINDOW
    for i in range(1, len(results)):
        if end >= results[i]:
            end = results[i] + SLIDING_WINDOW
        else:
            l.append((start, end))
            start = results[i]
            end = start + SLIDING_WINDOW
    l.append((start, end))
    return l

def findWangfan(traj):
    speed = getFeature(traj, 12, FeatureType.V)
    # angle = calAngle(speed[1:], speed[:-1])
    speed_norm = np.linalg.norm(speed, axis=1)
    move = speed_norm > thrMove
    for i in range(1, len(move)-1):
        if move[i-1] and move[i+1]:
            move[i] = True
    speed[~move] = 0
    # max_ind = np.where(angle == np.max(angle[move]))
    # if max_ind > 2 and max_ind < len(speed) - 4:
    #     # todo 直线拟合/距离法
    #     # 直线拟合
    inner = np.sum(speed[:-1] * speed[1:], axis=1)
    corners = np.where(inner < 0)[0]
    # 连续多个转折
    if len(corners) > 1:
        for i in range(len(corners)-1):
            if corners[i+1] - corners[i] < 5: # todo 5是门限
                return True
    # TODO 转折点在一个水平面上，或者转折前后的速度方向为水平方向
    # if len(corners) > 1 and corners[-1] - corners[0] >
    # for p in corners+1:
    #     if p > 1 and p < len(speed) - 2:
    #         # print(speed[p - 2:p + 3, 0])
    #         # print(speed[p - 2:p + 3, 1])
    #         # print(speed[p - 2:p + 3, 2])
    #         # print(p)
    #         if np.abs(np.max(speed[p-2:p+3, 2])) < thrHeightChange:
    #             return True
    return False

def findHandshake(traj):
    ## 关键点只能是一个
    speed = getFeature(traj, 12, FeatureType.V)
    normSpeed = np.linalg.norm(speed, axis=1) > thrMove
    deltHeight = abs(speed[:, 2])+1e-12 # 避免除零异常
    deltHorizon = np.linalg.norm(speed[:, [0, 1]], axis=1)
    ratio = deltHorizon / deltHeight > thrRatio
    angle = calAngle(speed[1:], speed[:-1]) > thrAng
    # print('angle\n', np.where(angle))
    # print('ratio\n', np.where(ratio))
    # print('move\n', np.where(normSpeed))
    l = []
    for start in range(0, len(angle)-SLIDING_WINDOW+2, STEP):
        end = start+SLIDING_WINDOW-1
        if sum(normSpeed[start:end]) >= 2 and ratio[start:end].any() and angle[start:end].any():
            l.append(start)
    l = mergeResult(l)
    return l


def findTick(traj):
    # keypoint = (5, 14, 16, 18)
    # leg = traj[:, keypoint]
    # # leg = leg - leg[:, np.newaxis, 0, :]
    # velocity = getFeature(leg[:, 1:], range(3), FeatureType.V)
    # move = np.linalg.norm(velocity, axis=2) > thrMove
    # move = move.all(axis=1)
    # # 14号点与16号点的夹角
    # angle1416 = calAngle(velocity[:, 0], velocity[:, 1])/np.pi*180*move
    # angle1418 = calAngle(velocity[:, 0], velocity[:, 2])/np.pi*180*move
    # # distance = np.linalg.norm(leg[:, ])
    # # print()
    # print(angle1416)
    # print(angle1418)
    pos = getFeature(traj, 18, FeatureType.POS)
    deltaHeight = np.max(pos) - np.min(pos)
    # print(deltaHeight)
    return deltaHeight

# 获得物理学中的三维空间的外积
# 输入为frame-by-3大小的矩阵
def getOuterProduct(va, vb):
    assert(va.shape == vb.shape)
    prod = np.zeros(va.shape)
    prod[:, 0] = va[:, 1] * vb[:, 2] - va[:, 2] * vb[:, 1]
    prod[:, 1] = va[:, 2] * vb[:, 1] - va[:, 1] * vb[:, 2]
    prod[:, 2] = va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]
    return prod

def circleFit(points):
    assert(points.shape == (4, 3))
    A = np.empty((3 ,3))
    A[0] = points[0] - points[1]
    A[1] = points[0] - points[2]
    A[2] = points[0] - points[3]
    b = np.empty((3, 1))
    b[0] = np.sum(points[0]**2-points[1]**2) / 2
    b[1] = np.sum(points[0]**2-points[2]**2) / 2
    b[2] = np.sum(points[0]**2-points[3]**2) / 2
    try:
        center = np.linalg.solve(A, b).reshape(1, 3)
    except np.linalg.linalg.LinAlgError:
        return
    r = np.linalg.norm(center - points[0].reshape(3, 1))
    return center, r

def findCircleByOuterProd(traj):
    keytraj = traj[:, 12]
    velocity = keytraj[1:] - keytraj[:-1]
    outer = getOuterProduct(velocity[1:], velocity[:-1])
    outer = outer / (np.linalg.norm(outer, axis=1, keepdims=True)+1e-12)
    for i in range(0, len(outer) - SLIDING_WINDOW_FOR_CIRCLE + 1, 2):
        angle = calAngle(outer[i:i + SLIDING_WINDOW_FOR_CIRCLE - 1], outer[i + 1: i + SLIDING_WINDOW_FOR_CIRCLE])
        if (angle < 90).all():
            return True
    return False

def findCircleByFitting(traj, DebugMode=False):
    keytraj = traj[:, 12]
    speed = keytraj[1:] - keytraj[:-1]
    move = np.linalg.norm(speed, axis=1) > thrMove
    minerr = np.inf
    circleFrame = -1
    radius = []
    for i in range(0, len(keytraj) - SLIDING_WINDOW_FOR_CIRCLE, 2):
        trueNum = np.sum(move[i: i + SLIDING_WINDOW_FOR_CIRCLE])
        if trueNum >= SLIDING_WINDOW_FOR_CIRCLE*4/5:
            center = (i + i + SLIDING_WINDOW_FOR_CIRCLE - 1) // 2
            circle = circleFit(keytraj[center - 2: center + 2])
            radius.append(circle[1])
            if circle is not None:
                err1 = np.sum(np.abs(np.linalg.norm(keytraj[i: center - 2] - circle[0], axis=1) - circle[1]))
                err2 = np.sum(np.abs(np.linalg.norm(keytraj[center + 2: i + SLIDING_WINDOW_FOR_CIRCLE] - circle[0], axis=1) - circle[1]))
                err = (err1 + err2) / circle[1]
                if minerr > err:
                    minerr = err
                    circleFrame = i
    if DebugMode:
        print('start: {}\nerror: {}\n'.format(circleFrame, minerr))
        # return err
    print(radius)
    return radius

findCircle = findCircleByFitting

SLIDING_WINDOW_FOR_STRETCH = 5
def findStretchWave(traj):
    # keytraj = traj[:, [1, 8, 10]]
    # upperarm = traj[:, 8] - traj[:, 1]
    # lowerarm = traj[:, 10] - traj[:, 8]
    # angle = calAngle(upperarm, lowerarm).squeeze()
    # print(angle)
    for i in range(0, len(traj) - SLIDING_WINDOW_FOR_STRETCH + 1, 2):
        hist = getArmAngleHist(traj)
        # 期望
        exp = np.sum(hist * np.arange(0, 180 // BIN_WIDTH))
        if exp < 2:
            return True
    return False


def findBend(traj):
    pos = getFeature(traj, [0, 1, 2, 19], FeatureType.POS)
    headarrow = traj[:, np.newaxis, 19]
    angle = calAngle(headarrow, np.array([[0, 0, 1]]))/np.pi*180
    deltaPos = pos[1:] - pos[:-1]
    moveDown = np.uint8(deltaPos < -thrMove)
    moveDown = np.sum(moveDown, axis=1) >= 3
    for start in range(0, len(moveDown) - SLIDING_WINDOW_FOR_BEND, STEP):
        end = start+SLIDING_WINDOW_FOR_BEND-1
        if moveDown[start:end].all() and (angle[start:end] > 30).all():
            return True

    # bend = np.where(pos[:, 2] < shoulder)
    # print(angle)
    return False


def testFindBend():
    with open('bend基元检测.txt', 'w') as f:
        for a in range(1, 21):
            for s in range(1, 11):
                for e in range(1, 4):
                    traj = get_trajectory(a, s, e)
                    if traj is not None:
                        if findBend(traj):
                            f.write('a{},s{},e{}\n'.format(a, s, e))

def testFindCircle():
    for a in range(1, 21):
        for s in range(1, 11):
            for e in range(1, 4):
                traj = get_trajectory(a, s, e)
                if traj is not None:
                    if findCircle(traj):
                        print('a{}'.format(a), 's{}'.format(s), 'e{}'.format(e))

# testStretchWave()
# testFindCircle()

# a = 1
# s=1
# e=1
# traj = get_trajectory(a, s, e)
# if traj is not None:
#     logger.info(findWangfan(traj))

if __name__ == '__main__':
    a = 16
    # s = 1
    # e = 2
    # traj = get_trajectory(a, s, e)
    # if traj is not None:
    #     print(findStretchWave(traj))
    for s in range(1, 11):
        for e in range(1, 4):
            traj = get_trajectory(a, s, e)
            if traj is not None:
                print(a, s, e)
                print(findStretchWave(traj))
