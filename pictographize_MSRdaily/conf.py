import os
from enum import Enum, unique


# root path for skeleton directory
rootpath = 'F:\\GameDownload\\OneDrive_1_2018-4-16\\OneDrive_1_2018-4-16\\'
if not os.path.exists(rootpath):
    rootpath = 'E:\\data set\\MSRAction3D\\MSRAction3DSkeletonReal3D\\'
J = [[4, 3, 8, 0, 2, 4, 5, 6, 8, 9, 10,  0, 0, 12, 13, 14, 17, 18,  19],
     [2, 2, 2, 1, 1, 5, 6, 7, 9, 10, 11,12,16, 13, 14, 15, 16, 17, 18]]
confFile = 'F:\\MSRAction3D\\UTKinect\\joints\\conf\\conf.txt'
# The subset numbers of msaction3d data set
as_map = {1: (2 ,3 ,5, 6, 10, 13, 18, 20),    #AS1
          2: (1, 4, 7, 8, 9, 11, 12, 14),     #AS2
          3: (6, 14, 15, 16, 17, 18, 19, 20)} #AS3

weight = [[0, 1, 1, 1, 2, 3, 4, 4, 2, 3, 4, 4, 2, 3, 5, 5, 2, 3, 5, 5],         #walk: 252 390
          [0, 1, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6],         #sitDown: 572 686
          [0, 1, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6],         #standUp: 704 752
          [0, 1, 1, 1, 4, 4, 8, 8, 4, 4, 8, 8, 2, 3, 4, 5, 2, 3, 4, 5],         #pickUp: 822 954
          [0, 1, 1, 1, 4, 5, 6, 7, 4, 5, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1],         #carry: 1016 1242
          [0, 2, 2, 2, 2, 2, 3, 3, 4, 4, 9, 9, 2, 2, 3, 3, 2, 2, 3, 3],         #throw: 1434 1488
          [0, 1, 1, 1, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1, 1, 1],         #push: 1686 1748
          [0, 1, 1, 1, 1, 1, 1, 1, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1, 1, 1],         #pull: 1640 1686
          [0, 1, 1, 1, 3, 4, 5, 5, 3, 4, 5, 5, 1, 2, 2, 1, 1, 2, 2, 1],         #waveHands: 1834 2064
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1],
          [0, 1, 1, 1, 2, 3, 6, 6, 2, 3, 6, 6, 1, 2, 2, 1, 1, 2, 2, 1]]         #clapHands: 2110 2228




@unique
class Limb(Enum):
    LEFT_ARM = 1
    RIGHT_ARM = 2
    TORSO = 3
    LEFT_LEG = 4
    RIGHT_LEG = 5
    ALL = 6
    P1 = 101
    P2 = 102
    P3 = 103
    P4 = 104
    P5 = 105
    P6 = 106
    P7 = 107
    P8 = 108
    P9 = 109
    P10 = 110
    P11 = 111
    P12 = 112
    P13 = 113
    P14 = 114
    P15 = 115
    P16 = 116
    P17 = 117
    P18 = 118
    P19 = 119
    P20 = 120
    ONE = 100

@unique
class FeatureType(Enum):
    V_MAG = 0 # velocity magnitude
    V_ANG = 1 # velocity angle
    V = 2
    POS = 3   # position
    TRAJ = 4  # trajectory shape
    POS2 = 5
    POS3 = 6


@unique
class Plane(Enum):
    xOy = 0
    xOz = 1
    yOz = 2
mPlane2Str = {Plane.xOy: 'xOy',
              Plane.xOz: 'xOz',
              Plane.yOz: 'yOz'}

limb_map = {Limb.LEFT_ARM: [5, 6, 7, 4],
        Limb.RIGHT_ARM: [9, 10, 11, 8],
        Limb.TORSO: [1, 2, 3, 0],
        Limb.LEFT_LEG: [13, 14, 12, 15],
        Limb.RIGHT_LEG: [17, 18, 19, 16],
        Limb.ALL: range(0, 20),
        Limb.P1:[0],
        Limb.P2: [1],
        Limb.P3: [2],
        Limb.P4: [3],
        Limb.P5: [4],
        Limb.P6: [5],
        Limb.P7: [6],
        Limb.P8: [7],
        Limb.P9: [8],
        Limb.P10: [9],
        Limb.P11: [10],
        Limb.P12: [11],
        Limb.P13: [12],
        Limb.P14: [13],
        Limb.P15: [14],
        Limb.P16: [15],
        Limb.P17: [16],
        Limb.P18: [17],
        Limb.P19: [18],
        Limb.P20: [19],
        Limb.ONE: [20]}

limb_map2 = [[4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3], [12,13,14,15], [16,17,18,19]]