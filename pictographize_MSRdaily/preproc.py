import matplotlib
import numpy as np
import math
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3D

from conf import *

# import warnings
# warnings.filterwarnings('error')


def getskt_msr_action3d(action_id, subject_id, event_id):
    """
    Get the original data from data set.

    :param action_id: an integer of actions. For example: 1 specify action a01 of the data set
    :param subject_id: an integer of subjects.
    :param event_id: an integer of events. each subject perform 3 times for each action, which are denoted as events.
    :returns: The original data , which are stored in a frameNum-by-jointsNum-by-3  numpy.ndarray.
    """

    filename = rootpath+'outputUT\\a{:0>2}_s{:0>2}_e{:0>2}_skeleton3D.txt'.format(action_id, subject_id, event_id)
    if not os.path.exists(filename):
        return
    data = np.loadtxt(filename)
    #data[:,2] /= 4
    data[:,[1,2]] = data[:,[2,1]]
    #print(data.shape[0]/20)
    return np.reshape(data, [-1, 20, 3])

getskt = getskt_msr_action3d


def normalize(skts, joint_origin_id=0, joint_a=1, joint_b=0, joint_hand_a=7, joint_hand_b=11, joint_leg_a=15, joint_leg_b=19 ):
    length, lengthHand, lengthLeg = 0, 0, 0
    n_frames = len(skts)
    for i in range(n_frames):
        # transform the original skeletons to the body coordinate system with the its origin on the given joint_origin
        joint_origin = skts[i][joint_origin_id]
        skts[i] -= joint_origin
        # the length of skeleton should be normalized by the length between joint_a and joint_b
        length += np.linalg.norm(skts[i][joint_a]-skts[i][joint_b])
        lengthHand += np.linalg.norm(skts[i][joint_hand_a]-skts[i][joint_hand_b])
        lengthLeg +=np.linalg.norm(skts[i][joint_leg_a] - skts[i][joint_leg_b])
    length /= n_frames
    lengthLeg/=n_frames
    lengthHand /=n_frames
    if length != 0:
        skts[:, limb_map[Limb.TORSO], :] /= length*4
    if lengthLeg != 0:
        skts[:, limb_map[Limb.LEFT_LEG], :] /= lengthLeg
        skts[:, limb_map[Limb.RIGHT_LEG], :] /= lengthLeg
    if lengthHand != 0:
        skts[:, limb_map[Limb.LEFT_ARM], :] /= lengthHand
        skts[:, limb_map[Limb.RIGHT_ARM], :] /= lengthHand



def rotate(skts):
    n_frames = len(skts)
    angleSin = np.mean(skts[:, 1, 1] - skts[:, 0, 1]) / np.sqrt(
        np.mean(skts[:, 1, 0] - skts[:, 0, 0]) * np.mean(skts[:, 1, 0] - skts[:, 0, 0]) + np.mean(
            skts[:, 1, 1] - skts[:, 0, 1]) * np.mean(skts[:, 1, 1] - skts[:, 0, 1]))
    angleCos = np.mean(skts[:, 1, 0] - skts[:, 0, 0]) / np.sqrt(
        np.mean(skts[:, 1, 0] - skts[:, 0, 0]) * np.mean(skts[:, 1, 0] - skts[:, 0, 0]) + np.mean(
            skts[:, 1, 1] - skts[:, 0, 1]) * np.mean(skts[:, 1, 1] - skts[:, 0, 1]))
    for i in range(n_frames):
        angleSin = np.mean(skts[i, 1, 1] - skts[i, 0, 1]) / np.sqrt(
            (skts[i, 1, 0] - skts[i, 0, 0]) * (skts[i, 1, 0] - skts[i, 0, 0]) + (
                skts[i, 1, 1] - skts[i, 0, 1]) * (skts[i, 1, 1] - skts[i, 0, 1]))
        angleCos = (skts[i, 1, 0] - skts[i, 0, 0]) / np.sqrt(
            (skts[i, 1, 0] - skts[i, 0, 0]) * (skts[i, 1, 0] - skts[i, 0, 0]) + (
                skts[i, 1, 1] - skts[i, 0, 1]) * (skts[i, 1, 1] - skts[i, 0, 1]))
        skts[i] = np.dot(skts[i], np.array([[angleCos, -angleSin, 0], [angleSin, angleCos, 0], [0, 0, 1]]))

def visualize(data, a, s, e):
    plt.figure('3D')
    ax = plt.gca(projection='3d')
    plt.ion()

    for d in data:
        plt.cla()
        plt.title('a{}_s{}_e{}'.format(a, s, e))
        ax.scatter(d[:,0], d[:,1], d[:,2], c='r')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        for j in range(19):
            c1 = J[0][j]
            c2 = J[1][j]
            ax.add_line(Line3D([d[c1, 0], d[c2, 0]], [d[c1, 1],d[c2, 1]],[d[c1, 2],d[c2, 2]], color='blue'))
        try:
            plt.pause(0.05)
        except Exception:
            pass


def get_trajectory(action_id, subject_id, event_id, limb=Limb.ALL):
    skts = getskt(action_id, subject_id, event_id)
    if skts is None:
        return
    normalize(skts)
    rotate(skts)
    if limb == Limb.ALL:
        return skts
    tr = np.empty((skts.shape[0], len(limb_map[limb]), 3), np.float64)
    for frame in range(len(skts)):
        tr[frame] = skts[frame, limb_map[limb]]
    return tr

def preproc_train_data(as_id):
    """
    preprocessing training samples

    :param as_id: range from 1 to 3, which correspond to AS1~AS3 of MSR_ACtion3D
    """
    data = []
    label = []
    min_frames = 100000
    for action in as_map[as_id]:
        for subject in range(1, 11):
            for event in range(1, 3):
                # left hand feature
                lh_feat = get_trajectory(action, subject, event, Limb.LEFT_HAND)
                if lh_feat is not None:
                    label.append(action)
                    data.append(lh_feat)
                    min_frames = min(min_frames, lh_feat.shape[0])
    for k in range(len(data)):
        data[k] = data[k][:min_frames, ...]
        data[k].shape = data[k].size
    data = np.array(data)
    return data, label, min_frames


def prepare_test_data(as_id, min_frames):
    data = []
    groundtruth = []
    for action in as_map[as_id]:
        for subject in range(1, 11):
            lh_feat = get_trajectory(action, subject, 3, Limb.LEFT_HAND)
            if lh_feat is not None:
                if lh_feat.shape[0] >= min_frames:
                    lh_feat = lh_feat[:min_frames, ...]
                else:
                    tmp = np.empty((min_frames, lh_feat.shape[1], lh_feat.shape[2]))
                    tmp[:lh_feat.shape[0], ...] = lh_feat.copy()
                    for k in range(lh_feat.shape[0], min_frames):
                        tmp[k] = lh_feat[lh_feat.shape[0]-1]
                    lh_feat = tmp
                lh_feat.shape = lh_feat.size
                data.append(lh_feat)
                groundtruth.append(action)
    return np.array(data), groundtruth

if __name__ == '__main__':
    a = 1
    s = 1
    e = 1
    # for s in range(1, 11):
    #     for e in range(1, 4):
    skts = getskt(a, s,e)

    if skts is not None:
        visualize(skts, a, s, e)
        normalize(skts)
        visualize(skts, a, s, e)
        rotate(skts)
        visualize(skts, a, s, e)

    # plt.ioff()
    # plt.show()
