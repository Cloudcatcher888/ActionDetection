from collections import Iterable

from preproc import *

traject = []
f = 0
ax = 0

def visualizeTraj(action, plane, point, draw=False, record=False):
    # record 只能在point为单个点的情况下使用
    assert(isinstance(action, Iterable) and len(action) == 3)
    assert(isinstance(plane, Plane))
    traj = get_trajectory(action[0], action[1], action[2], Limb.ALL)
    if traj is None:
        return
    if isinstance(point, Iterable):
        traj = traj[:, point]
    else:
        traj = traj[:, np.newaxis, point]
        point = [point]
    if plane == Plane.xOy:
        traj = traj[..., [0, 1]]
    elif plane == Plane.xOz:
        traj = traj[..., [0, 2]]
    elif plane == Plane.yOz:
        traj = traj[..., [1, 2]]
    else:
        raise(Exception('Unknown plane'))
    maxAxisH, maxAxisV = np.max(traj[..., 0]), np.max(traj[..., 1])
    minAxisH, minAxisV = np.min(traj[..., 0]), np.min(traj[..., 1])
    maxAxis = maxAxisH + (maxAxisH - minAxisH) * 0.1 , maxAxisV + (maxAxisV - minAxisV) * 0.1
    minAxis = minAxisH - (maxAxisH - minAxisH) * 0.1 , minAxisV - (maxAxisV - minAxisV) * 0.1

    if not (record and len(point) == 1):
        plt.ion()
    plt.figure('arrow')
    plt.cla()
    try:
        plt.xlim(minAxis[0], maxAxis[0])
        plt.ylim(minAxis[1], maxAxis[1])
    except Exception as e:
        if minAxis == (0, 0):
            pass
        else:
            raise(e)

    for p in range(len(point)):
        for frame in range(traj.shape[0]-1):
            try:
                plt.arrow(traj[frame, p, 0],
                          traj[frame, p, 1],
                          traj[frame+1, p, 0] - traj[frame, p, 0],
                          traj[frame+1, p, 1] - traj[frame, p, 1],
                          shape='full', lw=1, length_includes_head=True, head_width=.1)
            except Exception:
                if traj[frame, p, 0] == 0 or traj[frame, p, 1]:
                    pass
                else:
                    print(traj[frame, p, 0])
                    exit(-1)


    if record and len(point) == 1:
        if not os.path.isdir('result'):
            os.mkdir('result')
        dirname = 'result/{:0>2}/{}/'.format(point[0]+1, mPlane2Str[plane])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        plt.savefig(dirname+'a{:0>2}_s{:0>2}_e{:0>2}'.format(action[0], action[1], action[2]))

    if draw:
        plt.figure('drawing')
        ax = plt.gca()
        plt.xlim(minAxis[0], maxAxis[0])
        plt.ylim(minAxis[1], maxAxis[1])
        x, y = traj[..., 0], traj[..., 1]
        for i in range(len(x)):
            ax.scatter(x[i], y[i] , c = 'b')
            plt.pause(0.1)
        plt.cla()
    # if not (record and len(point) == 1):
    #     plt.ioff()
    #     plt.show()

def onPress(event):
    global traject, f, ax
    f += 1
    if f == len(traject):
        f = 0
        plt.close()
    else:
        ax.cla()
        ax.scatter(traject[f, :, 0], traject[f, :, 1], traject[f, :, 2], c='r')
        plt.title(f)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        for j in range(19):
            c1 = J[0][j]
            c2 = J[1][j]
            ax.add_line(Line3D([traject[f, c1, 0], traject[f, c2, 0]], [traject[f, c1, 1], traject[f, c2, 1]],
                               [traject[f, c1, 2], traject[f, c2, 2]], color='blue'))

def showTheGesture(action):
    global traject, f, ax
    traject = getskt(action[0], action[1], action[2])
    if traject is None:
        return
    normalize(traject)
    fig = plt.figure('Gesture')
    ax = plt.gca(projection='3d')
    plt.cla()
    plt.title('a{}_s{}_e{}'.format(action[0], action[1], action[2]))
    ax.scatter(traject[f, :, 0], traject[f, :, 1], traject[f, :, 2], c='r')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    for j in range(19):
        c1 = J[0][j]
        c2 = J[1][j]
        ax.add_line(Line3D([traject[f, c1, 0], traject[f, c2, 0]], [traject[f, c1, 1], traject[f, c2, 1]], [traject[f, c1, 2], traject[f, c2, 2]], color='blue'))
    fig.canvas.mpl_connect('key_press_event', onPress)
    plt.show()

if __name__ == '__main__':
    # point = 12
    # a = 4
    # for s in range(1, 11):
    #     for e in range(1, 4):
    #         visualizeTraj((a, s, e), Plane.xOz, point, draw=True)
            # visualizeTraj((a, s, e), Plane.xOy, point, draw=True)
            # visualizeTraj((a, s, e), Plane.yOz, point, draw=True)
    showTheGesture((19,8,1))
    # plt.show()

