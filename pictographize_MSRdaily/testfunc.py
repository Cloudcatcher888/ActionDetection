from getfeature import *
from trajectory_base import circleFit

def testcalAngle():
    a = np.array([[-1,-1, -1],[0,1,0]])
    b = np.array([1,1,1])
    print(calAngle(a,b))

def testCircleFit():
    p = np.array([[0, 0 ,1], [0, 1, 0], [1, 0, 0], [-1, 0, 0]])
    print(circleFit(p))

testCircleFit()
