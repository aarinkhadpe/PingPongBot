import numpy as np

def coorDistance(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.sqrt(np.sum((c2 - c1) ** 2))



#motor mounting coordinates
motorNet_BM = [36.0, 72.0, 0.0]
motorNet_TL = [0.0, 72.0, 36.0]
motorNet_TR = [72.0, 72.0, 36.0]
motorBack_TM = [36.0, 0.0, 36.0]
motorBack_BL = [0.0, 0.0, 0.0]
motorBack_BR = [72.0, 0.0, 0.0]


#input values
beta = np.radians(30.0) #x axis roation amount
gamma = np.radians(30.0) #z axis roation amount

x_pos = 67.0 #x coordinate of paddle center
y_pos = 67.0 #y coordinate of paddle center
z_pos = 30.0 #z coordinate of paddle center

#paddle attachment relative coordinates
h = 3.0
w = 4.0

local_PTL = np.array([-w, 0, h])
local_PTM = np.array([0, 0, h])
local_PTR = np.array([w, 0, h])
local_PBL = np.array([-w, 0, -h])
local_PBM = np.array([0, 0, -h])
local_PBR = np.array([w, 0, -h])

matrix_Rx = np.array(
    [
    [1,            0,             0],
    [0, np.cos(beta), -np.sin(beta)],
    [0, np.sin(beta),  np.cos(beta)]
    ])

matrix_Rz = np.array(
    [
    [np.cos(gamma), -np.sin(gamma), 0],
    [np.sin(gamma),  np.cos(gamma), 0],
    [0,          0,                 1]
    ])

matrix_rotationXZ = matrix_Rz @ matrix_Rx

global_PTL = matrix_rotationXZ @ local_PTL

global_PTL[0] = global_PTL[0] + x_pos
global_PTL[1] = global_PTL[1] + y_pos
global_PTL[2] = global_PTL[2] + z_pos


print(coorDistance(global_PTL, motorBack_BL))


