import unreal
import math
import time
import json
import os
import random
import datetime
# ================= Utilities =================
def rotator_to_vec(rotator:unreal.Rotator):
    return unreal.Vector(rotator.roll, rotator.pitch, rotator.yaw)

def vec_to_rotator(rot_vec:unreal.Vector):
    return unreal.Rotator(rot_vec.x, rot_vec.y, rot_vec.z)


def matrix_multiply(A, B):
    """
        A Mult B
    """
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions are not compatible for multiplication.")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_vector_multiply(matrix, vector):
    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix not equal vector")
    
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]

def euler_to_rotation_matrix(rotaion):
    
    # 转换为弧度
    pitch_rad = - math.radians(rotaion.y)
    yaw_rad = math.radians(rotaion.z)
    roll_rad = math.radians(rotaion.x)
    
    cos_z, sin_z = math.cos(yaw_rad), math.sin(yaw_rad)
    R_z = [[cos_z, -sin_z, 0],
           [sin_z,  cos_z, 0],
           [0,      0,     1]]
    
    cos_y, sin_y = math.cos(pitch_rad), math.sin(pitch_rad)
    R_y = [[cos_y,  0, sin_y],
           [0,      1, 0],
           [-sin_y, 0, cos_y]]
    
    cos_x, sin_x = math.cos(roll_rad), math.sin(roll_rad)
    R_x = [[1, 0,      0],
           [0, cos_x, -sin_x],
           [0, sin_x,  cos_x]]
    
    
    R_temp = matrix_multiply(R_z, R_y) 
    R_ue = matrix_multiply(R_temp, R_x) 
    
    return R_ue
