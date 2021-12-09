"""
Created on Sun Oct 21 16:38:14 2018
Edited on Thu Dec 09 09:02:36 2021 

@author: bjohau
@editor: eliashk
"""

import numpy as np

def plante(ex,ey,ep,D,eq=None):
    
    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')
        
    if ep[0] == 1 :
        return tri3e(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return tri3e(ex,ey,Dinv,ep[1],eq)

def tri3e(ex,ey,D,th,eq=None):
   
    A2_mat = np.array([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(A2_mat)  # Double of triangle area
    A  = A2 / 2
       
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    ##################################TODO####################################
    dzeta_dx, dzeta_dy = zeta_partials_x_and_y(ex, ey)

    #Tøynings-forskyvningsmatrise B
    B = np.array([
                    [dzeta_dx[0],     0    , dzeta_dx[1],     0    , dzeta_dx[2],      0   ],
                    [    0    , dzeta_dy[0],     0    , dzeta_dy[1],     0    , dzeta_dy[2]],
                    [dzeta_dy[0], dzeta_dx[0], dzeta_dy[1], dzeta_dx[1], dzeta_dy[2], dzeta_dx[2]]
                ])

    #elementstivhetsmatrisa
    Ke = A*th*(np.transpose(B)*D*B)

    if eq is None:  #fordelt last

        return Ke
    else:
        fx = eq[0]/3 *A*th
        fy = eq[1]/3 *A*th
        fe = np.array([[fx], [fy], [fx], [fy], [fx], [fy]])  #lastvektor

        return Ke, fe
    
    ##########################################################################
    
def zeta_partials_x_and_y(ex,ey):

    tmp = np.array([[1,ex[0],ey[0]],
                    [1,ex[1],ey[1]],
                    [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(tmp)  # Double of triangle area
       
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k
    
    dzeta_dx = np.zeros(3)           # Partial derivative with respect to x
    dzeta_dy = np.zeros(3)           # Partial derivative with respect to y

    ##################################TODO####################################
    for i in range(0, 3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        dzeta_dx[i] = (ey[j]-ey[k])/A2
        dzeta_dy[i] = (ex[k]-ex[j])/A2

    ##########################################################################

    return dzeta_dx, dzeta_dy

# Functions for 6 node triangle
    
def tri6_area(ex,ey):
        
    A_mat = np.array([[1,ex[0],ey[0]],
                    [1,ex[1],ey[1]],
                    [1,ex[2],ey[2]]])
    
    A = np.linalg.det(A_mat)/2
    
    return A


def tri6_shape_functions(zeta):
    
    N6 = np.zeros(6)

    ##################################TODO####################################
    for i in range(0, 3):
        N6[i]=zeta[i]*(2*zeta[i]-1)
        N6[i+3]=4*zeta[i]*zeta[i-2]

    ##########################################################################

    return N6


def tri6_shape_function_partials_x_and_y(zeta,ex,ey):
    
    dzeta_dx, dzeta_dy = zeta_partials_x_and_y(ex,ey)
    
    N6_dx = np.zeros(6)
    N6_dy = np.zeros(6)
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    ##################################TODO####################################
    for i in range(0, 3):
        N6_dx[i]=(4*zeta[i]-1)*dzeta_dx[i] #renger ut dette for hånd
        N6_dy[i]=(4*zeta[i]-1)*dzeta_dy[i]

        N6_dx[i+3]=4*zeta[i]*dzeta_dx[i-2] + 4*zeta[i-2]*dzeta_dx[i]
        N6_dy[i+3]=4*zeta[i]*dzeta_dy[i-2] + 4*zeta[i-2]*dzeta_dy[i]
    ##########################################################################

    return N6_dx, N6_dy


def tri6_Bmatrix(zeta,ex,ey):
    
    nx,ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.zeros((3,12))

    # TODO: fill out missing parts (or reformulate completely)

    return Bmatrix


def tri6_Kmatrix(ex,ey,D,th,eq=None):
    
    zetaInt = np.array([[0.5,0.5,0.0],
                        [0.0,0.5,0.5],
                        [0.5,0.0,0.5]])
    
    wInt = np.array([1.0/3.0,1.0/3.0,1.0/3.0])

    A    = tri6_area(ex,ey)
    
    # Ke = np.zeros((12,12))
    Ke = np.eye(12)

    # TODO: fill out missing parts (or reformulate completely)

    if eq is None:
        return Ke
    else:
        fe = np.zeros((12,1))

        # TODO: fill out missing parts (or reformulate completely)

        return Ke, fe

def tri6e(ex,ey,D,th,eq=None):
    return tri6_Kmatrix(ex,ey,D,th,eq)