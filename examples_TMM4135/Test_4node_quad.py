import numpy as np
import calfem.core as cfc
import quads_with_TODO as quads

# Topology
ex = np.array([0.,1.,1.,0.])
ey = np.array([0.,0.,1.,1.])

th = 0.1 # Thickness

# Material properties
E = 2.1e11
nu = 0.3

Dmat = np.array([
    [1., nu,                0.           ],
    [nu, 1.,                0.           ],
    [0., 0., (1.0-nu)/2.0 * E/(1.0-nu**2)]])


# Distributed load (x,y)
eq = [1.,3.]

# Calculate element stiffness, element forces
K_el, f_el, Bmat = quads.quad4e(ex, ey, Dmat, th, eq, returnBmat=True)


# Rigid body translation and rotation
rigX = np.array([1,0,1,0,1,0,1,0])
rigY = np.array([0,1,0,1,0,1,0,1])
rigR = np.array([ey[0],-ex[0],ey[1],-ex[1],ey[2],-ex[2],ey[3],-ex[3]])


# Calculating forces
fx = K_el @ rigX.T
fy = K_el @ rigY.T
fr = K_el @ rigR.T

print('Force from rigX translation:\n',fx)
print('Force from rigY translation:\n',fy)
print('Force from rigR rotation:\n',fr)

# Calculating strains
e_x = Bmat @ rigX.T
e_y = Bmat @ rigY.T
e_r = Bmat @ rigR.T

print('Strain from rigX translation:\n',e_x)
print('Strain from rigY translation:\n',e_y)
print('Strain from rigR rotation:\n',e_r)



constEx = np.array([ex[0],0,ex[1],0,ex[2],0,ex[3],0])
constEy = np.array([0,ey[0],0,ey[1],0,ey[2],0,ey[3]])
constGamma1 = np.array([0,ex[0],0,ex[1],0,ex[2],0,ex[3]])
constGamma2 = np.array([ey[0],0,ey[1],0,ey[2],0,ey[3],0])

Ex = Bmat @ constEx.T # Const strain in x
Ey = Bmat @ constEy.T # Const strain in y
G1  = Bmat @ constGamma1.T # Const shear angle x
G2  = Bmat @ constGamma2.T # Const shear angle y

print('Bmat\n', Bmat) 
print('Ex:\n',Ex)
print('Ey:\n',Ey)
print('G:\n',G1)
print('G:\n',G2)

def test_rigidStrains(): # Rigid translation/rotation on Bmat
    for x, y, r in zip(e_x, e_y, e_r):
        if max(abs(x), abs(y), abs(r)) > 1.e-05: # Check if almost zero
            return False
    return True

def test_rigidStiffness(): # Rigid translation/rotation on Kmat
    for x, y, r in zip(fx, fy, fr):
        if max(abs(x), abs(y), abs(r)) > 1.e-05: # Check if almost zero
            return False
    return True

def test_constStrains(): # Constant Strain on Bmat
    if not np.array_equal(Ex, [1,0,0]):
        return False

    if not np.array_equal(Ey, [0,1,0]):
        return False

    if not np.array_equal(G1, [0,0,1]):
        return False

    if not np.array_equal(G2, [0,0,1]):
        return False

    return True

if __name__=="__main__":

    if test_rigidStiffness():
        print("Test rigidbody translation/rotation on Kmat: success")
    else:
        print("Test rigidbody translation/rotation on Kmat: fail")

    if test_rigidStrains():
        print("Test rigidbody translation/rotation on Bmat: success")
    else:
        print("Test rigidbody translation/rotation on Bmat: fail")
    
    if test_constStrains():
        print("Test const strain on Bmat: success")
    else:
        print("Test const strain on Bmat: fail")