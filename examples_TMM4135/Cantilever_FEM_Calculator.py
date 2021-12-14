# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:38:14 2018

@author: bjohau
"""
import numpy as np
import calfem.core as cfc
import tri_with_TODO as tri
import quads_with_TODO as quad
import calfem.vis as cfv

#------------- Define element type, cantilever dimentions, number of nodes and material properties -----------------------------

# Select element type
numElementNodes = 9  # Valid numbers 3, 33, 4, 44, 6, 9
# 33 and 44 are existing 3 and 4 node element-types

# Cantilever with dimensions H x L x thickness
H         =  2.0
L         = 10.0
thickness =  0.1

# Distributed load in x and y, load pr unit area
#eq = np.array([1.0e100,1.0e3]) #ekstrem load i x for å teste hvordan det ville sett ut
eq = np.array([0.,0.])


#End load, Given as resultant

#endLoadXY = np.array([0.,0.]) #no load at end
endLoadXY = np.array([0.0,3.0e6])
#endLoadXY = np.array([3.0e6,0])
#endLoadXY = np.array([4.2e9,0.0]) # Should give unit disp at Poisson = 0

eqTotal = eq * L * H * thickness #Total load for plotting purpose

# Material properties and thickness
ep = [1,thickness, 2] #ep[2]: some of the cfc functions need a number of gausspoints
E  = 2.1e11
nu = 0.3
Dmat = np.mat([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

# Number of nodes: Should be odd numbers in order to handle
scale = 1 # For increasing amount of nodes at same ratio
numNodesX = 5 * scale 
numNodesY = 2 * scale
meshText = 'Unknown mesh'



elTypeInfo= [-1,'Unknown elementtype']
if numElementNodes == 3:
    elTypeInfo= [2,'Our 3 Node Triangle']
elif numElementNodes == 33:
    elTypeInfo= [2,'Existing 3 Node Triangle']
elif numElementNodes == 4:
    elTypeInfo= [3,'Our 4 node Quad mesh']
elif numElementNodes == 44:
    elTypeInfo= [3,'Existing 4 node Quad mesh'] #No change in the meshing func, only in Ke, fe
elif numElementNodes == 6:
    elTypeInfo= [9,'6 node Triangle mesh']
elif numElementNodes == 9:
    elTypeInfo= [10,'9 node Quad mesh']

# Variable for distinguishing our and existing implementation
existingImpl = False
if numElementNodes == 44:
    existingImpl = True
    numElementNodes = 4
elif numElementNodes == 33:
    existingImpl = True
    numElementNodes = 3
    

# number of patches that will fit a 9 node element
numPatchX = (numNodesX-1) // 2
numPatchX = 1 if numPatchX < 1 else numPatchX
numPatchY = (numNodesY-1) // 2
numPatchY = 1 if numPatchY < 1 else numPatchY

numNodesX = numPatchX*2 + 1
numNodesY = numPatchY*2 + 1

if numElementNodes == 6 or numElementNodes == 9:
    numElementsX = (numNodesX-1) // 2
    numElementsY = (numNodesY-1) // 2
else:
    numElementsX = numNodesX -1
    numElementsY = numNodesY -1

bDrawMesh = True


numNodes    = numNodesX * numNodesY
numElements = numElementsX * numElementsY
if numElementNodes in [3,6]:
    numElements *= 2

L_elx = L / (numNodesX-1)
L_ely = H / (numNodesY-1)

nelnod = 6

coords = np.zeros((numNodes,2))
dofs   = np.zeros((numNodes,2),int)       #Dofs is starting on 1 on first dof
edofs  = np.zeros((numElements,numElementNodes*2),int) #edofs are also starting on 1 based dof


inod = 0 # The coords table starts numbering on 0
idof = 1 # The values of the dofs start on 1
ndofs = numNodes * 2

# Set the node coordinates and node dofs

for i in range(numNodesX):
    for j in range(numNodesY):
        coords[inod,0] = L_elx * i
        coords[inod,1] = L_ely * j
        dofs[inod,0] = idof
        dofs[inod,1] = idof+1
        idof += 2
        inod += 1

# Set the element connectivites and element dofs
elnods = np.zeros((numElements,numElementNodes),int)
eldofs = np.zeros((numElements,numElementNodes*2),int)

iel = 0
for ip in range(numPatchX):
    ii = ip*2
    for jp in range(numPatchY):
        jj = jp*2
        # 0 based node numbers, 9 nodes of a 3x3 patch
        nod9 = np.array([
            (ii  )*numNodesY + (jj  ),
            (ii+1)*numNodesY + (jj  ),
            (ii+2)*numNodesY + (jj  ),
            (ii  )*numNodesY + (jj+1),
            (ii+1)*numNodesY + (jj+1),
            (ii+2)*numNodesY + (jj+1),
            (ii  )*numNodesY + (jj+2),
            (ii+1)*numNodesY + (jj+2),
            (ii+2)*numNodesY + (jj+2)],'i')

        if numElementNodes == 3:
            for i in range(2):
                for j in range(2):
                    elnods[iel,:] = [nod9[3*i+j],nod9[3*i+j+1],nod9[3*(i+1)+j+1]]
                    iel += 1
                    elnods[iel,:] = [nod9[3*(i+1)+j+1],nod9[3*(i+1)+j],nod9[3*i+j]]
                    iel += 1
        elif numElementNodes == 6:
            elnods[iel,:] = [nod9[0],nod9[2],nod9[8],nod9[1],nod9[5],nod9[4]]
            iel += 1
            elnods[iel,:] = [nod9[8],nod9[6],nod9[0],nod9[7],nod9[3],nod9[4]]
            iel += 1
        elif numElementNodes == 4:
            for i in range(2):
                for j in range(2):
                    elnods[iel,:] = [nod9[3*i+j],nod9[3*i+j+1],nod9[3*(i+1)+j+1],nod9[3*(i+1)+j]]
                    iel += 1
        elif numElementNodes == 9:
            elnods[iel,:] = [nod9[0],nod9[2],nod9[8],nod9[6],
                             nod9[1],nod9[5],nod9[7],nod9[3],
                             nod9[4]]
            iel += 1


for iel in range(elnods.shape[0]):
    eldofs[iel, ::2] = elnods[iel,:] * 2 + 1 # The x dofs
    eldofs[iel,1::2] = elnods[iel,:] * 2 + 2 # The y dofs


# Draw the mesh.
if bDrawMesh:
    cfv.drawMesh(
        coords=coords,
        edof=eldofs,
        dofsPerNode=2,
        elType=elTypeInfo[0],
        filled=True,
        title=elTypeInfo[1])
    cfv.showAndWait()

# Extract element coordinates
ex, ey = cfc.coordxtr(eldofs,coords,dofs)

# Set fixed boundary condition on left side, i.e. nodes 0-nNody
bc = np.array(np.zeros(numNodesY*2),'i')
idof = 1
for i in range(numNodesY):
    idx = i*2
    bc[idx]   = idof
    bc[idx+1] = idof+1
    idof += 2

# Assemble stiffness matrix

K = np.zeros((ndofs,ndofs))
R = np.zeros((ndofs,1))

#Set the load at the right hand edge
for i in range(numNodesY):
    R[-(i*2+2),0] = endLoadXY[0] / numNodesY
    R[-(i*2+1),0] = endLoadXY[1] / numNodesY

for iel in range(numElements):
    if numElementNodes == 3 and not existingImpl:
        K_el, f_el = tri.tri3e(ex[iel],ey[iel],Dmat,thickness,eq)
    if numElementNodes == 3 and existingImpl:
        K_el, f_el = cfc.plante(ex[iel],ey[iel], ep=ep[:2], D=Dmat,eq=eq) # Existing 3 node tri. 
        f_el = f_el.T # Have to transpose this to fit the given system of implementations
    elif numElementNodes == 6:
        K_el, f_el = tri.tri6e(ex[iel],ey[iel],Dmat,thickness,eq)
    elif numElementNodes == 4 and not existingImpl: #Our implementation
        K_el, f_el = quad.quad4e(ex[iel],ey[iel],Dmat,thickness,eq)
    elif numElementNodes == 4 and existingImpl: #Existing 4 node quad
        #K_el, f_el = cfc.planqe(ex[iel],ey[iel], ep=ep[0:2], D=Dmat,eq=eq) # Trying different existing element functions
        ep[2] = 2 # Number of points for gauss integration 1-3. Used by plani4e
        K_el, f_el = cfc.plani4e(ex[iel],ey[iel], ep=ep, D=Dmat,eq=eq) # Also Using 2 point gaussian integration
    elif numElementNodes == 9:
        K_el, f_el = quad.quad9e(ex[iel],ey[iel],Dmat,thickness,eq)

    cfc.assem(eldofs[iel],K,K_el,R,f_el) # Assemble System K matrix

r, R0 = cfc.solveq(K,R,bc)

print(f"Cantilever with {numElementNodes} element nodes and {numNodes} nodes")

nodMiddle = numNodesY//2 +1  # Mid nod on right edge
xC = r[-(nodMiddle*2)  ,0] # 2 dofs per node, so this is the middle dof on end
yC = r[-(nodMiddle*2)+1,0] # 2 dofs per node, so this is the middle dof on end
print("Displacement center node right end,  x:{:12.3e}   y:{:12.3e}".format(xC, yC))

# Sum uf reaction forces
R0Sum = np.zeros(2,'f')
for i in range(0,(numNodesY*2),2):
    R0Sum[0] += R0[i  ,0]
    R0Sum[1] += R0[i+1,0]

eqTotal = eq * L * H * thickness #Total load for plotting purpose
print("Total reaction force in x:{:12.3e} y:{:12.3e})".format(R0Sum[0],R0Sum[1]))

# Draw the displacements


if bDrawMesh:
    disp = np.array(np.zeros((numNodes,2)),'f')
    rMax = max(abs(max(r)),abs(min(r)))
    scale = 0.15 * L / rMax

    for i in range( np.size(disp,0)):
        disp[i,0] = r[i*2   ,0] * scale
        disp[i,1] = r[i*2 +1,0] * scale

    mesh = cfv.drawDisplacements(displacements=disp, #Return mesh for saving
        coords=coords,
        edof=eldofs,
        dofsPerNode=2,
        elType=elTypeInfo[0],
        title=elTypeInfo[1])

    cfv.showAndWait()
