# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi, eta):
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    # TODO: fill inn values of the  shape functions
    N = np.zeros(4)

    N[0] = 0.25 * (1 + xsi) * (1 + eta)
    N[1] = 0.25 * (1 - xsi) * (1 + eta)
    N[2] = 0.25 * (1 - xsi) * (1 - eta)
    N[3] = 0.25 * (1 + xsi) * (1 - eta)

    return N

def quad4_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi

    Ndxi = np.zeros(4)

    Ndxi[0] = 0.25 * (1 + eta)
    Ndxi[1] = -0.25 * (1 + eta)
    Ndxi[2] = -0.25 * (1 - eta)
    Ndxi[3] = 0.25 * (1 - eta)

    return Ndxi

def quad4_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    """
    # ----- Derivatives of shape functions with respect to eta -----
    # TODO: fill inn values of the  shape functions gradients with respect to xsi
    Ndeta = np.zeros(4)

    Ndeta[0] = 0.25 * (1 + xsi)
    Ndeta[1] = 0.25 * (1 - xsi)
    Ndeta[2] = -0.25 * (1 - xsi)
    Ndeta[3] = -0.25 * (1 + xsi)

    return Ndeta

def quad4e(ex, ey, D, thickness, eq=None, returnBmat=False):
    """
    Calculates the stiffness matrix for a 8 node isoparametric element in plane stress

    Parameters:

        ex  = [x1 ... x4]           Element coordinates. Row matrix
        ey  = [y1 ... y4]
        D   =           Constitutive matrix
        thickness:      Element thickness
        eq = [bx; by]       bx:     body force in x direction
                            by:     body force in y direction
        returnBmat[bool]: Default=False. Returns the Bmatrix as 3rd output if True

    Returns:

        Ke : element stiffness matrix (8 x 8)
        fe : equivalent nodal forces (4 x 1)
        B  : (not returned by default) Deformation matrix

    """
    t = thickness

    if eq is None:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            #Calculate Jacobian, inverse Jacobian and determinant of the Jacobian

            J = G @ H # Jacobian
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            B  = np.zeros((3,8))

            for i in range(8):

                if(i%2 == 0):
                    B[0][i] = dNdx[i // 2]
                    B[2][i] = dNdy[i // 2]

            for i in range(8):

                if(i%2 == 1):
                    B[1][i] = dNdy[i // 2]
                    B[2][i] = dNdx[i // 2]


            #TODO: Fill out correct values for displacement interpolation xsi and eta
            N2 = np.zeros((2,8))
            for i in range(8):

                if (i%2 == 0):
                    N2[0][i] = N1[i // 2]

                else:
                    N2[1][i] = N1[i // 2]

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    if returnBmat: # if specified, return B matrix aswell
        return Ke, fe, B
    return Ke, fe  # Returns stiffness matrix and nodal force vector


#################################### 9 node ###########################################
def quad9_shapefuncs(xsi, eta):

    N = np.zeros(9)

    N[0] = 0.25 * (1 + xsi) * (1 + eta) * xsi * eta
    N[1] = -0.25 * (1 - xsi)*(1 + eta) * xsi * eta
    N[2] = 0.25 * (1 - xsi) * (1 - eta) * xsi * eta
    N[3] = -0.25 * (1 + xsi) * (1 - eta) * xsi * eta
    N[4] = 0.5 * (1 + xsi) * (1 - xsi) * (1 + eta) * eta
    N[5] = -0.5 * (1 + eta) * (1 - eta) * (1 - xsi) * xsi
    N[6] = -0.5 * (1 + xsi) * (1 - xsi) * (1 - eta) * eta
    N[7] = 0.5 * (1 + eta) * (1 - eta) * (1 + xsi) * xsi
    N[8] = (1 + eta) * (1 - eta) * (1 + xsi) * (1 - xsi)

    return N

def quad9_shapefuncs_grad_xsi(xsi, eta):
    
    Ndxsi = np.zeros(9)

    Ndxsi[0] = 0.25 * (1 + 2 * xsi) * (1 + eta) * eta
    Ndxsi[1] = -0.25 * (1 - 2 * xsi) * (1 + eta) * eta
    Ndxsi[2] = 0.25 * (1 - 2 * xsi) * (1 - eta) * eta
    Ndxsi[3] = - 0.25 * (1 + 2 * xsi) * (1 - eta) * eta
    Ndxsi[4] = - xsi * (1 + eta) * eta
    Ndxsi[5] = - 0.5 * (1 - 2 * xsi) * (1 + eta) * (1 - eta)
    Ndxsi[6] = xsi * (1 - eta) * eta
    Ndxsi[7] = 0.5 * (1 - eta) * (1 + eta) * (1 + 2 * xsi)
    Ndxsi[8] = (-2 * xsi) * (1 - eta**2)

    return Ndxsi

def quad9_shapefuncs_grad_eta(xsi, eta):
    
    Ndeta = np.zeros(9)

    Ndeta[0] = 0.25 * (1 + xsi) * (1 + 2 * eta) * xsi
    Ndeta[1] = -0.25 * (1 - xsi) * (1 + 2 * eta) * xsi
    Ndeta[2] = 0.25 * (1 - xsi) * (1 - 2 * eta) * xsi
    Ndeta[3] = -0.25 * (1 + xsi) * (1 - 2 * eta) * xsi
    Ndeta[4] = 0.5 * (1 + xsi) * (1 - xsi) * (1 + 2 * eta)
    Ndeta[5] = eta * (1 - xsi) * xsi
    Ndeta[6] = -0.5 * (1 + xsi) * (1 - xsi) * (1 - 2 * eta)
    Ndeta[7] = -eta * (1 + xsi) * xsi
    Ndeta[8] = (-2 * eta) * (1 - xsi**2)

    return Ndeta

def quad9e(ex,ey,D,th,eq=None, returnBmat=False):
    """
    Compute the stiffness matrix for a nine node membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :param bool returnBmat: Returns the Bmatrix as 3rd output if True
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    :(not by default) return mat B:  Deformation Matrix
    """

    if eq is None:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix


    Ke = np.zeros((18,18))        # Create zero matrix for stiffness matrix
    fe = np.zeros((18,1))        # Create zero matrix for distributed load

    numGaussPoints = 3  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight
    
    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)
            N1    = quad9_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            #Calculate Jacobian, inverse Jacobian and determinant of the Jacobian

            J = G @ H # Jacobian
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            B  = np.zeros((3,18))

            for i in range(18):

                if(i%2 == 0):
                    B[0][i] = dNdx[i // 2]
                    B[2][i] = dNdy[i // 2]
            #TODO: add else = 0 again?

            for i in range(18):

                if(i%2 == 1):
                    B[1][i] = dNdy[i // 2]
                    B[2][i] = dNdx[i // 2]
            #TODO: add else = 0 again?


            #TODO: Fill out correct values for displacement interpolation xsi and eta
            N2 = np.zeros((2,18))
            for i in range(18):

                if (i%2 == 0):
                    N2[0][i] = N1[i // 2]

                else:
                    N2[1][i] = N1[i // 2]

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * th * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * th * gw[iGauss] * gw[jGauss]

    if returnBmat: # if specified, return B matrix aswell
        return Ke, fe, B
    return Ke, fe  # Returns stiffness matrix and nodal force vector