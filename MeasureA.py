import numpy as np
    
def MeasureA(LightTransportMeasurements): 
    ####
# This function is used to convert color and light transport measurements of a real material
# to A. It uses brute force on a dense grid to solve optimization problem (3) in the paper.
# Note that also a lookup table can be precomputed to solve (3) in O(1)
# time.
    
    # Input parameters :
# LightTransportMeasurements : nx4 array having columns as:
# [2mm reflectance, 8mm reflectance, Tx, white background reflectance]
    
    # Output parameters :
# A : nx1 array including A values corresponding to input LightTransportMeasurements
# AbsAndScattr : nx2 array including absorption and scattering coeffiecents of reference materials corresponding to input LightTransportMeasurements
####
    ## Loading and interpolating the 6D reference table
    ReferenceTable = np.load('ReferenceTablePost.npy')
    ## finding A
    AbsAndScattr = np.zeros((LightTransportMeasurements.shape[0],2))
    d = 1
    for j in range(LightTransportMeasurements.shape[0]):
        OriginalArrayWhiteBackground = ReferenceTable[:,:,5]
        TestArrayWhiteBackground = LightTransportMeasurements[j,3] * np.ones(OriginalArrayWhiteBackground.shape)
        C = (TestArrayWhiteBackground - OriginalArrayWhiteBackground) ** 2
        #Extract table entries with almost similar reflectance lightness (white backing)
        Index = (np.abs(C) < d ** 2)
        OriginalArray2mm = ReferenceTable[:,:,2]
        OriginalArray2mm = OriginalArray2mm
        OriginalArray8mm = ReferenceTable[:,:,3]
        OriginalArray8mm = OriginalArray8mm
        OriginalArrayDiff = OriginalArray8mm[Index] - OriginalArray2mm[Index]
        OriginalArrayTx = ReferenceTable[:,:,4]
        OriginalArrayTx = OriginalArrayTx
        OriginalArrayTx = OriginalArrayTx[Index]
        #Minimize light transport differences for the extracted values
        dL = LightTransportMeasurements[j,1] - LightTransportMeasurements[j,0]
        TestArrayDiff = dL * np.ones((OriginalArrayDiff.shape))
        TestArrayTx = LightTransportMeasurements[j,2] * np.ones((OriginalArrayTx.shape))
        A = (TestArrayDiff - OriginalArrayDiff) ** 2
        B = (TestArrayTx - OriginalArrayTx) ** 2
        Difference = (A + B)
        OptimIndex = np.argmin(Difference)
        Absorp = ReferenceTable[:,:,0]
        Absorp = Absorp
        Absorp = Absorp[Index]
        Scatt = ReferenceTable[:,:,1]
        Scatt = Scatt
        Scatt = Scatt[Index]
        AbsAndScattr[j,:] = [Absorp[OptimIndex],Scatt[OptimIndex]]
    
    varargout = RefMat2A(AbsAndScattr)
    
    return varargout


def RefMat2A(AbsorptionAndScatteringCoefficients = None): 
    ####
# This function converts the absorption and scattering coefficients of the
# reference material to A (see equation (1))
    
    
    # This script is a part of supplementary material towards the paper "Redefining A in RGBA: Towards a Standard for Graphical 3D
# Printing".
    
    # Authors :
# Philipp Urban, Fraunhofer Institute for Computer Graphic Research IGD, Norwegian University of Science and Technology NTNU
# Tejas Madan Tanksale and Alan Brunton, Fraunhofer Institute for Computer Graphic Research IGD
# Bui Minh Vu and Shigeki Nakauchi, Toyohashi University of Technology
    
    # Parameters :
# AbsorptionAndScatteringCoefficients : Input absorption and scattering coefficients
# (nx2 matrix, first column is the absorption coefficient and second column is the scattering coefficient)
    
    ####
##
    ConstantC = 0.0153
    Eta = EtaAS(AbsorptionAndScatteringCoefficients[:,0],AbsorptionAndScatteringCoefficients[:,1])
    A_ = (1 - np.exp(- ConstantC * Eta))
    A = Phi(A_)
    return A
    
    
def EtaAS(A = None,S = None): 
    ConstantP = 0.4
    Eta = ConstantP * A + S
    return Eta
    
    
def Phi(A_ = None): 
    ConstantQ = 0.6
    Val = A_ ** ConstantQ
    return Val