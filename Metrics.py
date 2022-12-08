import numpy as np

def DE2000(lab1, lab2):
    '''
    Calculates the CIEDE2000 metric between 2 numpy arrays of lab colors
    '''
       
    L1 = lab1[:,0]
    A1 = lab1[:,1]
    B1 = lab1[:,2]
    L2 = lab2[:,0]
    A2 = lab2[:,1]
    B2 = lab2[:,2]   
    kL = 1
    kC = 1
    kH = 1
    
    mask_value_0_input1=((A1==0)*(B1==0))
    mask_value_0_input2=((A2==0)*(B2==0))
    mask_value_0_input1_no=1-mask_value_0_input1
    mask_value_0_input2_no=1-mask_value_0_input2
    B1=B1+0.0001*mask_value_0_input1
    B2=B2+0.0001*mask_value_0_input2 
    
    C1 = np.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = np.sqrt((A2 ** 2.) + (B2 ** 2.))   
   
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - np.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = np.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = np.sqrt((a2P ** 2.) + (B2 ** 2.))


    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P=h1P*mask_value_0_input1_no
    h2P=h2P*mask_value_0_input2_no 
    
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * np.sqrt(c1P * c2P) * np.sin(radians(dhP) / 2.)
    mask_0_no=1-np.maximum(mask_value_0_input1,mask_value_0_input2)
    dHP=dHP*mask_0_no

    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * np.cos(radians(aHP - 39)) + 0.24 * np.cos(radians(2. * aHP)) + 0.32 * np.cos(radians(3. * aHP + 6.)) - 0.2 * np.cos(radians(4. * aHP - 63.))
    dRO = 30. * np.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = np.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / np.sqrt(20. + ((aL - 50.) ** 2.)))
    
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * np.sin(radians(2. * dRO))

#     res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))

    res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.)*mask_0_no + ((dHP / (sH * kH)) ** 2.)*mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))*mask_0_no
    mask_0=(res_square<=0)
    mask_0_no=1-mask_0
    res_square=res_square+0.0001*mask_0    
    res=np.sqrt(res_square)
    res=res*mask_0_no

    return res

def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)
def hpf_diff(x, y):
    mask1=((x == 0) * (y == 0))
    mask1_no = 1-mask1

    tmphp = degrees(np.arctan2(x*mask1_no, y*mask1_no))
    tmphp1 = tmphp * (tmphp >= 0)
    tmphp2 = (360+tmphp)* (tmphp < 0)

    return tmphp1+tmphp2

def dhpf_diff(c1, c2, h1p, h2p):

    mask1  = ((c1 * c2) == 0)
    mask1_no  = 1-mask1
    res1=(h2p - h1p)*mask1_no*(np.abs(h2p - h1p) <= 180)
    res2 = ((h2p - h1p)- 360) * ((h2p - h1p) > 180)*mask1_no
    res3 = ((h2p - h1p)+360) * ((h2p - h1p) < -180)*mask1_no

    return res1+res2+res3

def ahpf_diff(c1, c2, h1p, h2p):

    mask1=((c1 * c2) == 0)
    mask1_no=1-mask1
    mask2=(np.abs(h2p - h1p) <= 180)
    mask2_no=1-mask2
    mask3=(np.abs(h2p + h1p) < 360)
    mask3_no=1-mask3

    res1 = (h1p + h2p) *mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3 
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1+res2+res3)+(res1+res2+res3)*mask1
    return res*0.5