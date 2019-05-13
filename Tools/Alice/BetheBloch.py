import numpy as np


def BetheBlochAlephNP(lnbg, kp1=0.76176e-1, kp2=10.632, kp3=0.13279e-4, kp4=1.8631, kp5=1.9479):
    """ numpy version of the dEdx parameterization"""
    bg = np.exp(lnbg)
    beta = bg / np.sqrt(1. + bg * bg)
    aa = np.exp(kp4 * np.log(beta))
    bb = np.exp(-kp5 * np.log(bg))
    bb = np.log(kp3 + bb)
    return (kp2 - aa - bb) * kp1 / aa


def BetheBlochGeantNP(lnbg, kp0=2.33, kp1=0.20, kp2=3.00, kp3=173e-9, kp4=0.49848):
    bg = np.exp(lnbg)
    mK = 0.307075e-3
    me = 0.511e-3
    rho = kp0
    x0 = kp1 * 2.303
    x1 = kp2 * 2.303
    mI = kp3
    mZA = kp4
    bg2 = bg * bg
    maxT = 2 * me * bg2
    d2 = 0.
    x = np.log(bg)
    lhwI = np.log(28.816 * 1e-9 * np.sqrt(rho * mZA) / mI)
    if x > x1:
        d2 = lhwI + x - 0.5
    else:
        if x > x0:
            r = (x1 - x) / (x1 - x0)
            d2 = lhwI + x - 0.5 + (0.5 - lhwI - x0) * r * r * r

    return mK * mZA * (1 + bg2) / bg2 * (0.5 * np.log(2 * me * bg2 * maxT / (mI * mI)) - bg2 / (1 + bg2) - d2)


def BetheBlochSolidNP(lnbg):
    return BetheBlochGeantNP(lnbg)
