import numpy as np
import sympy as sp
from sympy import Sum, oo, IndexedBase, Function, I, symbols

from sympyHelpers import *

r = sp.symbols('r', positive=True)
rho = sp.symbols('rho', positive=True)
theta = sp.symbols('theta', real=True)
lamda = sp.symbols('lambda', positive=True)
l = sp.symbols('l', integer=True)
j = sp.symbols('j', integer=True)

n, m = sp.symbols('n m', integer=True)
L = sp.symbols('L', integer=True)
f = sp.symbols('f', real=True)
x0, y0 = sp.symbols('x0, y0', real=True)

def noll_to_zern(jj):
    if (jj == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    nn = 0
    jj1 = jj - 1
    while (jj1 > nn):
        nn += 1
        jj1 -= nn
    mm = (-1)**jj * ((nn % 2) + 2 * int((jj1 + ((nn + 1) % 2)) / 2.0))
    return nn, mm


def kronDelta(ii, jj):
    if ii == jj:
        return 1
    else:
        return 0


def emValue(mm):
    if mm == 0:
        return 2
    else:
        return 1

# Real Zernike Polynomials


def cCoeff(n, m):
    return sp.Piecewise((sp.sqrt(n + 1), m == 0),
                        (sp.sqrt(2 * (n + 1)), m != 0))


def tangetialFunc(m):
    return sp.Piecewise((-sp.sin(m * theta), m < 0),
                        (sp.cos(m * theta), m >= 0))


def complexTangetialFunc(m):
    return sp.exp(I * m * theta)


def radialFunc(n, m):
    ma = sp.Abs(m)
    p = (n - ma) / 2
# return sp.Piecewise( ( ((-1)**p) * (rho**ma) * sp.jacobi(p, ma, 0,
# 1-2*rho**2), sp.Eq(sp.Mod(n-m, 2), 0) ), (0, sp.Eq(sp.Mod(n-m, 2), 1) )
# )
    return ((-1)**p) * (rho**ma) * sp.jacobi(p, ma, 0, 1 - 2 * rho**2)


def realZernike(n, m):
    return radialFunc(n, m) * tangetialFunc(m)


def realZernikeCartesian(n, m, aa):
    return realZernike(n,m).subs(rho,sp.sqrt(x0**2+y0**2)/aa).subs(theta,sp.atan2(x0,y0))


def realZernikeNormalized(n, m):
    return cCoeff(n, m) * realZernike(n, m)

# Complex Zernike Polynomials


def complexZernike(n, m):
    return sp.sqrt(2) * radialFunc(n, m) * complexTangetialFunc(m) / 2


def complexZernikeNormalized(n, m):
    return sp.sqrt(n + 1) * radialFunc(n, m) * complexTangetialFunc(m)

# with this wiegths and this transformation scheme, (realZernike,
# complexZernike) and (realZernikeNormalized, complexZernikeNormalized)
# are coerent


def realZFromComplexZ(zcf, nn, mm):
    zc1 = zcf(n, m).subs({m: mm, n: nn})
    zc2 = zcf(n, m).subs({m: -mm, n: nn})
    if mm >= 0:
        return (zc1 + zc2) / np.sqrt(2)
    elif mm < 0:
        return 1j * (zc1 - zc2) / np.sqrt(2)    
    
    
def _vlj(n, m, l, j):
    p = (n - m) / 2
    q = (n + m) / 2
    return (-1)**p * (m + l + j * 2) * sp.binomial(m + j + l - 1,
                                                   l - 1) * sp.binomial(j + l - 1,
                                                                        l - 1) * sp.binomial(l - 1,
                                                                                             p - j) / sp.binomial(q + l + j,
                                                                                                                  l)


def Vnm(n, m, f, max_order=10):
    ma = sp.Abs(m)
    p = (n - ma) / 2
    q = (n + ma) / 2
    return sp.exp(I * f) * sp.Sum((-2 * I * f)**(l - 1) * sp.Sum((-1)**p * (ma + l + j * 2) * sp.binomial(ma + j + l - 1,
                                                                                                          l - 1) * sp.binomial(j + l - 1,
                                                                                                                               l - 1) * sp.binomial(l - 1,
                                                                                                                                                    p - j) / sp.binomial(q + l + j,
                                                                                                                                                                         l) * (sp.besselj(ma + l + j * 2,
                                                                                                                                                                                          2 * sp.pi * r) / (l * (2 * sp.pi * r)**l)),
                                                                 (j,
                                                                  0,
                                                                  p)),
                                  (l,
                                   1,
                                   max_order))


def diffractedZernikeAtFocus(n, m):
    #    return ((-1)**((n+m)/2) * sp.besselj( n+1, 2*sp.pi*r) / (2*sp.pi*r) ) * tangetialFunc(m)
    # return 2*sp.pi*(-1**(n+1)) * ( sp.besselj( n+1, 2*sp.pi*r) / (2*sp.pi*r)
    # ) * sp.Piecewise( (-sp.sin(m*theta), m<0), (sp.cos(m*theta), m>=0) )
    return 2 * sp.pi * (sp.besselj(n + 1, 2 * sp.pi * r) / (2 * sp.pi * r)) * \
        sp.Piecewise((-sp.sin(m * theta), m < 0), (sp.cos(m * theta), m >= 0))


def diffractedComplexZernikeAtFocus(n, m):
    # return 2*sp.pi* (-1**(n+1)) * ( sp.besselj( n+1, 2*sp.pi*r) /
    # (2*sp.pi*r) ) * complexTangetialFunc(m) / sp.sqrt(2)
    return 2 * sp.pi * (sp.besselj(n + 1, 2 * sp.pi * r) /
                        (2 * sp.pi * r)) * complexTangetialFunc(m) / sp.sqrt(2)


def diffractedZernike(n, m, f=sp.pi, max_order=10):
    return 2 * Vnm(n, m, f, max_order) * tangetialFunc(m)


def diffractedComplexZernike(n, m, f=sp.pi, max_order=10):
    return 2 * Vnm(n, m, f, max_order) * complexTangetialFunc(m) / sp.sqrt(2)


def checkOrthoPair(f1, f2):
    f2c = sp.conjugate(f2)
    res = sp.integrate(f1 * f2 * r, (r, 0, 1), (theta, 0, 2 * sp.pi))
    if (res != 0):
        display(f1)
        display(f2)
    return res


def checkZernikeOrthoPair(n1, m1, n2, m2):
    f1 = realZernikeNormalized(n1, m1)
    f2 = realZernikeNormalized(n2, m2)
    return checkOrthoPair(f1, f2)


def circleDotProduct(f1, f2):
    f2c = sp.conjugate(f2)
    itR = sp.re(f1 * f2c * rho) / sp.S.Pi
    itI = sp.im(f1 * f2c * rho) / sp.S.Pi
    i1 = sp.N(sp.integrate(itR, (theta, 0, 2 * sp.pi), (rho, 0, 1)))
    i2 = sp.N(sp.integrate(itI, (theta, 0, 2 * sp.pi), (rho, 0, 1)))
    return i1 + I * i2

# inputFunction defined over the unit disc, in ro, theta


def zernikeAnalysysReal(inputFunction, max_noll):
    result = [0] * max_noll
    for ni in range(max_noll):
        nn, mm = noll_to_zern(ni + 1)
        z_ni = realZernikeNormalized(n, m).subs({m: int(mm), n: int(nn)})
        result[ni] = circleDotProduct(inputFunction, z_ni)
        print(ni + 1, nn, mm, result[ni] / emValue(mm))
    return result


def zernikeSynthesysReal(decomposition):
    result = sp.S(0)
    for ni, coefficient in enumerate(decomposition):
        nn, mm = noll_to_zern(ni + 1)
        z_ni = realZernikeNormalized(n, m).subs({m: int(mm), n: int(nn)})
        result += coefficient * z_ni
        print(ni + 1, nn, mm, coefficient)
    return result


def zernikeAnalysysComplex(inputFunction, max_noll):
    result = [0] * max_noll
    for ni in range(max_noll):
        nn, mm = noll_to_zern(ni + 1)
        z_ni = complexZernikeNormalized(n, m).subs({m: mm, n: nn})
        result[ni] = circleDotProduct(inputFunction, z_ni)
        print(ni + 1, nn, mm, result[ni])
    return result


def zernikeSynthesysComplex(decomposition):
    result = sp.S(0)
    for ni, coefficient in enumerate(decomposition):
        nn, mm = noll_to_zern(ni + 1)
        z_ni = complexZernikeNormalized(n, m).subs({m: mm, n: nn})
        result += coefficient * z_ni
        print(ni + 1, nn, -mm, coefficient)
    return result


def capital_v_vector(max_noll, f):
    result = [0] * max_noll
    for ni in range(max_noll):
        nn, mm = noll_to_zern(ni + 1)
        z_ni = CVVZernike(nn, mm, rho, theta)
        result[ni] = z_ni
    return result


def NollIndex(nn, mm):
    n4 = nn % 4
    if (mm > 0 and (n4 == 0 or n4 == 1)) or (mm < 0 and (n4 == 2 or n4 == 3)):
        k = 0
    else:
        k = 1
    return nn * (nn + 1) // 2 + abs(mm) + k


def order_to_max_noll(nn):
    return (nn + 2) * (nn + 1) // 2


def createZernikeFormulary(lastMode):
    zf = Formulary()
    for i in range(2, lastMode + 1):
        idx = noll_to_zern(i)
        idstr = str(idx[0]) + str(idx[1])
        zzz = sp.symbols('Z_' + idstr)
        az = realZernike(n, m).subs({m: int(idx[1]), n: int(idx[0])})
        zname = 'Z' + idstr
        zf.addFormula(zname, (zzz, az, sp.Eq(zzz, az)))
    return zf


def getZernikeDomain(nn):
    x1 = np.linspace(-1.0, 1.0, nn)
    y1 = np.linspace(-1.0, 1.0, nn)
    X1, Y1 = np.meshgrid(x1, y1)
    rr = np.sqrt(X1**2 + Y1**2)
    rr[np.where(rr > 1.0)] = np.nan
    return rr, np.arctan2(Y1, X1)


def evaluateZernike(zerninke_mode_expression, sampling_points):
    zerninke_mode_lambda = sp.lambdify(
        [rho, theta], zerninke_mode_expression, 'numpy')
#    r1 = 1.0 - np.geomspace(1.0/sampling_points, 1.0, sampling_points, endpoint=True)
    r1 = np.power(np.linspace(0.0, 1.0, sampling_points), 1.0 / 2.0)
    theta1 = np.linspace(0, 2 * np.pi, sampling_points)
    r1, theta1 = np.meshgrid(r1, theta1)
    X1 = r1 * np.sin(theta1)
    Y1 = r1 * np.cos(theta1)
    Z1 = np.asarray(zerninke_mode_lambda(r1, theta1))
    return X1, Y1, Z1
