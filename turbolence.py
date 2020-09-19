from scipy.special import *
import scipy
import time
import random
import sympy as sp

from SEEING.seeing import *

def createTurbolenceFormulary():
    #R_q = sp.Function("\U0000211B_q")(r)
    V0, L0, ni0, Re, l0, r0, r, c = sp.symbols(r'V_0 L_0 \nu_0 Re l_0 r_0 r c', positive=True)
    h, f, k, k0, km, gamma = sp.symbols(r'h f k k_0 k_m \gamma', positive=True)
    dx, dt, v = sp.symbols("\u03B4x \u03B4t v", positive=True)
    cn2 = sp.Function("C_N")(h)
    dPhi = sp.Function("D_phi")(dx, dt)
    dPhi_r = sp.Function("D_phi")(r)
    cPhi_r = sp.Function("C_phi")(r)
    pPhi = sp.Function("P_phi")(k)
    wPhi = sp.Function("W_phi")(f)
    
    def ReynoldsNumber():
        _lhs = Re
        _rhs = V0 * L0 / ni0
        return sp.Eq(_lhs, _rhs)


    def innerScale():
        _lhs = l0
        _rhs = L0 / Re ** (sp.S(3) / sp.S(4))
        return sp.Eq(_lhs, _rhs)


    def FriedParameter():
        _lhs = r0
        with sp.evaluate(False):
            _rhs = ((0.423 * k**2 / sp.cos(gamma)) *
                    sp.Integral(cn2, (h, 0, sp.oo))) ** (-sp.S(3) / sp.S(5))
        return sp.relational.Eq(_lhs, _rhs)


    def phaseStructureFunctionOrig():
        _lhs = dPhi
        _rhs = 6.88 * (sp.Abs(dx - v * dt) / r0) ** (sp.Rational(5.0 / 3.0))
        return sp.relational.Eq(_lhs, _rhs)


    def phaseStructureFunctionOrig_r():
        _lhs = dPhi_r
        with sp.evaluate(False):
            _rhs = 6.88 * (r / r0)**(sp.S(5) / sp.S(3))
        return sp.relational.Eq(_lhs, _rhs)


    def phaseStructureFunction():
        _lhs = dPhi_r
        expr0 = c * (L0 / r0)**(sp.S(5) / sp.S(3)) * \
            sp.gamma(sp.S(5) / sp.S(6)) / (sp.S(2)**(sp.S(1) / sp.S(6)))
        frac1 = sp.S(5) / sp.S(3)
        frac2 = sp.S(5) / sp.S(6)
        frac3 = sp.S(1) / sp.S(6)
        frac4 = sp.S(11) / sp.S(6)
        frac5 = sp.S(8) / sp.S(3)
        frac6 = sp.S(24) / sp.S(5)
        frac7 = sp.S(6) / sp.S(5)
        c_expr = (sp.S(2)**frac3 * sp.gamma(frac4) / sp.pi**frac5) * \
            (frac6 * sp.gamma(frac7)) ** frac2
        _rhs = D_expr = c_expr * (L0 / r0)**frac1 * (sp.gamma(frac2) / sp.S(2)**frac3 - (
            2 * sp.pi * r / L0)**frac2 * sp.besselk(frac2, 2 * sp.pi * r / L0))
        return sp.relational.Eq(_lhs, _rhs)


    def phaseVariance():
        _lhs = cPhi_r
        frac1 = sp.S(5) / sp.S(3)
        frac2 = sp.S(5) / sp.S(6)
        frac3 = sp.S(1) / sp.S(6)
        frac4 = sp.S(11) / sp.S(6)
        frac5 = sp.S(8) / sp.S(3)
        frac6 = sp.S(24) / sp.S(5)
        frac7 = sp.S(6) / sp.S(5)
        c_expr = (sp.S(2)**frac3 * sp.gamma(frac4) / sp.pi**frac5) * \
            (frac6 * sp.gamma(frac7)) ** frac2
        _rhs = C_expr = (L0 / r0) ** frac1 * (c_expr / 2) * (2 *
                                                             sp.pi * r / L0)**frac2 * sp.besselk(frac2, 2 * sp.pi * r / L0)
        return sp.relational.Eq(_lhs, _rhs)


    #P_expr = 0.00058 * r0**(-sp.S(5)/sp.S(3)) * (f**2 + (1/L0**2)) ** ( -sp.S(11) / sp.S(6) )
    def phaseSpatialPowerSpectrumKolmogorov():
        _lhs = pPhi
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * k ** (-sp.S(11) / sp.S(3))

        return sp.relational.Eq(_lhs, _rhs)


    def phaseSpatialPowerSpectrumVonKarmanO():
        _lhs = pPhi
        k0 = 2 * sp.pi / L0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * \
                (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))

        return sp.relational.Eq(_lhs, _rhs)


    def phaseSpatialPowerSpectrumVonKarmanO_f():
        _lhs = wPhi
        f0 = 1 / L0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * \
                (f**2 + f0**2) ** (-sp.S(11) / sp.S(6))

        return sp.relational.Eq(_lhs, _rhs)


    def phaseSpatialPowerSpectrumVonKarman():
        # Von-Karman outer/inner
        _lhs = pPhi
        k0 = 2 * sp.pi / L0
        km = 5.92 / l0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * (k**2 + k0 **
                                                        2) ** (-sp.S(11) / sp.S(6)) * sp.exp(-(k * l0 / 5.92)**2)

        return sp.relational.Eq(_lhs, _rhs)
    
    
    _turbolenceFormulas = Formulary("Turbolence",
                                   ['ReynoldsNumber',
                                    'innerScale',
                                    'FriedParameter',
                                    'phaseStructureFunctionOrig',
                                    'phaseStructureFunctionOrig_r',
                                    'phaseStructureFunction',
                                    'phaseVariance',
                                    'phaseSpatialPowerSpectrumKolmogorov',
                                    'phaseSpatialPowerSpectrumVonKarmanO',
                                    'phaseSpatialPowerSpectrumVonKarmanO_f',
                                    'phaseSpatialPowerSpectrumVonKarman'],
                                   [ReynoldsNumber(),
                                       innerScale(),
                                       FriedParameter(),
                                       phaseStructureFunctionOrig(),
                                       phaseStructureFunctionOrig_r(),
                                       phaseStructureFunction(),
                                       phaseVariance(),
                                       phaseSpatialPowerSpectrumKolmogorov(),
                                       phaseSpatialPowerSpectrumVonKarmanO(),
                                       phaseSpatialPowerSpectrumVonKarmanO_f(),
                                       phaseSpatialPowerSpectrumVonKarman()])


    return _turbolenceFormulas


turbolenceFormulas = createTurbolenceFormulary()

pplib = {
    'factorial': factorial,
    'binomial': binom,
    'besselj': jv,
    'besselk': kv,
    'besseli': iv,
    'bessely': yv,
    'erf': erf,
    'gamma': gamma}

scipyext = [pplib, "scipy"]


def ft_ift2(G, delta_f):
    N = G.shape[0]
    g = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(G))) * (N * delta_f)**2
    return g


def ft_PSD_phi(rrr0, N, frq_range, L0, l0, method='VonKarman'):
    del_f = frq_range / N
    fx = np.arange(-N / 2., N / 2.) * del_f
    (fx, fy) = np.meshgrid(fx, fx)
    f = np.sqrt(fx**2 + fy**2)
    if method == 'VonKarman':
        (_, PSD_phi) = turbolenceFormulas.evaluateFormula('phaseSpatialPowerSpectrumVonKarman', {
            'r_0': rrr0, 'L_0': L0, 'l_0': l0}, ['k'], [2 * np.pi * f], scipyext)
    else:
        (_, PSD_phi) = turbolenceFormulas.evaluateFormula(
            'phaseSpatialPowerSpectrumKolmogorov', {'r_0': rrr0}, ['k'], [2 * np.pi * f], scipyext)
    PSD_phi[int(N / 2), int(N / 2)] = 0
    return PSD_phi, del_f


def ft_phase_screen(rrr0, N, delta, L0, l0, method='VonKarman', seed=None):
    delta = float(delta)
    R = random.SystemRandom(time.time())
    if seed is None:
        seed = int(R.random() * 100000)
    np.random.seed(seed)
    frq_range = 1.0 / delta
    del_f = frq_range / N
    PSD_phi = ft_PSD_phi(rrr0, N, frq_range, L0, l0, method)
    cn = (
        (np.random.normal(
            size=(
                N,
                N)) +
         1j *
         np.random.normal(
            size=(
                N,
                N))) *
        np.sqrt(PSD_phi) *
        del_f)
    phs = ft_ift2(cn, 1).real
    return phs
