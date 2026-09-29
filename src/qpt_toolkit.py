"""
qpt_toolkit.py - Bo cong cu Python de GIAI va KIEM TRA dap an khi on QPT (WQU MScFE)
====================================================================================

Cach dung (Jupyter hoac terminal):
    from qpt_toolkit import *
    show("det cua tong", det_of_sum([[1,2],[3,4]], [[0,1],[1,0]]))

Chay tu kiem tra (tat ca vi du phai in OK):
    python qpt_toolkit.py

Cai thu vien neu thieu:
    pip install numpy scipy sympy

Nguyen tac:
  * Bieu thuc viet dang chuoi kieu Python: "exp(x)/x**2", "sin(x)+cos(3*x)", "ln(x)", "pi".
    Dau ^ cung duoc (tu doi thanh **).
  * Cong cu nay de KIEM TRA va TINH NHANH. Van phai tu nhan ra dang bai va chon dung ham.
  * Phan cuoi file ("TANG 2") la kien thuc nen cho MScFE, chua nam trong pham vi de mau QPT.

Muc luc:
  1 Algebra | 2 Linear Algebra | 3 Probability | 4 Statistics | 5 Differential Calculus
  6 Integral Calculus | 7 Differential Equations | 8 Discrete Math | 9 Python notes
  TANG 2: Lagrange, hypothesis test, OLS, PCA
"""
from __future__ import annotations

import copy
import itertools
import math
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy import integrate as _integrate
from scipy import stats

# ----------------------------------------------------------------------------
# 0. Ham dung chung
# ----------------------------------------------------------------------------
x, y, z, t, a, b, c, n, k, m, r, s, u, v, w = sp.symbols("x y z t a b c n k m r s u v w")
lam = sp.Symbol("lam")

_NS = {
    "x": x, "y": y, "z": z, "t": t, "a": a, "b": b, "c": c, "n": n, "k": k,
    "m": m, "r": r, "s": s, "u": u, "v": v, "w": w, "lam": lam,
    "e": sp.E, "E": sp.E, "pi": sp.pi, "ln": sp.log, "oo": sp.oo, "inf": sp.oo,
}


def S(expr):
    """Chuoi -> bieu thuc sympy (giu nguyen neu da la sympy/so)."""
    if isinstance(expr, str):
        return sp.sympify(expr, locals=_NS)
    return sp.sympify(expr)


def _sym(name):
    return S(name) if isinstance(name, str) else name


def show(label, value):
    """In ket qua kem gia tri so thap phan (neu co the)."""
    line = f"{label}: {value}"
    try:
        approx = sp.N(value) if not isinstance(value, (list, tuple, dict)) else None
        if approx is not None and str(approx) != str(value):
            line += f"   (~ {approx})"
    except Exception:
        pass
    print(line)
    return value


def _mat(A):
    return A if isinstance(A, sp.MatrixBase) else sp.Matrix(A)


# ----------------------------------------------------------------------------
# 1. COLLEGE ALGEBRA  (ham so, log/exp, ham nguoc, bat phuong trinh)
# ----------------------------------------------------------------------------
def solve_eq(eq, var="x"):
    """Giai phuong trinh. eq dang 'lhs = rhs' hoac bieu thuc (= 0).
    Vi du: solve_eq("exp(2*x) = 7"), solve_eq("x**2 - 5*x + 6")
    """
    v_ = _sym(var)
    if isinstance(eq, str) and "=" in eq:
        lhs, rhs = eq.split("=", 1)
        equation = sp.Eq(S(lhs), S(rhs))
    else:
        equation = sp.Eq(S(eq), 0)
    return sp.solve(equation, v_)


def inverse_function(f, var="x"):
    """Ham nguoc: giai y = f(x) theo x. Vi du: inverse_function("ln(x)**2/3")"""
    v_ = _sym(var)
    yy = sp.Symbol("y_out")
    return sp.solve(sp.Eq(yy, S(f)), v_)


def solve_ineq(ineq, var="x"):
    """Bat phuong trinh mot bien. Vi du: solve_ineq("x**2 - 5*x + 6 < 0")"""
    return sp.solve_univariate_inequality(S(ineq), _sym(var), relational=False)


def log_base(value, base):
    """log_base(value) bang cong thuc doi co so: ln(value)/ln(base)."""
    return math.log(value) / math.log(base)


def simplify_expr(expr):
    return sp.simplify(S(expr))


def expand_expr(expr):
    return sp.expand(S(expr))


def factor_expr(expr):
    return sp.factor(S(expr))


# ----------------------------------------------------------------------------
# 2. LINEAR ALGEBRA  (ma tran, dinh thuc, nghich dao, he pt, eigen)
# ----------------------------------------------------------------------------
def det(A):
    return _mat(A).det()


def det_of_sum(A, B):
    """Dinh thuc cua TONG hai ma tran: det(A+B) (KHONG phai det(A)+det(B))."""
    return (_mat(A) + _mat(B)).det()


def inv(A):
    return _mat(A).inv()


def matmul(A, B):
    return _mat(A) * _mat(B)


def transpose(A):
    return _mat(A).T


def rank(A):
    return _mat(A).rank()


def trace(A):
    return _mat(A).trace()


def rref(A):
    """Ma tran bac thang rut gon + cac cot pivot (Gaussian-Jordan)."""
    return _mat(A).rref()


def solve_linear(A, b_vec):
    """Giai A x = b. Tra ve tap nghiem (duy nhat / vo so / rong)."""
    xs = sp.symbols(f"x1:{_mat(A).shape[1] + 1}")
    return sp.linsolve((_mat(A), _mat(b_vec)), *xs)


def char_poly(A):
    """Da thuc dac trung det(A - lam*I)."""
    return sp.factor(_mat(A).charpoly(lam).as_expr())


def eigen(A):
    """Tra ve list (eigenvalue, boi, [eigenvector])."""
    return _mat(A).eigenvects()


# ----------------------------------------------------------------------------
# 3. PROBABILITY  (dem, nhi thuc, Poisson, chuan, Bayes, ky vong)
# ----------------------------------------------------------------------------
def perm(n_, r_):
    return math.perm(n_, r_)


def comb(n_, r_):
    return math.comb(n_, r_)


def binom_exact(n_, k_, p):
    """P(X = k) nhi thuc dang phan so CHINH XAC. p co the la '1/6'."""
    p_ = sp.Rational(str(p)) if isinstance(p, (str, Fraction)) else sp.nsimplify(p)
    return sp.binomial(n_, k_) * p_**k_ * (1 - p_) ** (n_ - k_)


def binom_pmf(n_, k_, p):
    return stats.binom.pmf(k_, n_, p)


def binom_cdf(n_, k_, p):
    """P(X <= k)."""
    return stats.binom.cdf(k_, n_, p)


def poisson_pmf(lmbda, k_):
    return stats.poisson.pmf(k_, lmbda)


def poisson_cdf(lmbda, k_):
    return stats.poisson.cdf(k_, lmbda)


def normal_cdf(x_, mu=0.0, sigma=1.0):
    """P(X <= x)."""
    return stats.norm.cdf(x_, mu, sigma)


def normal_between(lo, hi, mu=0.0, sigma=1.0):
    """P(lo <= X <= hi)."""
    return stats.norm.cdf(hi, mu, sigma) - stats.norm.cdf(lo, mu, sigma)


def normal_ppf(p, mu=0.0, sigma=1.0):
    """Gia tri x sao cho P(X <= x) = p."""
    return stats.norm.ppf(p, mu, sigma)


def expon_cdf(x_, rate):
    """P(X <= x) phan phoi mu voi toc do lambda = rate."""
    return stats.expon.cdf(x_, scale=1 / rate)


def uniform_prob(lo, hi, a_, b_):
    """P(a <= X <= b) voi X ~ Uniform(lo, hi)."""
    a_, b_ = max(a_, lo), min(b_, hi)
    return max(0.0, (b_ - a_) / (hi - lo))


def bayes(prior, p_e_given_h, p_e_given_not_h):
    """P(H | E) = P(E|H)P(H) / [P(E|H)P(H) + P(E|not H)P(not H)]."""
    num = p_e_given_h * prior
    return num / (num + p_e_given_not_h * (1 - prior))


def bayes_from_evidence(prior, p_e_given_h, p_e):
    """P(H | E) when the marginal probability P(E) is known directly."""
    return p_e_given_h * prior / p_e


def expectation(values, probs):
    return sum(v_ * p_ for v_, p_ in zip(values, probs))


def variance_discrete(values, probs):
    mu = expectation(values, probs)
    return sum(p_ * (v_ - mu) ** 2 for v_, p_ in zip(values, probs))


# ----------------------------------------------------------------------------
# 4. STATISTICS  (mean, phuong sai, z-score, khoang tin cay, tuong quan)
# ----------------------------------------------------------------------------
def mean(data):
    return float(np.mean(data))


def median(data):
    return float(np.median(data))


def mode(data):
    values, counts = np.unique(data, return_counts=True)
    return values[counts == counts.max()].tolist()


def variance(data, sample=True):
    """sample=True: chia n-1 (mau); sample=False: chia n (tong the)."""
    return float(np.var(data, ddof=1 if sample else 0))


def std(data, sample=True):
    return math.sqrt(variance(data, sample))


def z_score(x_, mu, sigma):
    return (x_ - mu) / sigma


def z_critical(confidence=0.95):
    """Moc z hai phia: 90% -> 1.645, 95% -> 1.96, 99% -> 2.576."""
    return float(stats.norm.ppf(1 - (1 - confidence) / 2))


def ci_mean_z(sample_mean, sigma, n_, confidence=0.95):
    """Khoang tin cay cho mean khi biet sigma."""
    half = z_critical(confidence) * sigma / math.sqrt(n_)
    return sample_mean - half, sample_mean + half


def ci_mean_t(data, confidence=0.95):
    """Khoang tin cay cho mean khi chua biet sigma (dung t)."""
    d = np.asarray(data, dtype=float)
    lo, hi = stats.t.interval(confidence, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return float(lo), float(hi)


def cov(xs, ys, sample=True):
    return float(np.cov(xs, ys, ddof=1 if sample else 0)[0, 1])


def corr(xs, ys):
    return float(np.corrcoef(xs, ys)[0, 1])


def linreg(xs, ys):
    """Hoi quy don bien: tra ve (slope, intercept, r)."""
    res = stats.linregress(xs, ys)
    return float(res.slope), float(res.intercept), float(res.rvalue)


# ----------------------------------------------------------------------------
# 5. DIFFERENTIAL CALCULUS  (dao ham, limit, Taylor)
# ----------------------------------------------------------------------------
def derivative(f, var="x", order=1):
    """Dao ham. Vi du: derivative("exp(x)/x**2") roi factor_expr de rut gon."""
    return sp.simplify(sp.diff(S(f), _sym(var), order))


def partial(f, var):
    return sp.diff(S(f), _sym(var))


def gradient(f, variables=("x", "y")):
    return [sp.diff(S(f), _sym(v_)) for v_ in variables]


def limit_of(f, var="x", point=0, direction="+-"):
    """Gioi han. point co the la 'oo' hoac '-oo'. direction: '+', '-', '+-'."""
    return sp.limit(S(f), _sym(var), S(point), dir=direction)


def taylor(f, about=0, order=2, var="x"):
    """Khai trien Taylor den bac `order` quanh diem `about` (bo phan du)."""
    return sp.series(S(f), _sym(var), S(about), order + 1).removeO()


def evaluate(expr, **values):
    """Thay so vao bieu thuc: evaluate("x**2+1", x=3)"""
    return S(expr).subs({_sym(k_): v_ for k_, v_ in values.items()})


# ----------------------------------------------------------------------------
# 6. INTEGRAL CALCULUS  (nguyen ham, tich phan xac dinh)
# ----------------------------------------------------------------------------
def antiderivative(f, var="x"):
    """Nguyen ham (chua ghi +C - tu nho them +C!)."""
    return sp.integrate(S(f), _sym(var))


def integral(f, lo, hi, var="x"):
    """Tich phan xac dinh chinh xac. lo/hi co the la 'pi', '-pi/2', 'oo'."""
    return sp.simplify(sp.integrate(S(f), (_sym(var), S(lo), S(hi))))


def integral_numeric(f, lo, hi, var="x"):
    """Kiem tra so bang scipy.quad (doi chieu voi ket qua chinh xac)."""
    fn = sp.lambdify(_sym(var), S(f), "numpy")
    val, _ = _integrate.quad(fn, float(S(lo)), float(S(hi)))
    return val


# ----------------------------------------------------------------------------
# 7. DIFFERENTIAL EQUATIONS
# ----------------------------------------------------------------------------
def ode(eq, y0=None, dy0=None, func="y", var="t"):
    """Giai ODE bang sympy.dsolve.
    Viet dao ham bang Derivative(y(t), t) hoac y(t).diff(t):
        ode("Derivative(y(t), t, 2) + 6*Derivative(y(t), t) + 9*y(t) = 0", y0=2, dy0=-10)
        ode("Derivative(y(t), t) = 3*y(t)", y0=5)          # bac 1 tach bien
    Dat var='x' va func='y' neu de viet theo x.
    """
    F = sp.Function(func)
    ns = {**_NS, func: F}
    v_ = _sym(var)
    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        equation = sp.Eq(sp.sympify(lhs, locals=ns), sp.sympify(rhs, locals=ns))
    else:
        equation = sp.Eq(sp.sympify(eq, locals=ns), 0)
    ics = {}
    if y0 is not None:
        ics[F(0)] = y0
    if dy0 is not None:
        ics[F(v_).diff(v_).subs(v_, 0)] = dy0
    return sp.dsolve(equation, F(v_), ics=ics or None)


def char_roots(A2, A1, A0):
    """Nghiem phuong trinh dac trung A2*r^2 + A1*r + A0 = 0 cua ODE bac 2 he so hang.
    Delta > 0: 2 nghiem thuc | Delta = 0: nghiem kep | Delta < 0: nghiem phuc."""
    delta = A1**2 - 4 * A2 * A0
    roots = sp.solve(A2 * r**2 + A1 * r + A0, r)
    kind = "2 nghiem thuc" if delta > 0 else ("nghiem kep" if delta == 0 else "nghiem phuc")
    return {"delta": delta, "loai": kind, "nghiem": roots}


# ----------------------------------------------------------------------------
# 8. DISCRETE MATHEMATICS  (so hoc, modular, logic)
# ----------------------------------------------------------------------------
def gcd_steps(a_, b_, verbose=True):
    """Thuat toan Euclid, in tung buoc. Tra ve UCLN."""
    while b_:
        if verbose:
            print(f"{a_} = {a_ // b_} * {b_} + {a_ % b_}")
        a_, b_ = b_, a_ % b_
    return a_


def lcm(a_, b_):
    return abs(a_ * b_) // math.gcd(a_, b_)


def is_prime(n_):
    return sp.isprime(n_)


def prime_factors(n_):
    """Phan tich thua so nguyen to: {so nguyen to: so mu}."""
    return sp.factorint(n_)


def divisors(n_):
    return sp.divisors(n_)


def mod_pow(base, exp, modulus):
    return pow(base, exp, modulus)


def mod_inverse(a_, modulus):
    return pow(a_, -1, modulus)


def find_counterexample(premise, conclusion, limit=10_000):
    """Tim so nguyen duong n dau tien de premise(n) dung nhung conclusion(n) sai.
    Vi du ('n^2 chia het cho 12' thi 'n chia het cho 4' co bat buoc khong?):
        find_counterexample(lambda n: n*n % 12 == 0, lambda n: n % 4 == 0)   # -> 6
    Tra ve None neu khong co phan vi du trong pham vi tim.
    """
    for n_ in range(1, limit + 1):
        if premise(n_) and not conclusion(n_):
            return n_
    return None


def implies(p, q):
    return (not p) or q


def truth_table(expr, names=("p", "q")):
    """Bang chan tri. expr la chuoi Python, dung and/or/not/implies(p, q).
    Vi du: truth_table("implies(p, q) == (not p or q)")"""
    print(" ".join(f"{nm:>5}" for nm in names) + "   ket qua")
    rows = []
    for vals in itertools.product([True, False], repeat=len(names)):
        env = dict(zip(names, vals))
        env["implies"] = implies
        res = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - chi de tu hoc
        rows.append((*vals, res))
        print(" ".join(f"{str(v_):>5}" for v_ in vals) + f"   {res}")
    return rows


# ----------------------------------------------------------------------------
# 9. PYTHON NOTES  (doc hieu code, data structures)
# ----------------------------------------------------------------------------
def methods_of(obj):
    """Liet ke method cong khai: methods_of(dict) -> keys, values, items, get, ..."""
    return [nm for nm in dir(obj) if not nm.startswith("_")]


def copy_demo():
    """Shallow vs deep copy: sua phan tu long ben trong."""
    original = [[1, 2], [3, 4]]
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)
    original[0][0] = 99
    print("original:", original)
    print("shallow :", shallow, "<- bi doi theo (chia se list con)")
    print("deep    :", deep, "<- khong doi")


PY_CHEATS = {
    "dict": "keys() values() items() get() pop() update() setdefault() clear() copy()  (KHONG co elements())",
    "list": "append() extend() insert() remove() pop() sort() reverse() index() count()",
    "slicing": "a[start:stop:step]; stop KHONG bao gom; a[::-1] dao nguoc; a[-1] phan tu cuoi",
    "range": "range(5) -> 0..4; range(2, 10, 3) -> 2, 5, 8",
    "comprehension": "[f(i) for i in xs if cond]  |  {k: v for k, v in pairs}",
    "division": "7 / 2 = 3.5 | 7 // 2 = 3 | 7 % 2 = 1 | 2 ** 3 = 8",
    "mutable": "list dict set THAY DOI duoc; tuple str int KHONG",
}


# ----------------------------------------------------------------------------
# TANG 2 - nen cho MScFE (chua nam trong de mau QPT)
# ----------------------------------------------------------------------------
def lagrange(f, constraint, variables=("x", "y")):
    """Cuc tri co dieu kien: toi uu f voi rang buoc g = 0 (nhap g dang bieu thuc = 0).
    Vi du: lagrange("x*y", "x + y - 10")  -> x = y = 5"""
    vs = [_sym(v_) for v_ in variables]
    L = S(f) - lam * S(constraint)
    eqs = [sp.diff(L, v_) for v_ in vs] + [S(constraint)]
    return sp.solve(eqs, vs + [lam], dict=True)


def z_test_mean(sample_mean, mu0, sigma, n_, alternative="two-sided"):
    """Kiem dinh z cho mean (biet sigma). Tra ve (z, p_value)."""
    z_ = (sample_mean - mu0) / (sigma / math.sqrt(n_))
    if alternative == "two-sided":
        p = 2 * (1 - stats.norm.cdf(abs(z_)))
    elif alternative == "greater":
        p = 1 - stats.norm.cdf(z_)
    else:
        p = stats.norm.cdf(z_)
    return float(z_), float(p)


def t_test_mean(data, mu0):
    """Kiem dinh t mot mau. Tra ve (t, p_value) hai phia."""
    res = stats.ttest_1samp(data, mu0)
    return float(res.statistic), float(res.pvalue)


def ols(X, y_):
    """Hoi quy da bien: tra ve he so [intercept, b1, b2, ...] va R^2."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y_, dtype=float), rcond=None)
    pred = A @ beta
    ss_res = float(np.sum((np.asarray(y_) - pred) ** 2))
    ss_tot = float(np.sum((np.asarray(y_) - np.mean(y_)) ** 2))
    return beta.tolist(), 1 - ss_res / ss_tot


def pca(X, n_components=2):
    """PCA co ban qua eigen cua ma tran hiep phuong sai.
    Tra ve (ty le phuong sai giai thich, cac thanh phan chinh)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(Xc, rowvar=False))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return (vals / vals.sum())[:n_components], vecs[:, :n_components]


# ----------------------------------------------------------------------------
# Tu kiem tra: python qpt_toolkit.py
# ----------------------------------------------------------------------------
def _close(p, q, tol=1e-6):
    return abs(float(p) - float(q)) < tol


def run_examples():
    ok = []

    def check(name, cond):
        print(("OK   " if cond else "FAIL ") + name)
        ok.append(cond)

    # 1 Algebra
    check(
        "solve_eq exp(2x)=7",
        any(
            solution.is_real and _close(solution, math.log(7) / 2)
            for solution in solve_eq("exp(2*x) = 7")
        ),
    )
    check("inverse_function ln(x)", sp.simplify(inverse_function("ln(x)")[0] - sp.exp(sp.Symbol("y_out"))) == 0)
    check("solve_ineq x^2-5x+6<0", solve_ineq("x**2 - 5*x + 6 < 0") == sp.Interval.open(2, 3))
    check("log_base(8, 2)", _close(log_base(8, 2), 3))
    # 2 Linear Algebra
    check("det_of_sum", det_of_sum([[1, 2], [3, 4]], [[0, 1], [1, 0]]) == -8)
    check("inv 2x2", inv([[2, 1], [1, 1]]) == sp.Matrix([[1, -1], [-1, 2]]))
    check("rank", rank([[1, 2], [2, 4]]) == 1)
    check("solve_linear", solve_linear([[1, 1], [1, -1]], [3, 1]) == sp.FiniteSet((2, 1)))
    check("eigenvalues", sorted(v_[0] for v_ in eigen([[2, 0], [0, 3]])) == [2, 3])
    # 3 Probability
    check("binom_exact(5,2,1/3)", binom_exact(5, 2, "1/3") == sp.Rational(80, 243))
    check("bayes", _close(bayes(0.01, 0.99, 0.05), 1 / 6, 1e-4))
    check("bayes_from_evidence", _close(bayes_from_evidence(0.01, 0.9, 0.0585), 0.1538461538))
    check("normal 1.96", _close(normal_cdf(1.96), 0.9750021, 1e-5))
    check("poisson_pmf(2, 0)", _close(poisson_pmf(2, 0), math.exp(-2)))
    check("expectation", _close(expectation([1, 2, 3], [0.2, 0.5, 0.3]), 2.1))
    # 4 Statistics
    d = [2, 4, 4, 4, 5, 5, 7, 9]
    check("mean/var/std", _close(mean(d), 5) and _close(variance(d, False), 4) and _close(std(d, False), 2))
    check("z_critical", _close(z_critical(0.95), 1.959964, 1e-5) and _close(z_critical(0.99), 2.575829, 1e-5))
    check("linreg", _close(linreg([1, 2, 3], [2, 4, 6])[0], 2))
    # 5 Differential Calculus
    check("derivative", sp.simplify(derivative("x**2*exp(x)") - sp.exp(x) * (x**2 + 2 * x)) == 0)
    check("limit sin(x)/x", limit_of("sin(x)/x", "x", 0) == 1)
    check("limit at infinity", limit_of("(2*x+1)/(x-3)", "x", "oo") == 2)
    tay = taylor("sqrt(x)", 4, 2)
    check("taylor sqrt(4.2)", _close(tay.subs(x, 4.2), 2.049375, 1e-6))
    # 6 Integral Calculus
    check("integral sin 0..pi", integral("sin(x)", 0, "pi") == 2)
    check("integral numeric", _close(integral_numeric("x**2", 0, 3), 9))
    # 7 ODE
    sol = ode("Derivative(y(t), t, 2) + 4*Derivative(y(t), t) + 4*y(t) = 0", y0=1, dy0=0)
    check("ode repeated root", _close(sol.rhs.subs(t, 1), 3 * math.exp(-2)))
    check("char_roots", char_roots(1, 6, 9)["loai"] == "nghiem kep")
    # 8 Discrete
    check("gcd_steps", gcd_steps(48, 18, verbose=False) == 6)
    check("find_counterexample", find_counterexample(lambda q: q * q % 12 == 0, lambda q: q % 4 == 0) == 6)
    check("mod_inverse", mod_inverse(3, 11) == 4)
    check("prime_factors", prime_factors(360) == {2: 3, 3: 2, 5: 1})
    # 9 Python
    check("dict has no elements()", "elements" not in methods_of(dict) and "items" in methods_of(dict))
    # Tang 2
    check("lagrange x*y s.t. x+y=10", any(sol_[x] == 5 and sol_[y] == 5 for sol_ in lagrange("x*y", "x + y - 10")))
    zz, pp = z_test_mean(103, 100, 10, 100)
    check("z_test_mean", _close(zz, 3.0) and pp < 0.01)
    beta, r2 = ols([1, 2, 3, 4], [3, 5, 7, 9])
    check("ols", _close(beta[0], 1) and _close(beta[1], 2) and _close(r2, 1))

    print(f"\n{sum(ok)}/{len(ok)} vi du dung.")
    return all(ok)


if __name__ == "__main__":
    raise SystemExit(0 if run_examples() else 1)
