# ==============================================================================
# CHEATSHEET PYTHON HỖ TRỢ LÀM BÀI TEST MScFE - WORLDQUANT UNIVERSITY
# Hướng dẫn: Chạy ô khai báo thư viện trước khi dùng các hàm bên dưới.
# ==============================================================================

import math
import numpy as np
import scipy.stats as stats
from scipy import integrate
import sympy as sp

print("✅ Đã tải xong các thư viện! Bạn có thể bắt đầu sử dụng.")

# ==============================================================================
# DẠNG 1: ĐẠI SỐ TUYẾN TÍNH (LINEAR ALGEBRA - MA TRẬN)
# ==============================================================================

# --- 1.1. Khai báo Ma trận ---
# Thay đổi các con số trong ngoặc vuông [] theo đề bài
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])

B = np.array([[2, 0, 1], [1, 4, 2], [0, 0, 1]])

# --- 1.2. Tính Định thức (Determinant) ---
det_A = np.linalg.det(A)
print(f"1.2. Định thức det(A) = {det_A:.4f}")

# --- 1.3. Tìm Ma trận Nghịch đảo (Inverse Matrix) ---
try:
  inv_A = np.linalg.inv(A)
  print(f"1.3. Ma trận nghịch đảo A^-1:\n{inv_A}")
except np.linalg.LinAlgError:
  print("1.3. Ma trận KHÔNG có nghịch đảo (det = 0)")

# --- 1.4. Nhân 2 Ma trận (Matrix Multiplication A x B) ---
mult_AB = np.dot(A, B)  # Hoặc dùng A @ B
print(f"1.4. Tích A x B:\n{mult_AB}")

# --- 1.5. Chuyển vị Ma trận (Transpose) ---
transpose_A = A.T
print(f"1.5. Ma trận chuyển vị A^T:\n{transpose_A}")

# --- 1.6. Tìm Trị riêng (Eigenvalues) & Vector riêng (Eigenvectors) ---
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"1.6. Trị riêng (Eigenvalues): {eigenvalues}")
print(f"     Vector riêng (mỗi cột là một vector):\n{eigenvectors}")

# --- 1.7. Một số phép tính ma trận thường gặp ---
det_A_plus_B = np.linalg.det(A + B)
rank_A = np.linalg.matrix_rank(A)
trace_A = np.trace(A)
solution_Ax_b = np.linalg.solve(A, np.array([1, 2, 3]))
print(f"1.7. det(A + B) = {det_A_plus_B:.4f}; rank(A) = {rank_A}; trace(A) = {trace_A}")
print(f"     Nghiệm A x = [1, 2, 3]: {solution_Ax_b}")

# RREF và đa thức đặc trưng (SymPy giữ kết quả chính xác dạng ký hiệu)
A_symbolic = sp.Matrix(A)
rref_A, pivot_columns = A_symbolic.rref()
char_poly_A = A_symbolic.charpoly().as_expr()
print(f"     RREF(A) =\n{rref_A}; pivot columns = {pivot_columns}")
print(f"     Đa thức đặc trưng của A: {char_poly_A}")


# ==============================================================================
# DẠNG 2: GIẢI TÍCH (CALCULUS - ĐẠO HÀM, TÍCH PHÂN & CỰC TRỊ)
# ==============================================================================
x, y = sp.symbols('x y')  # Khai báo biến ký hiệu

# --- 2.1. Đạo hàm hàm 1 biến f(x) ---
# Ví dụ: f(x) = 3*x**3 - 5*x**2 + 2*x - 7  (Dấu ** là mũ)
f_x = 3 * x**3 - 5 * x**2 + 2 * x - 7
df_dx = sp.diff(f_x, x)
print(f"\n2.1. Đạo hàm f'(x) = {df_dx}")

# Tính giá trị đạo hàm tại x = 2
df_at_2 = df_dx.subs(x, 2)
print(f"     Giá trị f'(2) = {df_at_2}")

# --- 2.2. Đạo hàm riêng (Partial Derivatives) f(x, y) ---
# Ví dụ: f(x, y) = 3*x**2*y + y**3
f_xy = 3 * x**2 * y + y**3
df_dx_partial = sp.diff(f_xy, x)  # Đạo hàm theo x
df_dy_partial = sp.diff(f_xy, y)  # Đạo hàm theo y
print(f"2.2. Đạo hàm riêng theo x: {df_dx_partial}")
print(f"     Đạo hàm riêng theo y: {df_dy_partial}")

# --- 2.3. Tìm Cực trị (Điểm tới hạn nơi f'(x) = 0) ---
critical_points = sp.solve(df_dx, x)
print(f"2.3. Các điểm tới hạn (f'(x) = 0): {critical_points}")

# --- 2.4. Tích phân (Integration) ---
# Tích phân xác định của f(x) từ a đến b (Ví dụ: từ 0 đến 2)
integral_val = sp.integrate(f_x, (x, 0, 2))
print(f"2.4. Tích phân từ 0 đến 2 của f(x) = {integral_val}")

# --- 2.5. Giải phương trình, bất phương trình, giới hạn và chuỗi Taylor ---
equation_solutions = sp.solve(sp.Eq(sp.exp(2 * x), 7), x)
inequality_solution = sp.solve_univariate_inequality(x**2 - 5 * x + 6 < 0, x)
limit_value = sp.limit(sp.sin(x) / x, x, 0)
taylor_series = sp.series(sp.sqrt(x), x, 4, 3).removeO()
print(f"2.5. Nghiệm exp(2x) = 7: {equation_solutions}")
print(f"     Nghiệm x^2 - 5x + 6 < 0: {inequality_solution}")
print(f"     lim sin(x)/x khi x -> 0 = {limit_value}; Taylor sqrt(x) quanh 4: {taylor_series}")
print(f"     simplify / expand / factor: {sp.simplify((x**2 - 1)/(x - 1))} / {sp.expand((x + 1)**3)} / {sp.factor(x**2 - 1)}")
print(f"     log cơ số 2 của 8 = {math.log(8, 2)}")

# Hàm ngược: giải y = f(x) theo x; có thể có nhiều nhánh nghiệm
inverse_solutions = sp.solve(sp.Eq(y, x**2 + 1), x)
print(f"     Nghiệm x theo y khi y = x^2 + 1: {inverse_solutions}")

# --- 2.6. Tích phân bất định và tích phân số ---
antiderivative = sp.integrate(sp.exp(-x**2), x)
numeric_integral, integration_error = integrate.quad(lambda value: value**2, 0, 3)
print(f"2.6. Nguyen ham exp(-x^2): {antiderivative}")
print(f"     Tich phan so x^2 tu 0 den 3: {numeric_integral:.4f}")


# ==============================================================================
# DẠNG 3: XÁC SUẤT & THỐNG KÊ (PROBABILITY & STATISTICS)
# ==============================================================================

# --- 3.1. Thống kê mô tả (Mean, Variance, Std, Median) ---
data = [12, 15, 18, 22, 30, 25, 19]  # Nhập dãy số từ đề bài

mean_val = np.mean(data)
var_val = np.var(data, ddof=0)  # ddof=0 cho tổng thể, ddof=1 cho mẫu
std_val = np.std(data, ddof=0)
median_val = np.median(data)
mode_values, mode_counts = np.unique(data, return_counts=True)
mode_val = mode_values[mode_counts == mode_counts.max()].tolist()

print(f"\n3.1. Trung bình (Mean): {mean_val:.4f}")
print(f"     Phương sai (Variance): {var_val:.4f}")
print(f"     Độ lệch chuẩn (Std Dev): {std_val:.4f}")
print(f"     Trung vị (Median): {median_val}")
print(f"     Mode (các giá trị xuất hiện nhiều nhất): {mode_val}")

# --- 3.2. Định lý Bayes (Bayes' Theorem) ---
# P(A|B) = P(B|A) * P(A) / P(B)
def bayes_theorem(p_a, p_b_given_a, p_b):
  return (p_b_given_a * p_a) / p_b


# Ví dụ: P(A)=0.01, P(B|A)=0.9, P(B)=0.0585
print(f"3.2. Xác suất Bayes P(A|B) = {bayes_theorem(0.01, 0.9, 0.0585):.4f}")

# Dạng tương đương khi biết P(B|not A), thay vì biết trực tiếp P(B)
def bayes_from_complement(p_a, p_b_given_a, p_b_given_not_a):
  p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
  return bayes_theorem(p_a, p_b_given_a, p_b)


print(f"     Dạng dùng P(B|not A): {bayes_from_complement(0.01, 0.9, 0.05):.4f}")

# --- 3.3. Phân phối chuẩn (Normal Distribution - Z-score) ---
# Ví dụ: Trung bình mu = 100, độ lệch chuẩn sigma = 15. Tính P(X < 115)
mu = 100
sigma = 15
x_val = 115

z_score = (x_val - mu) / sigma
p_less_than_x = stats.norm.cdf(z_score)  # P(X < x)
p_greater_than_x = 1 - p_less_than_x  # P(X > x)

print(f"3.3. Z-score = {z_score:.2f}")
print(f"     Xác suất P(X < {x_val}) = {p_less_than_x:.4f}")
print(f"     Xác suất P(X > {x_val}) = {p_greater_than_x:.4f}")

# --- 3.4. Độ tương quan (Correlation) giữa 2 chuỗi số X và Y ---
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(X, Y)
print(f"3.4. Hệ số tương quan Corr(X, Y) = {corr_matrix[0, 1]:.4f}")


# ==============================================================================
# DẠNG 4: PHÂN PHỐI XÁC SUẤT RỜI RẠC VÀ LIÊN TỤC
# ==============================================================================

# Nhị thức: P(X = k), P(X <= k); Poisson; phân phối đều và mũ
n_trials, successes, probability = 10, 3, 0.25
binomial_exact = sp.binomial(n_trials, successes) * sp.Rational(1, 4)**successes * sp.Rational(3, 4)**(n_trials - successes)
binomial_cdf = stats.binom.cdf(successes, n_trials, probability)
poisson_probability = stats.poisson.pmf(2, mu=3)
poisson_cumulative = stats.poisson.cdf(2, mu=3)
uniform_probability = stats.uniform.cdf(0.75, loc=0, scale=1) - stats.uniform.cdf(0.25, loc=0, scale=1)
exponential_probability = stats.expon.cdf(2, scale=1 / 0.5)
print(f"\n4. P(Binomial = 3) exact = {binomial_exact}; P(X <= 3) = {binomial_cdf:.4f}")
print(f"   P(Poisson(3) = 2) = {poisson_probability:.4f}; P(Poisson(3) <= 2) = {poisson_cumulative:.4f}")
print(f"   P(0.25 <= Uniform(0,1) <= 0.75) = {uniform_probability:.4f}")
print(f"   P(Exponential(rate=0.5) <= 2) = {exponential_probability:.4f}")

# Kỳ vọng và phương sai của phân phối rời rạc
outcomes, outcome_probabilities = [1, 2, 3], [0.2, 0.5, 0.3]
expected_value = np.dot(outcomes, outcome_probabilities)
discrete_variance = np.dot((np.array(outcomes) - expected_value) ** 2, outcome_probabilities)
print(f"   E[X] = {expected_value:.4f}; Var(X) = {discrete_variance:.4f}")


# ==============================================================================
# DẠNG 5: KHOẢNG TIN CẬY, KIỂM ĐỊNH VÀ HỒI QUY
# ==============================================================================

# Khoảng tin cậy cho trung bình: z khi biết sigma, t khi ước lượng sigma từ mẫu
sample = np.array([12, 15, 18, 22, 30, 25, 19], dtype=float)
confidence = 0.95
z_critical = stats.norm.ppf(1 - (1 - confidence) / 2)
ci_z = (sample.mean() - z_critical * 4 / np.sqrt(len(sample)),
    sample.mean() + z_critical * 4 / np.sqrt(len(sample)))
ci_t = stats.t.interval(confidence, df=len(sample) - 1, loc=sample.mean(), scale=stats.sem(sample))
t_statistic, t_p_value = stats.ttest_1samp(sample, popmean=20)
print(f"\n5. 95% CI (biết sigma=4): {ci_z}")
print(f"   95% CI (t): {ci_t}; one-sample t-test so với mean=20: t={t_statistic:.4f}, p={t_p_value:.4f}")

# Hồi quy tuyến tính đơn và đa biến; OLS có intercept
regression = stats.linregress(X, Y)
design_matrix = np.column_stack((np.ones(len(X)), X))
ols_coefficients, *_ = np.linalg.lstsq(design_matrix, np.array(Y), rcond=None)
print(f"   Hồi quy đơn: slope={regression.slope:.4f}, intercept={regression.intercept:.4f}, r={regression.rvalue:.4f}")
print(f"   OLS [intercept, slope] = {ols_coefficients}")

# PCA: tỷ lệ phương sai giải thích và các vector thành phần chính
X_pca = np.array([[2, 0], [0, 2], [3, 1], [1, 3]], dtype=float)
X_centered = X_pca - X_pca.mean(axis=0)
pca_values, pca_vectors = np.linalg.eigh(np.cov(X_centered, rowvar=False))
pca_order = np.argsort(pca_values)[::-1]
explained_variance_ratio = pca_values[pca_order] / pca_values.sum()
principal_components = pca_vectors[:, pca_order]
print(f"   PCA explained variance ratio = {explained_variance_ratio}; components:\n{principal_components}")

# Kiểm định z cho trung bình khi biết sigma; hiệp phương sai của hai chuỗi
test_mean, null_mean, known_sigma = 103, 100, 10
z_statistic = (test_mean - null_mean) / (known_sigma / np.sqrt(100))
z_test_p_value = 2 * stats.norm.sf(abs(z_statistic))
covariance_xy = np.cov(X, Y, ddof=1)[0, 1]
normal_quantile = stats.norm.ppf(0.975)
print(f"   Z-test hai phía: z={z_statistic:.4f}, p={z_test_p_value:.4f}; z_0.975={normal_quantile:.4f}")
print(f"   Sample covariance Cov(X,Y) = {covariance_xy:.4f}")


# ==============================================================================
# DẠNG 6: PHƯƠNG TRÌNH VI PHÂN VÀ TỐI ƯU CÓ RÀNG BUỘC
# ==============================================================================

time = sp.symbols('time')
function_y = sp.Function('y')
ode_solution = sp.dsolve(sp.Eq(sp.diff(function_y(time), time), 3 * function_y(time)), function_y(time))
lagrange_multiplier = sp.symbols('lambda')
lagrange_solutions = sp.solve(
  [sp.diff(x * y - lagrange_multiplier * (x + y - 10), x),
   sp.diff(x * y - lagrange_multiplier * (x + y - 10), y), x + y - 10],
  [x, y, lagrange_multiplier], dict=True
)
print(f"\n6. ODE y' = 3y: {ode_solution}")
print(f"   Cực trị xy với ràng buộc x+y=10: {lagrange_solutions}")


# ==============================================================================
# DẠNG 7: TỔ HỢP, SỐ HỌC VÀ PHÉP TOÁN MODULO
# ==============================================================================

permutations = math.perm(5, 2)
combinations = math.comb(5, 2)
gcd_value = math.gcd(48, 18)
lcm_value = math.lcm(48, 18)
modular_power = pow(3, 4, 7)
modular_inverse = pow(3, -1, 11)
prime_factorization = sp.factorint(360)
positive_divisors = sp.divisors(12)
print(f"\n7. P(5,2)={permutations}; C(5,2)={combinations}; gcd(48,18)={gcd_value}; lcm(48,18)={lcm_value}")
print(f"   3^4 mod 7 = {modular_power}; nghịch đảo của 3 mod 11 = {modular_inverse}")
print(f"   Phân tích 360 thành thừa số nguyên tố: {prime_factorization}; ước dương của 12: {positive_divisors}")