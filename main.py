"""
Analisis OLS - Estimasi Daya Panel Surya
Data: NASA POWER Bontang, Kaltim (0.1333°N, 117.50°E)
Periode: Januari 2015 - Desember 2025 (132 data bulanan)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. BACA DATA CSV
# ============================================================
CSV_FILE = "POWER_Point_Monthly_20150101_20251231_000d13N_117d50E_UTC.csv"

df_raw = pd.read_csv(CSV_FILE, skiprows=19)
df_raw.columns = ['PARAMETER','YEAR','JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC','ANN']

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

PARAMS_NEEDED = ['ALLSKY_SFC_SW_DWN','T2M','WS10M','PS','CLOUD_AMT','IMERG_PRECTOT','ALLSKY_SFC_SW_DNI']

def pivot_param(df, param_name):
    sub = df[df['PARAMETER'] == param_name][['YEAR'] + MONTHS].copy()
    sub = sub.sort_values('YEAR')
    vals = []
    for _, row in sub.iterrows():
        for m in MONTHS:
            vals.append({'YEAR': int(row['YEAR']), 'MONTH': MONTHS.index(m)+1, param_name: float(row[m])})
    return pd.DataFrame(vals)

# Pivot semua parameter
df_list = []
for p in PARAMS_NEEDED:
    df_p = pivot_param(df_raw, p)
    df_list.append(df_p)

from functools import reduce
df_merged = reduce(lambda a, b: pd.merge(a, b, on=['YEAR','MONTH']), df_list)
df_merged = df_merged.sort_values(['YEAR','MONTH']).reset_index(drop=True)

# Ganti nilai -999 dengan NaN lalu drop
df_merged.replace(-999, np.nan, inplace=True)
df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)

print(f"Total data: {len(df_merged)} baris")
print(f"Periode: {int(df_merged['YEAR'].min())}-{int(df_merged.iloc[0]['MONTH']):02d} s/d {int(df_merged['YEAR'].max())}-{int(df_merged.iloc[-1]['MONTH']):02d}")
print(df_merged.describe().round(4))

# ============================================================
# 2. HITUNG VARIABEL Y
# ============================================================
eta_STC = 0.18
beta    = 0.004
T_STC   = 25.0

x1 = df_merged['ALLSKY_SFC_SW_DWN'].values
x2 = df_merged['T2M'].values
x3 = df_merged['WS10M'].values
x4 = df_merged['PS'].values
x5 = df_merged['CLOUD_AMT'].values
x6 = df_merged['IMERG_PRECTOT'].values
x7 = df_merged['ALLSKY_SFC_SW_DNI'].values

Y = x1 * eta_STC * (1 - beta * (x2 - T_STC))

# ============================================================
# 3. MATRIKS X (dengan intercept)
# ============================================================
n = len(Y)
X = np.column_stack([np.ones(n), x1, x2, x3, x4, x5, x6, x7])
feature_names = ['Intercept','x1(ALLSKY_DWN)','x2(T2M)','x3(WS10M)',
                 'x4(PS)','x5(CLOUD_AMT)','x6(PRECTOT)','x7(ALLSKY_DNI)']

# ============================================================
# 4. TRAIN/TEST SPLIT 80:20
# ============================================================
n_train = 106   # 2015-2023
n_test  = 26    # 2024-2025

X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# ============================================================
# 5. OLS - NORMAL EQUATIONS: β̂ = (XᵀX)⁻¹ XᵀY
# ============================================================
XtX     = X_train.T @ X_train
XtY     = X_train.T @ Y_train
beta_hat = np.linalg.solve(XtX, XtY)

Y_pred_train = X_train @ beta_hat
Y_pred_test  = X_test  @ beta_hat
Y_pred_all   = X       @ beta_hat

residuals_train = Y_train - Y_pred_train
residuals_test  = Y_test  - Y_pred_test
residuals_all   = Y       - Y_pred_all

# ============================================================
# 6. METRIK EVALUASI
# ============================================================
def compute_metrics(y_true, y_pred, label):
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"\n--- Metrik {label} ---")
    print(f"  MAE  : {mae:.6f}")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  R²   : {r2:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    return mae, mape, r2, rmse

metrics_train = compute_metrics(Y_train, Y_pred_train, "TRAINING")
metrics_test  = compute_metrics(Y_test,  Y_pred_test,  "TEST")
metrics_all   = compute_metrics(Y,       Y_pred_all,   "KESELURUHAN")

# ============================================================
# 7. UJI SIGNIFIKANSI KOEFISIEN (t-test)
# ============================================================
p_ols  = X_train.shape[1]        # jumlah parameter (termasuk intercept)
dof    = n_train - p_ols
mse    = np.sum(residuals_train**2) / dof
cov_beta = mse * np.linalg.inv(XtX)
se_beta  = np.sqrt(np.diag(cov_beta))
t_stat   = beta_hat / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))

print("\n\n{'='*65}")
print("KOEFISIEN MODEL OLS")
print(f"{'='*65}")
print(f"{'Parameter':<22} {'Koefisien':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
print("-"*65)
for i, name in enumerate(feature_names):
    sig = "***" if p_values[i]<0.001 else ("**" if p_values[i]<0.01 else ("*" if p_values[i]<0.05 else ""))
    print(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} {t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}")
print("Signifikansi: *** p<0.001  ** p<0.01  * p<0.05")

# ============================================================
# 8. VIF
# ============================================================
X_vars = X_train[:, 1:]   # tanpa intercept
k = X_vars.shape[1]
vif_values = []
for i in range(k):
    y_i  = X_vars[:, i]
    X_i  = np.delete(X_vars, i, axis=1)
    X_i  = np.column_stack([np.ones(n_train), X_i])
    b_i  = np.linalg.solve(X_i.T @ X_i, X_i.T @ y_i)
    y_hat_i = X_i @ b_i
    ss_res_i = np.sum((y_i - y_hat_i)**2)
    ss_tot_i = np.sum((y_i - np.mean(y_i))**2)
    r2_i = 1 - ss_res_i / ss_tot_i
    vif_i = 1 / (1 - r2_i) if r2_i < 1 else np.inf
    vif_values.append(vif_i)

print("\n\nVIF (Variance Inflation Factor)")
print("-"*40)
var_names = ['x1(ALLSKY_DWN)','x2(T2M)','x3(WS10M)','x4(PS)','x5(CLOUD_AMT)','x6(PRECTOT)','x7(ALLSKY_DNI)']
for name, vif in zip(var_names, vif_values):
    flag = " ⚠ TINGGI (>10)" if vif > 10 else (" ⚠ SEDANG (5-10)" if vif > 5 else "")
    print(f"  {name:<22} VIF = {vif:.4f}{flag}")

# ============================================================
# 9. DURBIN-WATSON
# ============================================================
res = residuals_train
dw = np.sum(np.diff(res)**2) / np.sum(res**2)
print(f"\n\nDurbin-Watson Statistic: {dw:.4f}")
if dw < 1.5:
    dw_interp = "Autokorelasi positif terdeteksi"
elif dw > 2.5:
    dw_interp = "Autokorelasi negatif terdeteksi"
else:
    dw_interp = "Tidak ada autokorelasi signifikan (DW ≈ 2)"
print(f"Interpretasi: {dw_interp}")

# ============================================================
# 10. SHAPIRO-WILK
# ============================================================
sw_stat, sw_pval = shapiro(residuals_all)
print(f"\n\nShapiro-Wilk Test (residual keseluruhan):")
print(f"  W = {sw_stat:.4f},  p-value = {sw_pval:.4f}")
if sw_pval < 0.05:
    sw_interp = "Residual TIDAK berdistribusi normal (p < 0.05) → catat sebagai batasan penelitian"
else:
    sw_interp = "Residual berdistribusi normal (p ≥ 0.05)"
print(f"  Interpretasi: {sw_interp}")

# ============================================================
# 11. VISUALISASI
# ============================================================
TIME_IDX = np.arange(1, n+1)

PARAM_LABELS = [
    ('ALLSKY_SFC_SW_DWN', 'ALLSKY DWN\n(kWh/m²/hari)', '#1f77b4'),
    ('T2M',               'T2M\n(°C)',                  '#ff7f0e'),
    ('WS10M',             'WS10M\n(m/s)',               '#2ca02c'),
    ('PS',                'PS\n(kPa)',                   '#d62728'),
    ('CLOUD_AMT',         'CLOUD AMT\n(%)',              '#9467bd'),
    ('IMERG_PRECTOT',     'PRECTOTCORR\n(mm/hari)',      '#8c564b'),
    ('ALLSKY_SFC_SW_DNI', 'ALLSKY DNI\n(kWh/m²/hari)', '#e377c2'),
]

# ---- A. HEATMAP KORELASI ----
data_corr = pd.DataFrame({
    'x1(ALLSKY_DWN)': x1, 'x2(T2M)': x2, 'x3(WS10M)': x3,
    'x4(PS)': x4, 'x5(CLOUD_AMT)': x5, 'x6(PRECTOT)': x6,
    'x7(ALLSKY_DNI)': x7, 'Y(Daya Est.)': Y
})
corr_matrix = data_corr.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_matrix.columns, fontsize=9)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)
ax.set_title('Heatmap Korelasi - 7 Parameter + Y\n(Bontang, Kaltim 2015–2025)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/A_heatmap_korelasi.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nGambar A disimpan: A_heatmap_korelasi.png")

# ---- B. DISTRIBUSI RESIDUAL + KURVA NORMAL ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(residuals_all, bins=30, density=True, color='steelblue', alpha=0.7,
        edgecolor='white', label='Histogram Residual')
mu, std = residuals_all.mean(), residuals_all.std()
x_norm = np.linspace(residuals_all.min(), residuals_all.max(), 300)
ax.plot(x_norm, norm.pdf(x_norm, mu, std), 'r-', linewidth=2.5, label='Kurva Normal Teoritis')
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Residual (kWh/m²/hari)', fontsize=11)
ax.set_ylabel('Densitas', fontsize=11)
ax.set_title(f'Distribusi Residual + Overlay Kurva Normal\nShapiro-Wilk: W={sw_stat:.4f}, p={sw_pval:.4f}  ({sw_interp[:40]}...)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/B_distribusi_residual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar B disimpan: B_distribusi_residual.png")

# ---- C. ACF PLOT RESIDUAL ----
def acf_manual(x, nlags=20):
    n_pts = len(x)
    x_dm  = x - x.mean()
    c0    = np.sum(x_dm**2) / n_pts
    acf_vals = [1.0]
    for lag in range(1, nlags+1):
        ck = np.sum(x_dm[:n_pts-lag] * x_dm[lag:]) / n_pts
        acf_vals.append(ck / c0)
    return np.array(acf_vals)

lags_range = 20
acf_vals = acf_manual(residuals_all, nlags=lags_range)
conf_bound = 1.96 / np.sqrt(n)

fig, ax = plt.subplots(figsize=(10, 5))
lags_x = np.arange(0, lags_range+1)
ax.bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
ax.axhline(conf_bound,  color='red', linestyle='--', linewidth=1.5, label='Batas 95% (±1.96/√n)')
ax.axhline(-conf_bound, color='red', linestyle='--', linewidth=1.5)
ax.fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Lag (bulan)', fontsize=11)
ax.set_ylabel('Autokorelasi', fontsize=11)
ax.set_title(f'ACF Plot Residual (Lags s/d 20)\nDurbin-Watson = {dw:.4f}  |  {dw_interp}', fontsize=11)
ax.set_xticks(lags_x)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/C_acf_residual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar C disimpan: C_acf_residual.png")

# ---- D. SCATTER AKTUAL vs PREDIKSI + PREDICTION INTERVAL 95% ----
# Prediction Interval
t_crit   = stats.t.ppf(0.975, dof)
leverage = np.diag(X @ np.linalg.inv(XtX) @ X.T)
pred_se  = np.sqrt(mse * (1 + leverage))
pi_lower = Y_pred_all - t_crit * pred_se
pi_upper = Y_pred_all + t_crit * pred_se

sorted_idx = np.argsort(Y_pred_all)
Y_pred_sorted = Y_pred_all[sorted_idx]
pi_lo_sorted  = pi_lower[sorted_idx]
pi_hi_sorted  = pi_upper[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 7))
ax.fill_between(Y_pred_sorted, pi_lo_sorted, pi_hi_sorted, alpha=0.2, color='green', label='Prediction Interval 95%')
ax.scatter(Y_pred_train, Y_train, c='royalblue', s=40, alpha=0.8, label=f'Training (n={n_train})', zorder=5)
ax.scatter(Y_pred_test,  Y_test,  c='tomato',    s=50, alpha=0.9, marker='s', label=f'Test (n={n_test})', zorder=5)
min_val = min(Y.min(), Y_pred_all.min()) - 0.05
max_val = max(Y.max(), Y_pred_all.max()) + 0.05
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='y = x')
ax.set_xlabel('Y Prediksi (kWh/m²/hari)', fontsize=11)
ax.set_ylabel('Y Aktual (kWh/m²/hari)', fontsize=11)
ax.set_title(f'Aktual vs Prediksi + Prediction Interval 95%\nR²_all={metrics_all[2]:.4f}  RMSE_all={metrics_all[3]:.4f}', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
plt.tight_layout()
plt.savefig('outputs/D_scatter_pred_interval.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar D disimpan: D_scatter_pred_interval.png")

# ---- E. TIME SERIES 7 PARAMETER ----
fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
data_vals = [x1, x2, x3, x4, x5, x6, x7]
for ax, (col, ylabel, color), data in zip(axes, PARAM_LABELS, data_vals):
    ax.plot(TIME_IDX, data, color=color, linewidth=1.2)
    ax.fill_between(TIME_IDX, data, alpha=0.15, color=color)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axvline(n_train+0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)
axes[-1].set_xlabel('Bulan ke- (Jan 2015 → Des 2025)', fontsize=10)
axes[-1].set_xticks(range(1, n+1, 12))
axes[-1].set_xticklabels([f'{y}\nJan' for y in range(2015, 2026)], fontsize=8)
fig.suptitle('Time Series 7 Parameter NASA POWER\nBontang, Kalimantan Timur (0.13°N, 117.50°E)  |  Jan 2015 – Des 2025',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/E_timeseries_7param.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar E disimpan: E_timeseries_7param.png")

# ---- F. RESIDUAL DIAGNOSTIC PLOTS (3-in-1) ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Q-Q Plot
(osm, osr), (slope, intercept, r) = stats.probplot(residuals_all, dist='norm')
axes[0].scatter(osm, osr, s=20, color='steelblue', alpha=0.7)
x_line = np.array([osm.min(), osm.max()])
axes[0].plot(x_line, slope*x_line + intercept, 'r-', linewidth=2)
axes[0].set_xlabel('Kuantil Teoritis', fontsize=10)
axes[0].set_ylabel('Kuantil Sampel', fontsize=10)
axes[0].set_title('Q-Q Plot Residual', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Residual vs Fitted
axes[1].scatter(Y_pred_all, residuals_all, s=20, alpha=0.7,
                c=['royalblue']*n_train + ['tomato']*n_test)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Nilai Fitted (Prediksi)', fontsize=10)
axes[1].set_ylabel('Residual', fontsize=10)
axes[1].set_title('Residual vs Fitted\n(Cek Heteroskedastisitas)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# ACF residual (ulang untuk panel ini)
axes[2].bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
axes[2].axhline(conf_bound,  color='red', linestyle='--', linewidth=1.5)
axes[2].axhline(-conf_bound, color='red', linestyle='--', linewidth=1.5)
axes[2].fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_xlabel('Lag', fontsize=10)
axes[2].set_ylabel('ACF', fontsize=10)
axes[2].set_title(f'ACF Residual\nDW = {dw:.4f}', fontsize=11)
axes[2].grid(True, alpha=0.3)

fig.suptitle('Residual Diagnostic Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/F_residual_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar F disimpan: F_residual_diagnostic.png")

# ---- G. TRAIN vs TEST OVERLAY SCATTER ----
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(Y_pred_train, Y_train, c='royalblue', s=50, alpha=0.8,
           marker='o', label=f'Training (n={n_train})', zorder=5)
ax.scatter(Y_pred_test,  Y_test,  c='tomato',    s=60, alpha=0.9,
           marker='s', label=f'Test (n={n_test})', zorder=5)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y = x')
ax.set_xlabel('Y Prediksi (kWh/m²/hari)', fontsize=12)
ax.set_ylabel('Y Aktual (kWh/m²/hari)', fontsize=12)
ax.set_title(f'Train vs Test Overlay Scatter\nR²_train={metrics_train[2]:.4f}  |  R²_test={metrics_test[2]:.4f}', fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/G_train_test_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gambar G disimpan: G_train_test_scatter.png")

# ============================================================
# 12. RINGKASAN AKHIR
# ============================================================
print("\n" + "="*65)
print("RINGKASAN MODEL OLS")
print("="*65)
print("Model: Ŷ = α + a·x1 + b·x2 + c·x3 + d·x4 + e·x5 + f·x6 + g·x7")
print(f"\nα  (Intercept)   = {beta_hat[0]:>12.6f}")
labels_short = ['a(x1)','b(x2)','c(x3)','d(x4)','e(x5)','f(x6)','g(x7)']
for i, lab in enumerate(labels_short):
    print(f"   {lab:<16} = {beta_hat[i+1]:>12.6f}")

print("\nMetrik Evaluasi:")
headers = f"{'Metrik':<8} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}"
print(headers)
print("-"*50)
mnames = ['MAE','MAPE(%)','R²','RMSE']
for i, mn in enumerate(mnames):
    val_train = metrics_train[i]
    val_test  = metrics_test[i]
    val_all   = metrics_all[i]
    fmt = f"{mn:<8} {val_train:>12.6f} {val_test:>12.6f} {val_all:>14.6f}"
    print(fmt)

# ============================================================
# SIMPAN HASIL KE FILE TXT
# ============================================================
with open('outputs/hasil_analisis.txt', 'w') as f:
    f.write("="*65 + "\n")
    f.write("ANALISIS OLS - ESTIMASI DAYA PANEL SURYA\n")
    f.write("="*65 + "\n\n")
    
    f.write(f"Total data: {len(df_merged)} baris\n")
    f.write(f"Periode: {int(df_merged['YEAR'].min())}-{int(df_merged.iloc[0]['MONTH']):02d} s/d {int(df_merged['YEAR'].max())}-{int(df_merged.iloc[-1]['MONTH']):02d}\n\n")
    
    f.write("KOEFISIEN MODEL OLS\n")
    f.write("-"*65 + "\n")
    for i, name in enumerate(feature_names):
        sig = "***" if p_values[i]<0.001 else ("**" if p_values[i]<0.01 else ("*" if p_values[i]<0.05 else ""))
        f.write(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} {t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}\n")
    
    f.write("\n\nMETRIK EVALUASI\n")
    f.write("-"*50 + "\n")
    f.write(f"{'Metrik':<8} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}\n")
    f.write("-"*50 + "\n")
    mnames = ['MAE','MAPE(%)','R²','RMSE']
    for i, mn in enumerate(mnames):
        f.write(f"{mn:<8} {metrics_train[i]:>12.6f} {metrics_test[i]:>12.6f} {metrics_all[i]:>14.6f}\n")
    
    f.write("\n\nDIAGNOSTIC TESTS\n")
    f.write("-"*50 + "\n")
    f.write(f"Durbin-Watson: {dw:.4f} - {dw_interp}\n")
    f.write(f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_pval:.4f} - {sw_interp.replace('→', '->')}\n")
    
    f.write("\nVIF (Variance Inflation Factor)\n")
    f.write("-"*40 + "\n")
    for name, vif in zip(var_names, vif_values):
        flag = " (TINGGI >10)" if vif > 10 else (" (SEDANG 5-10)" if vif > 5 else "")
        f.write(f"{name:<22} VIF = {vif:.4f}{flag}\n")

print("\n✓ Hasil analisis disimpan ke: outputs/hasil_analisis.txt")

print("\nSemua gambar tersimpan di outputs/ (300 dpi)")
print("="*65)