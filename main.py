"""
=======================================================================
ANALISIS MODEL PREDIKSI DAYA ENERGI SURYA - 7 PARAMETER
Bontang, Kalimantan Timur (0.1333°N, 117.50°E)
Data: NASA POWER Monthly 2015-2024
Metode: OLS Aljabar Linear (Normal Equations)

Bagian 2.4 - Kode Python Lengkap
Sumber: Output langsung dari eksekusi kode Python menggunakan
        data NASA POWER CSV aktual (2015–2024)

7 Variabel Bebas:
  x1 = ALLSKY_SFC_SW_DWN  (Irradiansi global, kWh/m²/hari)
  x2 = T2M                (Suhu 2 meter, °C)
  x3 = WS10M              (Kecepatan angin 10m, m/s)
  x4 = PS                 (Tekanan permukaan, kPa)
  x5 = CLOUD_AMT          (Tutupan awan, %)
  x6 = IMERG_PRECTOT      (Curah hujan, mm/hari)
  x7 = ALLSKY_SFC_SW_DNI  (Irradiansi langsung, kWh/m²/hari)

Target Y: Estimasi daya panel surya (kWh/m²/hari)
  Y = x1 × η_STC × [1 − β(T2M − T_STC)]
=======================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, norm
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

# Buat folder output
os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. BACA DATA CSV NASA POWER
# ============================================================
CSV_FILE = "POWER_Point_Monthly_20150101_20251231_000d13N_117d50E_UTC.csv"

df_raw = pd.read_csv(CSV_FILE, skiprows=19)
df_raw.columns = ['PARAMETER','YEAR','JAN','FEB','MAR','APR','MAY','JUN',
                   'JUL','AUG','SEP','OCT','NOV','DEC','ANN']

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
PARAMS_NEEDED = ['ALLSKY_SFC_SW_DWN','T2M','WS10M','PS','CLOUD_AMT',
                 'IMERG_PRECTOT','ALLSKY_SFC_SW_DNI']

def pivot_param(df, param_name):
    sub = df[df['PARAMETER'] == param_name][['YEAR'] + MONTHS].copy()
    sub = sub.sort_values('YEAR')
    vals = []
    for _, row in sub.iterrows():
        for m in MONTHS:
            vals.append({'YEAR': int(row['YEAR']),
                         'MONTH': MONTHS.index(m)+1,
                         param_name: float(row[m])})
    return pd.DataFrame(vals)

df_list = [pivot_param(df_raw, p) for p in PARAMS_NEEDED]
df_merged = reduce(lambda a, b: pd.merge(a, b, on=['YEAR','MONTH']), df_list)
df_merged = df_merged.sort_values(['YEAR','MONTH']).reset_index(drop=True)
df_merged.replace(-999, np.nan, inplace=True)
df_merged.dropna(inplace=True)

# Filter hanya 2015-2024 (120 bulan)
df_merged = df_merged[df_merged['YEAR'] <= 2024].reset_index(drop=True)

print(f"Total data: {len(df_merged)} baris")
print(f"Periode: {int(df_merged['YEAR'].min())}-{int(df_merged.iloc[0]['MONTH']):02d} "
      f"s/d {int(df_merged['YEAR'].max())}-{int(df_merged.iloc[-1]['MONTH']):02d}")

# ============================================================
# 2. DEFINISI VARIABEL & HITUNG Y
# ============================================================
eta_STC = 0.18     # Efisiensi STC panel surya
beta_T  = 0.004    # Koefisien temperatur
T_STC   = 25.0     # Suhu referensi STC (°C)

x1 = df_merged['ALLSKY_SFC_SW_DWN'].values    # kWh/m²/hari
x2 = df_merged['T2M'].values                  # °C
x3 = df_merged['WS10M'].values                # m/s
x4 = df_merged['PS'].values                   # kPa
x5 = df_merged['CLOUD_AMT'].values            # %
x6 = df_merged['IMERG_PRECTOT'].values        # mm/hari
x7 = df_merged['ALLSKY_SFC_SW_DNI'].values    # kWh/m²/hari

# Target: Daya panel surya dengan koreksi temperatur
Y = x1 * eta_STC * (1 - beta_T * (x2 - T_STC))

# ============================================================
# 3. MATRIKS X (intercept + 7 fitur)
# ============================================================
n = len(Y)
X = np.column_stack([np.ones(n), x1, x2, x3, x4, x5, x6, x7])
feature_names = ['Intercept', 'x1(ALLSKY_DWN)', 'x2(T2M)', 'x3(WS10M)',
                 'x4(PS)', 'x5(CLOUD_AMT)', 'x6(PRECTOT)', 'x7(ALLSKY_DNI)']
var_names = ['x1(ALLSKY_DWN)','x2(T2M)','x3(WS10M)',
             'x4(PS)','x5(CLOUD_AMT)','x6(PRECTOT)','x7(ALLSKY_DNI)']

# ============================================================
# 4. TRAIN/TEST SPLIT  →  96 train (2015-2022) | 24 test (2023-2024)
# ============================================================
n_train = 96   # Jan 2015 – Des 2022
n_test  = 24   # Jan 2023 – Des 2024
print(f"Train: {n_train} bulan  |  Test: {n_test} bulan  |  Total: {n} bulan")

X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# ============================================================
# 5. OLS ALJABAR LINEAR: β̂ = (XᵀX)⁻¹ XᵀY
# ============================================================
XtX      = X_train.T @ X_train
XtY      = X_train.T @ Y_train
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
def compute_metrics(y_true, y_pred, label=''):
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    if label:
        print(f"\n--- Metrik {label} ---")
        print(f"  MAE  : {mae:.8f}")
        print(f"  MAPE : {mape:.7f}%")
        print(f"  R²   : {r2:.8f}")
        print(f"  RMSE : {rmse:.8f}")
    return mae, mape, r2, rmse

metrics_train = compute_metrics(Y_train, Y_pred_train, "TRAINING")
metrics_test  = compute_metrics(Y_test,  Y_pred_test,  "TEST (2023-2024)")
metrics_all   = compute_metrics(Y,       Y_pred_all,   "KESELURUHAN")

# ============================================================
# 7. UJI SIGNIFIKANSI KOEFISIEN (t-test)
# ============================================================
p_ols    = X_train.shape[1]
dof      = n_train - p_ols
mse      = np.sum(residuals_train**2) / dof
cov_beta = mse * np.linalg.inv(XtX)
se_beta  = np.sqrt(np.diag(cov_beta))
t_stat   = beta_hat / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))

print(f"\n{'='*70}")
print("KOEFISIEN MODEL OLS")
print(f"{'='*70}")
print(f"{'Parameter':<22} {'Koefisien':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
print("-"*70)
for i, name in enumerate(feature_names):
    sig = "***" if p_values[i]<0.001 else ("**" if p_values[i]<0.01 else ("*" if p_values[i]<0.05 else "n.s."))
    print(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} {t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}")
print("Signifikansi: *** p<0.001  ** p<0.01  * p<0.05  n.s. = tidak signifikan")

# ============================================================
# 8. VIF (Variance Inflation Factor)
# ============================================================
X_vars = X_train[:, 1:]
k = X_vars.shape[1]
vif_values = []
for i in range(k):
    y_i   = X_vars[:, i]
    X_i   = np.column_stack([np.ones(n_train), np.delete(X_vars, i, axis=1)])
    b_i   = np.linalg.solve(X_i.T @ X_i, X_i.T @ y_i)
    y_hat = X_i @ b_i
    ss_res_i = np.sum((y_i - y_hat)**2)
    ss_tot_i = np.sum((y_i - np.mean(y_i))**2)
    r2_i  = 1 - ss_res_i / ss_tot_i
    vif_values.append(1 / (1 - r2_i) if r2_i < 1 else np.inf)

print("\nVIF (Variance Inflation Factor)")
print("-"*45)
for name, vif in zip(var_names, vif_values):
    flag = " ⚠ TINGGI (>10)" if vif > 10 else (" ⚠ SEDANG (5-10)" if vif > 5 else "")
    print(f"  {name:<22} VIF = {vif:.4f}{flag}")

# ============================================================
# 9. DURBIN-WATSON & SHAPIRO-WILK
# ============================================================
res = residuals_train
dw  = np.sum(np.diff(res)**2) / np.sum(res**2)
if dw < 1.5:
    dw_interp = "Autokorelasi positif terdeteksi"
elif dw > 2.5:
    dw_interp = "Autokorelasi negatif terdeteksi"
else:
    dw_interp = "Tidak ada autokorelasi signifikan (DW ≈ 2)"

sw_stat, sw_pval = shapiro(residuals_all)
if sw_pval < 0.05:
    sw_interp = "Residual TIDAK berdistribusi normal (p < 0.05)"
else:
    sw_interp = "Residual berdistribusi normal (p ≥ 0.05)"

skew_val = stats.skew(residuals_all)
kurt_val = stats.kurtosis(residuals_all, fisher=False)

print(f"\nDurbin-Watson: {dw:.4f}  →  {dw_interp}")
print(f"Shapiro-Wilk:  W={sw_stat:.4f}, p={sw_pval:.4f}  →  {sw_interp}")
print(f"Skewness={skew_val:.4f},  Kurtosis={kurt_val:.4f}")

# ============================================================
# 10. PREDICTION INTERVAL 95%
# ============================================================
t_crit   = stats.t.ppf(0.975, dof)
leverage = np.diag(X @ np.linalg.inv(XtX) @ X.T)
pred_se  = np.sqrt(mse * (1 + leverage))
pi_lower = Y_pred_all - t_crit * pred_se
pi_upper = Y_pred_all + t_crit * pred_se

sorted_idx     = np.argsort(Y_pred_all)
Y_pred_sorted  = Y_pred_all[sorted_idx]
pi_lo_sorted   = pi_lower[sorted_idx]
pi_hi_sorted   = pi_upper[sorted_idx]

# ============================================================
# 11. ACF MANUAL
# ============================================================
def acf_manual(x, nlags=20):
    n_pts = len(x)
    x_dm  = x - x.mean()
    c0    = np.sum(x_dm**2) / n_pts
    acf_v = [1.0]
    for lag in range(1, nlags+1):
        ck = np.sum(x_dm[:n_pts-lag] * x_dm[lag:]) / n_pts
        acf_v.append(ck / c0)
    return np.array(acf_v)

lags_range = 20
acf_vals   = acf_manual(residuals_all, nlags=lags_range)
conf_bound = 1.96 / np.sqrt(n)
lags_x     = np.arange(0, lags_range+1)

# ============================================================
# HELPERS
# ============================================================
TIME_IDX   = np.arange(1, n+1)
min_val    = min(Y.min(), Y_pred_all.min()) - 0.02
max_val    = max(Y.max(), Y_pred_all.max()) + 0.02

PARAM_LABELS = [
    ('ALLSKY_SFC_SW_DWN', 'ALLSKY DWN\n(kWh/m²/hari)', '#1f77b4'),
    ('T2M',               'T2M\n(°C)',                  '#ff7f0e'),
    ('WS10M',             'WS10M\n(m/s)',               '#2ca02c'),
    ('PS',                'PS\n(kPa)',                   '#d62728'),
    ('CLOUD_AMT',         'CLOUD AMT\n(%)',              '#9467bd'),
    ('IMERG_PRECTOT',     'PRECTOTCORR\n(mm/hari)',      '#8c564b'),
    ('ALLSKY_SFC_SW_DNI', 'ALLSKY DNI\n(kWh/m²/hari)', '#e377c2'),
]
data_vals = [x1, x2, x3, x4, x5, x6, x7]

# ============================================================
# GAMBAR 1 — PANEL UTAMA 4-in-1 (sesuai PNG referensi)
# ============================================================
fig = plt.figure(figsize=(17, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

idx_train = np.arange(n_train)
idx_test  = np.arange(n_train, n)

# --- (a) Time-series Aktual vs Prediksi ---
ax1.plot(idx_train, Y_train,       color='purple',  lw=1.3,  ls='-',  label='Y Aktual (Train)')
ax1.plot(idx_train, Y_pred_train,  color='magenta', lw=1.0,  ls='--', label='Y Prediksi (Train)')
ax1.plot(idx_test,  Y_test,        color='black',   lw=1.3,  ls='-',  label='Y Aktual (Test 2023–2024)')
ax1.plot(idx_test,  Y_pred_test,   color='gray',    lw=1.0,  ls='--', label='Y Prediksi (Test 2023–2024)')
ax1.axvline(n_train-0.5, color='orange', lw=1.5, ls='--', label='Batas Train/Test')
ax1.set_xlabel('Indeks Sampel Bulanan (Jan 2015 – Des 2024)', fontsize=9)
ax1.set_ylabel('Daya Prediksi (kWh/m²/hari)', fontsize=9)
ax1.set_title('(a) Daya Aktual vs Prediksi OLS – 120 Sampel Bulanan', fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, n+1)

# Kotak info koefisien + evaluasi (pojok kiri bawah)
coef_lines = "Koefisien OLS:\n"
coef_lines += f"  a (Intercept)    = {beta_hat[0]:.6f}\n"
param_letters = list('bcdefgh')
short_names   = ['ALLSKY','T2M','WS10M','PS','CLOUD','PREC','DNI']
for i in range(7):
    coef_lines += f"  {param_letters[i]} ({short_names[i]:<6}) = {beta_hat[i+1]:>10.6f}\n"
eval_lines  = (f"\nEvaluasi Test (2023–2024):\n"
               f"  MAE  = {metrics_test[0]:.8f}\n"
               f"  MAPE = {metrics_test[1]:.7f}%\n"
               f"  R²   = {metrics_test[2]:.8f}\n"
               f"  RMSE = {metrics_test[3]:.8f}")
asumsi_lines = (f"\n\nUji Asumsi:\n"
                f"  Durbin-Watson = {dw:.4f}\n"
                f"  Shapiro-Wilk p = {sw_pval:.4f}")
ax1.text(0.01, 0.01, coef_lines + eval_lines + asumsi_lines,
         transform=ax1.transAxes, fontsize=5.2, va='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.88))

# --- (b) Scatter Aktual vs Prediksi ---
ax2.scatter(Y_train, Y_pred_train, c='blue',  s=22, alpha=0.8, label=f'Train (n={n_train})', zorder=4)
ax2.scatter(Y_test,  Y_pred_test,  c='green', s=25, marker='D', alpha=0.9,
            label=f'Test (n={n_test})', zorder=4)
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Garis Ideal (y=x)')
ax2.set_xlabel('Y Aktual (kWh/m²/hari)', fontsize=9)
ax2.set_ylabel('Y Prediksi (kWh/m²/hari)', fontsize=9)
ax2.set_title(f'(b) Scatter Plot: Aktual vs Prediksi\nR² (Test) = {metrics_test[2]:.8f}',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)

# --- (c) Plot Residual ---
ax3.plot(idx_train, residuals_train, color='blue',  lw=1.0, label='Residual Train', alpha=0.85)
ax3.plot(idx_test,  residuals_test,  color='green', lw=1.2, marker='o', ms=3,
         label='Residual Test', alpha=0.9)
ax3.axhline(0,          color='red',    lw=1.0, ls='--')
ax3.axvline(n_train-0.5, color='orange', lw=1.5, ls='--', label='Batas Train/Test')
ax3.set_xlabel('Indeks Sampel Bulanan', fontsize=9)
ax3.set_ylabel('Residual (Y – Ŷ)', fontsize=9)
ax3.set_title('(c) Plot Residual – Distribusi Error Model OLS', fontsize=10, fontweight='bold')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, n+1)

# --- (d) Distribusi Residual ---
ax4.hist(residuals_all, bins=50, density=True, color='steelblue', alpha=0.70,
         edgecolor='white', label='Histogram')
mu_r, sd_r = residuals_all.mean(), residuals_all.std()
x_norm = np.linspace(residuals_all.min(), residuals_all.max(), 300)
ax4.plot(x_norm, norm.pdf(x_norm, mu_r, sd_r), 'r-', lw=2.2, label='Kurva Normal')
ax4.set_xlabel('Residual', fontsize=9)
ax4.set_ylabel('Densitas', fontsize=9)
ax4.set_title(f'(d) Distribusi Residual\nSkewness={skew_val:.4f}, Kurtosis={kurt_val:.4f}',
              fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

fig.suptitle(
    'Analisis Model Prediksi Daya Energi Surya – 7 Parameter, Bontang 2015–2024\n'
    '(Data NASA POWER, OLS Aljabar Linear)',
    fontsize=12, fontweight='bold', y=1.005
)
plt.tight_layout()
plt.savefig('outputs/Gambar1_Panel_Utama_4in1.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close()
print("\n✅ Gambar 1 disimpan: outputs/Gambar1_Panel_Utama_4in1.png")

# ============================================================
# GAMBAR 2 — HEATMAP KORELASI
# ============================================================
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
        val   = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)
ax.set_title('Heatmap Korelasi – 7 Parameter + Y\n(Bontang, Kaltim 2015–2024)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar2_Heatmap_Korelasi.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 2 disimpan: outputs/Gambar2_Heatmap_Korelasi.png")

# ============================================================
# GAMBAR 3 — TIME SERIES 7 PARAMETER
# ============================================================
fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
for ax, (col, ylabel, color), data in zip(axes, PARAM_LABELS, data_vals):
    ax.plot(TIME_IDX, data, color=color, linewidth=1.2)
    ax.fill_between(TIME_IDX, data, alpha=0.15, color=color)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axvline(n_train+0.5, color='gray', linestyle='--', lw=0.8, alpha=0.7,
               label='Batas Train/Test')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)
axes[-1].set_xlabel('Bulan ke- (Jan 2015 → Des 2024)', fontsize=10)
axes[-1].set_xticks(range(1, n+1, 12))
axes[-1].set_xticklabels([f'{y}\nJan' for y in range(2015, 2025)], fontsize=8)
axes[0].legend(fontsize=8, loc='upper right')
fig.suptitle('Time Series 7 Parameter NASA POWER\n'
             'Bontang, Kalimantan Timur (0.13°N, 117.50°E) | Jan 2015 – Des 2024',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/Gambar3_TimeSeries_7Param.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 3 disimpan: outputs/Gambar3_TimeSeries_7Param.png")

# ============================================================
# GAMBAR 4 — SCATTER + PREDICTION INTERVAL 95%
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.fill_between(Y_pred_sorted, pi_lo_sorted, pi_hi_sorted,
                alpha=0.20, color='green', label='Prediction Interval 95%')
ax.scatter(Y_pred_train, Y_train, c='royalblue', s=45, alpha=0.8,
           label=f'Training (n={n_train})', zorder=5)
ax.scatter(Y_pred_test,  Y_test,  c='tomato',    s=55, alpha=0.9, marker='s',
           label=f'Test (n={n_test})', zorder=5)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x')
ax.set_xlabel('Y Prediksi (kWh/m²/hari)', fontsize=11)
ax.set_ylabel('Y Aktual (kWh/m²/hari)',   fontsize=11)
ax.set_title(f'Aktual vs Prediksi + Prediction Interval 95%\n'
             f'R²_all={metrics_all[2]:.4f}  |  RMSE_all={metrics_all[3]:.4f}', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/Gambar4_Scatter_PredInterval.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 4 disimpan: outputs/Gambar4_Scatter_PredInterval.png")

# ============================================================
# GAMBAR 5 — DISTRIBUSI RESIDUAL + KURVA NORMAL
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(residuals_all, bins=30, density=True, color='steelblue', alpha=0.70,
        edgecolor='white', label='Histogram Residual')
ax.plot(x_norm, norm.pdf(x_norm, mu_r, sd_r), 'r-', lw=2.5,
        label='Kurva Normal Teoritis')
ax.axvline(0, color='gray', linestyle='--', lw=1)
ax.set_xlabel('Residual (kWh/m²/hari)', fontsize=11)
ax.set_ylabel('Densitas', fontsize=11)
ax.set_title(f'Distribusi Residual + Overlay Kurva Normal\n'
             f'Shapiro-Wilk: W={sw_stat:.4f}, p={sw_pval:.4f}  |  {sw_interp}',
             fontsize=10)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/Gambar5_Distribusi_Residual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 5 disimpan: outputs/Gambar5_Distribusi_Residual.png")

# ============================================================
# GAMBAR 6 — ACF PLOT RESIDUAL
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
ax.axhline( conf_bound, color='red', linestyle='--', lw=1.5, label='Batas 95% (±1.96/√n)')
ax.axhline(-conf_bound, color='red', linestyle='--', lw=1.5)
ax.fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Lag (bulan)', fontsize=11)
ax.set_ylabel('Autokorelasi', fontsize=11)
ax.set_title(f'ACF Plot Residual (Lags s/d 20)\n'
             f'Durbin-Watson = {dw:.4f}  |  {dw_interp}', fontsize=11)
ax.set_xticks(lags_x)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/Gambar6_ACF_Residual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 6 disimpan: outputs/Gambar6_ACF_Residual.png")

# ============================================================
# GAMBAR 7 — RESIDUAL DIAGNOSTIC 3-in-1
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Q-Q Plot
(osm, osr), (slope, intercept_qq, r_qq) = stats.probplot(residuals_all, dist='norm')
axes[0].scatter(osm, osr, s=20, color='steelblue', alpha=0.7)
x_line = np.array([osm.min(), osm.max()])
axes[0].plot(x_line, slope*x_line + intercept_qq, 'r-', lw=2)
axes[0].set_xlabel('Kuantil Teoritis', fontsize=10)
axes[0].set_ylabel('Kuantil Sampel', fontsize=10)
axes[0].set_title('Q-Q Plot Residual', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Residual vs Fitted
colors_rf = ['royalblue']*n_train + ['tomato']*n_test
axes[1].scatter(Y_pred_all, residuals_all, s=20, alpha=0.7, c=colors_rf)
axes[1].axhline(0, color='red', linestyle='--', lw=1.5)
axes[1].set_xlabel('Nilai Fitted (Prediksi)', fontsize=10)
axes[1].set_ylabel('Residual', fontsize=10)
axes[1].set_title('Residual vs Fitted\n(Cek Heteroskedastisitas)', fontsize=11)
axes[1].grid(True, alpha=0.3)

# ACF ringkas
axes[2].bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
axes[2].axhline( conf_bound, color='red', linestyle='--', lw=1.5)
axes[2].axhline(-conf_bound, color='red', linestyle='--', lw=1.5)
axes[2].fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
axes[2].axhline(0, color='black', lw=0.8)
axes[2].set_xlabel('Lag', fontsize=10)
axes[2].set_ylabel('ACF', fontsize=10)
axes[2].set_title(f'ACF Residual\nDW = {dw:.4f}', fontsize=11)
axes[2].grid(True, alpha=0.3)

fig.suptitle('Residual Diagnostic Plots – 7 Parameter OLS', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar7_Residual_Diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 7 disimpan: outputs/Gambar7_Residual_Diagnostic.png")

# ============================================================
# GAMBAR 8 — TRAIN vs TEST SCATTER OVERLAY
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(Y_pred_train, Y_train, c='royalblue', s=50, alpha=0.8,
           marker='o', label=f'Training (n={n_train})', zorder=5)
ax.scatter(Y_pred_test,  Y_test,  c='tomato',    s=60, alpha=0.9,
           marker='s', label=f'Test (n={n_test})', zorder=5)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y = x')
ax.set_xlabel('Y Prediksi (kWh/m²/hari)', fontsize=12)
ax.set_ylabel('Y Aktual (kWh/m²/hari)',   fontsize=12)
ax.set_title(f'Train vs Test Overlay Scatter\n'
             f'R²_train={metrics_train[2]:.4f}  |  R²_test={metrics_test[2]:.4f}', fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/Gambar8_TrainTest_Scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar 8 disimpan: outputs/Gambar8_TrainTest_Scatter.png")

# ============================================================
# 12. SIMPAN HASIL ANALISIS KE FILE TXT
# ============================================================
with open('outputs/hasil_analisis_lengkap.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ANALISIS OLS – ESTIMASI DAYA PANEL SURYA\n")
    f.write("7 Parameter, Bontang, Kaltim | NASA POWER 2015–2024\n")
    f.write("="*70 + "\n\n")

    f.write(f"Total data: {n} baris  |  Train: {n_train}  |  Test: {n_test}\n\n")

    f.write("MODEL: Ŷ = α + a·x1 + b·x2 + c·x3 + d·x4 + e·x5 + f·x6 + g·x7\n\n")

    f.write("KOEFISIEN OLS\n" + "-"*70 + "\n")
    f.write(f"{'Parameter':<22} {'Koefisien':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}\n")
    f.write("-"*70 + "\n")
    for i, name in enumerate(feature_names):
        sig = "***" if p_values[i]<0.001 else ("**" if p_values[i]<0.01 else ("*" if p_values[i]<0.05 else "n.s."))
        f.write(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} "
                f"{t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}\n")
    f.write("Signifikansi: *** p<0.001  ** p<0.01  * p<0.05  n.s.=tidak signifikan\n\n")

    f.write("VIF\n" + "-"*40 + "\n")
    for name, vif in zip(var_names, vif_values):
        flag = " (TINGGI >10)" if vif > 10 else (" (SEDANG 5-10)" if vif > 5 else "")
        f.write(f"  {name:<22} VIF = {vif:.4f}{flag}\n")

    f.write("\nMETRIK EVALUASI\n" + "-"*55 + "\n")
    f.write(f"{'Metrik':<10} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}\n")
    f.write("-"*55 + "\n")
    for i, mn in enumerate(['MAE','MAPE(%)','R²','RMSE']):
        f.write(f"{mn:<10} {metrics_train[i]:>12.6f} {metrics_test[i]:>12.6f} "
                f"{metrics_all[i]:>14.6f}\n")

    f.write("\nDIAGNOSTIC TESTS\n" + "-"*50 + "\n")
    f.write(f"Durbin-Watson  : {dw:.4f}  →  {dw_interp}\n")
    f.write(f"Shapiro-Wilk   : W={sw_stat:.4f}, p={sw_pval:.4f}  →  {sw_interp}\n")
    f.write(f"Skewness       : {skew_val:.4f}\n")
    f.write(f"Kurtosis       : {kurt_val:.4f}\n")

print("\n✅ Hasil analisis: outputs/hasil_analisis_lengkap.txt")

# ============================================================
# 13. RINGKASAN KONSOL
# ============================================================
print(f"\n{'='*65}")
print("RINGKASAN MODEL OLS – 7 PARAMETER")
print(f"{'='*65}")
print(f"Model: Ŷ = α + a·x1 + b·x2 + c·x3 + d·x4 + e·x5 + f·x6 + g·x7")
print(f"\n  α  (Intercept)   = {beta_hat[0]:>12.6f}")
letters = list('abcdefg')
for i, (lbl, lt) in enumerate(zip(var_names, letters)):
    print(f"  {lt}  {lbl:<22} = {beta_hat[i+1]:>12.6f}")
print(f"\n{'Metrik':<10} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}")
print("-"*50)
for i, mn in enumerate(['MAE','MAPE(%)','R²','RMSE']):
    print(f"{mn:<10} {metrics_train[i]:>12.6f} {metrics_test[i]:>12.6f} {metrics_all[i]:>14.6f}")
print(f"\nDurbin-Watson  = {dw:.4f}  |  Shapiro-Wilk p = {sw_pval:.4f}")
print(f"Skewness = {skew_val:.4f}  |  Kurtosis = {kurt_val:.4f}")
print(f"\n{'='*65}")
print("SEMUA OUTPUT TERSIMPAN DI FOLDER: outputs/")
print(f"{'='*65}")
print("  Gambar1_Panel_Utama_4in1.png       ← Gambar utama jurnal (4 panel)")
print("  Gambar2_Heatmap_Korelasi.png")
print("  Gambar3_TimeSeries_7Param.png")
print("  Gambar4_Scatter_PredInterval.png")
print("  Gambar5_Distribusi_Residual.png")
print("  Gambar6_ACF_Residual.png")
print("  Gambar7_Residual_Diagnostic.png")
print("  Gambar8_TrainTest_Scatter.png")
print("  hasil_analisis_lengkap.txt")