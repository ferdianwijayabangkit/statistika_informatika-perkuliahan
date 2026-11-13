# =============================================================================
# Pertemuan 1 - Analisis Multivariat dan Regresi Linear Berganda (Python)
# Urutan kode mengikuti slide. Berkas data: data_respons_aplikasi.csv
# Pustaka: pandas, scipy, statsmodels, matplotlib
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def judul(teks): print("\n" + "=" * 70 + "\n" + teks + "\n" + "=" * 70)

# --- Slide 3, 5, 6: data, eksplorasi, regresi sederhana ----------------------
judul("Slide 3 & 6: data dan statistik deskriptif")
df = pd.read_csv("data_respons_aplikasi.csv")
print(df.head())
print(df.describe().round(2))
print(df[["waktu_respons", "pengguna", "cpu", "memori"]].corr().round(2))

judul("Slide 5: slope regresi sederhana (bandingkan dengan koefisien berganda)")
for v in ["pengguna", "cpu", "memori"]:
    s = sm.OLS(df["waktu_respons"], sm.add_constant(df[[v]])).fit()
    print(f"{v:9s} slope = {s.params[v]:.2f}")

# --- Slide 9-10: membangun model ---------------------------------------------
judul("Slide 9-10: model regresi berganda")
cols = ["pengguna", "cpu", "memori"]
X = sm.add_constant(df[cols])
m1 = sm.OLS(df["waktu_respons"], X).fit()
print(m1.summary())
print(m1.conf_int().round(2))              # selang kepercayaan 95%

# --- Slide 13: uji simultan, R2 dan Adjusted R2 ------------------------------
judul("Slide 13: uji simultan (F), R2, Adjusted R2")
print(f"F = {m1.fvalue:.1f} (db 3 dan {int(m1.df_resid)}), p = {m1.f_pvalue:.3g}")
print(f"R2 = {m1.rsquared:.3f}, Adjusted R2 = {m1.rsquared_adj:.3f}")

# --- Slide 14-17: ANOVA regresi ----------------------------------------------
judul("Slide 14-17: ANOVA regresi")
m0 = smf.ols("waktu_respons ~ 1", df).fit()                                # model nol
m1 = smf.ols("waktu_respons ~ pengguna + cpu + memori", df).fit()         # model lengkap
print(anova_lm(m0, m1))                                                   # tabel ANOVA

y = df["waktu_respons"]
sst = ((y - y.mean()) ** 2).sum(); sse = (m1.resid ** 2).sum(); ssr = sst - sse
k, n = 3, len(df)
msr, mse, mst = ssr / k, sse / (n - k - 1), sst / (n - 1)
print(f"SST = {sst:,.1f}   SSR = {ssr:,.1f}   SSE = {sse:,.1f}")
print(f"KT regresi = {msr:,.1f}   KT galat = {mse:,.1f}   akar KT galat = {mse ** 0.5:.1f}")
print(f"F = KT regresi / KT galat = {msr / mse:.1f}")
r2 = ssr / sst
print(f"R2 = SSR/SST = {r2:.3f}")
print(f"F dari R2 = {(r2 / k) / ((1 - r2) / (n - k - 1)):.1f}")
print(f"KT total = {mst:,.1f};  Adjusted R2 = 1 - KT galat/KT total = {1 - mse / mst:.3f}")

# --- Slide 18-27: uji asumsi klasik ------------------------------------------
judul("Slide 21: Asumsi 1 - Linearitas (Ramsey RESET)")
print(linear_reset(m1, power=3, test_type="fitted", use_f=True))

judul("Slide 23: Asumsi 2 - Normalitas residual")
print(stats.shapiro(m1.resid))
print("Jarque-Bera (JB, p, skew, kurtosis):", [round(float(v), 4) for v in jarque_bera(m1.resid)])
sm.qqplot(m1.resid, line="s"); plt.title("Q-Q plot residual"); plt.show()

judul("Slide 25: Asumsi 3 - Homoskedastisitas (Breusch-Pagan)")
lm, p_lm, f, p_f = het_breuschpagan(m1.resid, X)
print(f"LM = {lm:.3f}, p = {p_lm:.3f}   (F = {f:.3f}, p = {p_f:.3f})")

judul("Slide 27: Asumsi 4 - Independensi galat")
print("Durbin-Watson d =", round(durbin_watson(m1.resid), 3))
bg = acorr_breusch_godfrey(m1, nlags=1)
print(f"Breusch-Godfrey: LM = {bg[0]:.3f}, p = {bg[1]:.3f}")

judul("Slide 29: Asumsi 5 - Multikolinearitas (VIF)")
print(dict(zip(cols, [round(vif(X.values, i), 2) for i in range(1, X.shape[1])])))

# --- Slide 30: demo multikolinearitas (tambah req_detik) ---------------------
judul("Slide 30: Model B (+ req_detik)")
X2 = sm.add_constant(df[cols + ["req_detik"]])
m2 = sm.OLS(df["waktu_respons"], X2).fit()
print(m2.summary().tables[1])
print("VIF:", dict(zip(cols + ["req_detik"], [round(vif(X2.values, i), 1) for i in range(1, X2.shape[1])])))
print(f"Adjusted R2: Model A = {m1.rsquared_adj:.3f}, Model B = {m2.rsquared_adj:.3f}")
