# =============================================================================
# Pertemuan 3 - Kualitas Data dan Kelayakan Hasil Statistik (Python)
# Berkas: data_kuesioner_kecepatan.csv, data_respons_aplikasi.csv,
#         data_respons_aplikasi_insiden.csv, data_variabel_acak.csv
# Pustaka: pandas, scipy, statsmodels
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.multitest import multipletests

def judul(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

# ===================== BAGIAN A: VALIDITAS DAN RELIABILITAS ==================
kq = pd.read_csv("data_kuesioner_kecepatan.csv")
Q = kq[[f"Q{i}" for i in range(1, 8)]]
judul("Statistik butir (rata-rata, SD)"); print(Q.agg(["mean", "std"]).round(2).T)

judul("Uji validitas: korelasi Pearson butir vs skor total (terkoreksi)")
total = Q.sum(axis=1)
for b in Q.columns:
    r, p = stats.pearsonr(Q[b], total - Q[b])          # total tanpa butir itu sendiri
    print(f"{b}: r = {r:.3f}, p = {p:.4f}", "-> valid" if p < 0.05 else "-> TIDAK valid")

def cronbach_alpha(d):
    k = d.shape[1]
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))

judul("Reliabilitas: Cronbach's alpha")
print("alpha 7 butir =", round(cronbach_alpha(Q), 3))
print("alpha jika butir dihapus:", {b: float(round(cronbach_alpha(Q.drop(columns=b)), 3)) for b in Q.columns})
Q6 = Q.drop(columns="Q7")                              # butir Q7 dibuang
print("alpha 6 butir =", round(cronbach_alpha(Q6), 3))
total6 = Q6.sum(axis=1)
print("validitas 6 butir:", {b: float(round(stats.pearsonr(Q6[b], total6 - Q6[b])[0], 3)) for b in Q6.columns})
print("alpha jika butir dihapus (6 butir):", {b: float(round(cronbach_alpha(Q6.drop(columns=b)), 3)) for b in Q6.columns})

# ===================== BAGIAN B: OUTLIER (data jam insiden) =================
judul("Outlier: data dengan 3 baris insiden")
cols = ["pengguna", "cpu", "memori"]
asli = pd.read_csv("data_respons_aplikasi.csv")
insiden = pd.read_csv("data_respons_aplikasi_insiden.csv")
def fit(d): return sm.OLS(d["waktu_respons"], sm.add_constant(d[cols])).fit()
mA, mB = fit(asli), fit(insiden)
for nama, m in (("tanpa insiden (n=120)", mA), ("dengan insiden (n=123)", mB)):
    print(nama, "| R2 =", round(m.rsquared, 3), "| galat residual =", round(np.sqrt(m.scale), 1))
    print(pd.DataFrame({"koef": m.params, "SE": m.bse, "p": m.pvalues}).round(3))
    print("Shapiro p =", round(stats.shapiro(m.resid).pvalue, 4), "| Breusch-Pagan p =", round(het_breuschpagan(m.resid, m.model.exog)[1], 3))

inf = OLSInfluence(mB); N = len(insiden)
diag = pd.DataFrame({"resid_stud": inf.resid_studentized_external, "leverage": inf.hat_matrix_diag, "cooks_d": inf.cooks_distance[0]})
print("Ambang: leverage >", round(2 * 4 / N, 3), "| Cook's D > 1 (atau 4/n =", round(4 / N, 4), ") | |resid_stud| > 3")
print(diag.sort_values("cooks_d", ascending=False).head(4).round(3))
q1, q3 = insiden["waktu_respons"].quantile([.25, .75]); iqr = q3 - q1
print("Pagar IQR waktu_respons:", round(q1 - 1.5 * iqr, 1), "sampai", round(q3 + 1.5 * iqr, 1),
      "| baris di luar:", list(insiden.index[(insiden["waktu_respons"] < q1 - 1.5 * iqr) | (insiden["waktu_respons"] > q3 + 1.5 * iqr)] + 1))

judul("Troubleshooting: regresi robust dan galat baku HC3")
rob = sm.RLM(insiden["waktu_respons"], sm.add_constant(insiden[cols]), M=sm.robust.norms.HuberT()).fit()
hc3 = sm.OLS(insiden["waktu_respons"], sm.add_constant(insiden[cols])).fit(cov_type="HC3")
print(pd.DataFrame({"OLS": mB.params, "RLM (Huber)": rob.params, "SE OLS": mB.bse, "SE RLM": rob.bse, "SE HC3": hc3.bse, "p HC3": hc3.pvalues}).round(3))

# ===================== BAGIAN B: BIAS SAMPEL ================================
judul("Bias sampel: model pada subset data")
sibuk = asli[asli["pengguna"] >= 30]; lambat = asli[asli["waktu_respons"] > 200]
for nama, d in (("semua jam (n=120)", asli), ("hanya jam sibuk, pengguna >= 30", sibuk), ("hanya jam lambat, waktu > 200 ms", lambat)):
    m = fit(d); print(f"{nama}: n = {len(d)}, R2 = {m.rsquared:.3f}, koef =", m.params.round(2).to_dict(), "| p memori =", round(m.pvalues['memori'], 3))

# ===================== BAGIAN B: CONFIRMATION BIAS / UJI BERGANDA ===========
judul("Uji berganda: 20 variabel acak (tidak berhubungan dengan Y)")
acak = pd.read_csv("data_variabel_acak.csv")
p = np.array([sm.OLS(asli["waktu_respons"], sm.add_constant(acak[c])).fit().pvalues.iloc[1] for c in acak])
print("p < 0,05:", [acak.columns[i] for i in np.where(p < .05)[0]], "| p minimum =", round(p.min(), 4))
print("Peluang minimal 1 'signifikan' kebetulan =", round(1 - 0.95 ** 20, 3))
print("p terkoreksi Bonferroni (minimum) =", round(multipletests(p, method="bonferroni")[1].min(), 3))
