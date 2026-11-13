# =============================================================================
# Pertemuan 3 - Kualitas Data dan Kelayakan Hasil Statistik (R)
# Berkas: data_kuesioner_kecepatan.csv, data_respons_aplikasi.csv,
#         data_respons_aplikasi_insiden.csv, data_variabel_acak.csv
# Paket tambahan: lmtest, sandwich, MASS (MASS sudah termasuk di R)
#   install.packages(c("lmtest", "sandwich"))
# =============================================================================
library(lmtest); library(sandwich); library(MASS)

# ===================== BAGIAN A: VALIDITAS DAN RELIABILITAS ==================
kq <- read.csv("data_kuesioner_kecepatan.csv")
Q  <- kq[, paste0("Q", 1:7)]
round(sapply(Q, function(x) c(rata2 = mean(x), sd = sd(x))), 2)

# Uji validitas: korelasi Pearson butir vs skor total (terkoreksi)
total <- rowSums(Q)
for (b in names(Q)) {
  uji <- cor.test(Q[[b]], total - Q[[b]])          # total tanpa butir itu sendiri
  cat(b, ": r =", round(uji$estimate, 3), ", p =", round(uji$p.value, 4), "\n")
}

# Reliabilitas: Cronbach's alpha
cronbach_alpha <- function(d) {
  k <- ncol(d)
  k / (k - 1) * (1 - sum(apply(d, 2, var)) / var(rowSums(d)))
}
cronbach_alpha(Q)                                   # 7 butir
sapply(names(Q), function(b) cronbach_alpha(Q[, names(Q) != b]))   # alpha jika butir dihapus
Q6 <- Q[, names(Q) != "Q7"]                         # butir Q7 dibuang
cronbach_alpha(Q6)
# Alternatif: psych::alpha(Q6)   # install.packages("psych")

# ===================== BAGIAN B: OUTLIER (data jam insiden) =================
asli    <- read.csv("data_respons_aplikasi.csv")
insiden <- read.csv("data_respons_aplikasi_insiden.csv")
mA <- lm(waktu_respons ~ pengguna + cpu + memori, data = asli)
mB <- lm(waktu_respons ~ pengguna + cpu + memori, data = insiden)
summary(mA); summary(mB)
shapiro.test(residuals(mB)); bptest(mB)

N <- nrow(insiden)
diag <- data.frame(resid_stud = rstudent(mB), leverage = hatvalues(mB), cooks_d = cooks.distance(mB))
head(diag[order(-diag$cooks_d), ], 4)               # 4 baris dengan Cook's D terbesar
# Ambang: leverage > 2*(k+1)/N ; Cook's D > 1 (atau 4/N) ; |resid_stud| > 3
subset(diag, cooks_d > 1 | leverage > 2 * 4 / N | abs(resid_stud) > 3)

q <- quantile(insiden$waktu_respons, c(.25, .75)); iqr <- q[2] - q[1]
which(insiden$waktu_respons < q[1] - 1.5 * iqr | insiden$waktu_respons > q[2] + 1.5 * iqr)

# Troubleshooting: regresi robust (Huber) dan galat baku HC3
rob <- rlm(waktu_respons ~ pengguna + cpu + memori, data = insiden)   # psi.huber (bawaan)
summary(rob)
coeftest(mB, vcov = vcovHC(mB, type = "HC3"))

# ===================== BAGIAN B: BIAS SAMPEL ================================
sibuk  <- subset(asli, pengguna >= 30)
lambat <- subset(asli, waktu_respons > 200)
for (d in list(asli, sibuk, lambat)) {
  m <- lm(waktu_respons ~ pengguna + cpu + memori, data = d)
  cat("n =", nrow(d), " R2 =", round(summary(m)$r.squared, 3), " koef =", round(coef(m), 2), "\n")
}

# ===================== BAGIAN B: CONFIRMATION BIAS / UJI BERGANDA ===========
acak <- read.csv("data_variabel_acak.csv")          # 20 variabel acak tak berhubungan dengan Y
p <- sapply(acak, function(x) summary(lm(asli$waktu_respons ~ x))$coefficients[2, 4])
names(p)[p < 0.05]; min(p)                          # yang 'signifikan' kebetulan
1 - 0.95^20                                         # peluang minimal 1 'signifikan'
min(p.adjust(p, method = "bonferroni"))             # setelah koreksi Bonferroni
