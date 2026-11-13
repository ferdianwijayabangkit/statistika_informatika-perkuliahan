# =============================================================================
# Pertemuan 1 - Analisis Multivariat dan Regresi Linear Berganda (R)
# Urutan kode mengikuti slide. Berkas data: data_respons_aplikasi.csv
# Paket tambahan: car, lmtest  ->  install.packages(c("car", "lmtest"))
# =============================================================================
library(car); library(lmtest)

# --- Slide 3, 5, 6: data, eksplorasi, regresi sederhana ----------------------
data <- read.csv("data_respons_aplikasi.csv")
str(data)
summary(data)                                        # rata-rata, min, maks
round(sapply(data[, c("waktu_respons", "pengguna", "cpu", "memori")], sd), 2)   # SD
round(cor(data[, c("waktu_respons", "pengguna", "cpu", "memori")]), 2)

# Slide 5: slope regresi sederhana (bandingkan dengan koefisien berganda)
for (v in c("pengguna", "cpu", "memori")) {
  s <- lm(as.formula(paste("waktu_respons ~", v)), data = data)
  cat(v, "slope =", round(coef(s)[2], 2), "\n")
}

# --- Slide 9-10: membangun model ---------------------------------------------
m1 <- lm(waktu_respons ~ pengguna + cpu + memori,
         data = data)

summary(m1)   # koefisien, uji t, R², uji F
confint(m1)   # selang kepercayaan 95%

# --- Slide 13: uji simultan, R2 dan Adjusted R2 ------------------------------
fs <- summary(m1)$fstatistic
cat("F =", round(fs[1], 1), "(db", fs[2], "dan", fs[3], "), p =",
    signif(pf(fs[1], fs[2], fs[3], lower.tail = FALSE), 3), "\n")
cat("R2 =", round(summary(m1)$r.squared, 3), " Adjusted R2 =", round(summary(m1)$adj.r.squared, 3), "\n")

# --- Slide 14-17: ANOVA regresi ----------------------------------------------
m0 <- lm(waktu_respons ~ 1, data = data)   # model nol
anova(m0, m1)   # m1: model lengkap  -> tabel ANOVA (RSS model nol = SST)

y <- data$waktu_respons
sst <- sum((y - mean(y))^2); sse <- sum(residuals(m1)^2); ssr <- sst - sse
k <- 3; n <- nrow(data)
msr <- ssr / k; mse <- sse / (n - k - 1); mst <- sst / (n - 1)
cat("SST =", round(sst, 1), " SSR =", round(ssr, 1), " SSE =", round(sse, 1), "\n")
cat("KT regresi =", round(msr, 1), " KT galat =", round(mse, 1), " akar KT galat =", round(sqrt(mse), 1), "\n")
cat("F = KT regresi / KT galat =", round(msr / mse, 1), "\n")
r2 <- ssr / sst
cat("R2 = SSR/SST =", round(r2, 3), "\n")
cat("F dari R2 =", round((r2 / k) / ((1 - r2) / (n - k - 1)), 1), "\n")
cat("KT total =", round(mst, 1), " Adjusted R2 = 1 - KT galat/KT total =", round(1 - mse / mst, 3), "\n")

# --- Slide 18-27: uji asumsi klasik ------------------------------------------
# Slide 21: Asumsi 1 - Linearitas (Ramsey RESET)
resettest(m1, power = 2:3, type = "fitted")
plot(m1, which = 1)   # residual vs prediksi

# Slide 23: Asumsi 2 - Normalitas residual
shapiro.test(residuals(m1))
qqnorm(residuals(m1))
qqline(residuals(m1))

# Slide 25: Asumsi 3 - Homoskedastisitas (Breusch-Pagan, versi Koenker)
bptest(m1)
plot(m1, which = 3)   # scale-location

# Slide 27: Asumsi 4 - Independensi galat
dwtest(m1)            # statistik d (p-value R memakai alternatif "greater")
bgtest(m1, order = 1)

# Slide 29: Asumsi 5 - Multikolinearitas (VIF)
vif(m1)

# --- Slide 30: demo multikolinearitas (tambah req_detik) ---------------------
m2 <- lm(waktu_respons ~ pengguna + cpu + memori + req_detik, data = data)
summary(m2)
round(vif(m2), 1)
cat("Adjusted R2: Model A =", round(summary(m1)$adj.r.squared, 3),
    " Model B =", round(summary(m2)$adj.r.squared, 3), "\n")
