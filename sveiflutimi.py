import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("mælingarexc.xlsx")
df["dags"] = pd.to_datetime(df["dags"])


Tcol = pd.to_numeric(df["sveiflutimi"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
Tcol = Tcol.dropna()

T_med = float(Tcol.median())   
T_min = float(Tcol.quantile(0.01)) 

print(f"Median sveiflutími: {T_med:.2f} s")
print(f"5% quantile sveiflutími (styttri endi): {T_min:.2f} s")


T = T_min 

f = 1.0 / T

# --- 2) Búa til "samfellda" bylgju sem stendur t.d. í 60 sek ---
duration = 60.0
fs_cont = 2000  
t = np.arange(0, duration, 1/fs_cont)
x = np.sin(2*np.pi*f*t)

# --- 3) Sýnatökur ---
fs_fiber = 200
fs_good  = 2

# Buoy sampling: 30 mín = 1800 sek -> fs = 1/1800 Hz
fs_buoy = 1/1800

def sample_signal(t_cont, x_cont, fs):
    ts = np.arange(0, duration, 1/fs)
    xs = np.interp(ts, t_cont, x_cont)
    return ts, xs

t200, x200 = sample_signal(t, x, fs_fiber)
t5,   x5   = sample_signal(t, x, fs_good)

# Fyrir buoy sampling yfir 60 sek verður bara 0 eða 1 punktur.
# Til að sýna aliasing sjónrænt, lengjum tímann fyrir buoy panel:
duration_buoy = 6*3600  # 6 klst til að sjást punktar á 30 mín fresti
t_b = np.arange(0, duration_buoy, 1/fs_cont)
x_b = np.sin(2*np.pi*f*t_b)
t30, x30 = sample_signal(t_b, x_b, fs_buoy)  # 30 mín sampling

# --- 4) Teikna "whiteboard" style mynd ---
fig = plt.figure(figsize=(12, 6))

# Panel A: Samfelld bylgja + 200 Hz punktar
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t, x, linewidth=2)
ax1.scatter(t200[:2000], x200[:2000], s=8)  # sýna fyrstu ~10 sek (2000 punktar í cont)
ax1.set_title(f"Samfelld bylgja T={T:.2f}s + sampling 200 Hz (ljósleiðari)")
ax1.set_xlabel("t (s)")
ax1.set_ylabel("x(t)")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# Panel B: Samfelld bylgja + 5 Hz punktar (gott downsample)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t, x, linewidth=2)
ax2.scatter(t5, x5, s=25)
ax2.set_title("Downsample: 2hz")
ax2.set_xlabel("t (s)")
ax2.set_ylabel("x(t)")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 20)



plt.tight_layout()
plt.show()
