import sys

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime


df_flu = pd.read_csv('flunet.csv')
df_vax = pd.read_csv('vaccini.csv', names=['anno', 'tasso'], header=None) 


def clean_flu_data(df):
    italy_df = df[df['COUNTRY_AREA_TERRITORY'] == 'Italy'].copy()
    italy_df['ISO_WEEKSTARTDATE'] = pd.to_datetime(italy_df['ISO_WEEKSTARTDATE'])

    # crea colonne Year e Month
    italy_df['Year'] = italy_df['ISO_WEEKSTARTDATE'].dt.year
    italy_df['Month'] = italy_df['ISO_WEEKSTARTDATE'].dt.month

    # Aggrega mensilmente
    monthly_flu = italy_df.groupby(['Year', 'Month'])['INF_ALL'].sum().reset_index()
    return monthly_flu

def get_vax_rate(df_vax, year_start):
    anno = f"{year_start}-{str(year_start+1)[2:]}" # crea "2017-18"
    
    riga = df_vax[df_vax['anno'] == anno]
    if not riga.empty:
        return riga['tasso'].values[0] / 100 # trasforma 15.3 in 0.153
    return 0.15 

def get_flu_season(df, year_start):
    season = df[((df['Year'] == year_start) & (df['Month'] >= 11)) | \
           ((df['Year'] == year_start + 1) & (df['Month'] <= 4))].copy()
    season = season.sort_values(['Year', 'Month'])

    # numero mese nella stagione
    season['Num'] = np.arange(len(season)) + 1
    return season

def sirnv_model(y, t, beta, gamma, r, vax_rate, n ):
    S = y[0]
    I = y[1]
    R = y[2:]

    #suscettibilità residua per gli stadi R
    s = [0.1 * (i+1) for i in range(n)]

    if t <= 1.5: 
        v = vax_rate * np.exp(-t) 
    else: 
        v = 0
    
    dsdt = -beta * S * I - v * S + r * R[n-1]
    
    #infezioni da S e da tutti gli stati R
    infections_from_R = sum(s[i] * beta * R[i] * I for i in range(n))
    didt = beta * S * I + infections_from_R - gamma * I

    # Equazioni R
    drdt = [0] * n
    # R1 riceve i guariti e i vaccinati
    drdt[0] = gamma * I + v * S - s[0] * beta * R[0] * I - r * R[0]
    # Passaggio tra stadi R_i -> R_{i+1} (invecchiamento immunità)
    for i in range(1, n-1):
        drdt[i] = r * R[i-1] - s[i] * beta * R[i] * I - r * R[i]
    # Ultimo stadio Rn
    drdt[n-1] = r * R[n-2] - s[n-1] * beta * R[n-1] * I - r * R[n-1]

    return [dsdt, didt] + drdt

df_clean = clean_flu_data(df_flu)

season_17_18 = get_flu_season(df_clean, 2017)
season_18_19 = get_flu_season(df_clean, 2018)

#parametri 
n_stages = 3
beta = 9.5 #trasmissibilità
gamma = 4.3 #perdita infezione
r = 0.5 #perdita immunità 
vax_rate_2017 = get_vax_rate(df_vax, 2017)
vax_rate_2018 = get_vax_rate(df_vax, 2018)

#condizioni iniziali
I0 = 0.001
S0 = 1 - I0 - vax_rate_2017
R0 = [vax_rate_2018] + [0.0] * (n_stages - 1)
y0 = [S0, I0] + R0

t = np.linspace(0, 5, 100)

# Integrazione
sol = odeint(sirnv_model, y0, t, args=(beta, gamma, r, vax_rate_2017, n_stages))

#Grafico
plt.figure(figsize=(10, 6))

real_scaled = season_17_18['INF_ALL'] / season_17_18['INF_ALL'].max() * sol[:, 1].max()
plt.scatter(season_17_18['Num'] - 1, real_scaled, color='blue', label='Dati Reali (Scalati)')

# Simulazione
plt.plot(t, sol[:, 1], 'r-', label='Modello SIRnV (Infetti)')
plt.title(f"Stagione Influenzale 2017-18: Modello vs Realtà (Vax: {vax_rate*100}%)")
plt.xlabel("Mesi (0=Nov, 1=Dic, ...)")
plt.ylabel("Proporzione Infetti")
plt.legend()
plt.show()