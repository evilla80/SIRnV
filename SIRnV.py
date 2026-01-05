import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Caricamento dati
df_flu = pd.read_csv('flunet.csv')
df_vax = pd.read_csv('vaccini.csv', sep=';')

# Parametri globali
n_stages = 3
gamma = 4.3
r = 0.5
beta = 9.5
I0 = 0.001


def clean_flu_data(df_flu_clean):
    italy_df = df_flu_clean[df_flu_clean['COUNTRY_AREA_TERRITORY'] == 'Italy'].copy()
    italy_df['ISO_WEEKSTARTDATE'] = pd.to_datetime(italy_df['ISO_WEEKSTARTDATE'])

    # crea colonne Year e Month
    italy_df['Year'] = italy_df['ISO_WEEKSTARTDATE'].dt.year
    italy_df['Month'] = italy_df['ISO_WEEKSTARTDATE'].dt.month

    # Aggrega mensilmente
    monthly_flu = italy_df.groupby(['Year', 'Month'])['INF_ALL'].sum().reset_index()
    return monthly_flu


def get_vax_rate(df_vax_clean, year_start):
    anno = f"{year_start}-{str(year_start + 1)[2:]}"  # crea "2017-18"

    riga = df_vax_clean[df_vax['anno'] == anno]
    if not riga.empty:
        return riga['tasso'].values[0] / 100  # trasforma 15.3 in 0.153
    return 0


def get_flu_season(df, year_start):
    season = df[((df['Year'] == year_start) & (df['Month'] >= 11)) | \
                ((df['Year'] == year_start + 1) & (df['Month'] <= 4))].copy()
    season = season.sort_values(['Year', 'Month'])

    # numero mese nella stagione
    season['Num'] = np.arange(len(season)) + 1
    return season


def sirnv_model(y, t, beta, gamma, r, vax_rate, n):
    S = y[0]
    I = y[1]
    R = y[2:]

    # suscettibilità residua per gli stadi R
    s = [0.1 * (i + 1) for i in range(n)]

    if t <= 1.5:
        v = vax_rate * np.exp(-t)
    else:
        v = 0

    dsdt = -beta * S * I - v * S + r * R[n - 1]

    # infezioni da S e da tutti gli stati R
    infections_from_R = sum(s[i] * beta * R[i] * I for i in range(n))
    didt = beta * S * I + infections_from_R - gamma * I

    # Equazioni R
    drdt = [0] * n
    # R1 riceve i guariti e i vaccinati
    drdt[0] = gamma * I + v * S - s[0] * beta * R[0] * I - r * R[0]
    # Passaggio tra stadi R_i -> R_{i+1} (invecchiamento immunità)
    for i in range(1, n - 1):
        drdt[i] = r * R[i - 1] - s[i] * beta * R[i] * I - r * R[i]
    # Ultimo stadio Rn
    drdt[n - 1] = r * R[n - 2] - s[n - 1] * beta * R[n - 1] * I - r * R[n - 1]

    return [dsdt, didt] + drdt


def run_simulation(beta, vax_rate, n_stages, t_grid, gamma, r, I0):
    S0 = 1 - I0 - vax_rate
    R0 = [vax_rate] + [0.0] * (n_stages - 1)
    y0 = [S0, I0] + R0

    sol = odeint(sirnv_model, y0, t_grid,
                 args=(beta, gamma, r, vax_rate, n_stages))
    return sol


def plot_results(t, sol, real_data, year, vax_rate, filename, title_suffix=""):
    plt.figure(figsize=(10, 6))

    # Scalatura dati reali sul picco della simulazione
    if not real_data.empty:
        # Calcolo fattore dinamico per allineare il picco visivamente
        factor = sol[:, 1].max() / real_data['INF_ALL'].max()
        real_scaled = real_data['INF_ALL'] * factor
        plt.scatter(real_data['Num'] - 1, real_scaled, color='blue', label='Dati Reali (Scalati)')

    plt.plot(t, sol[:, 1], 'r-', linewidth=2, label='Modello SIRnV (Infetti)')

    plt.title(f"Stagione {year}-{year + 1}: Modello vs Realtà (Vax: {vax_rate * 100:.1f}%) {title_suffix}")
    plt.xlabel("Mesi (0=Nov, 1=Dic, ...)")
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()  # Chiude per risparmiare memoria


# --- FUNZIONI PER ANALISI APPROFONDITE (What-If) ---

def plot_sensitivity_r(t, beta, gamma, vax_rate, n_stages, I0):
    # Valori di r da testare: Lento, Medio (Base), Veloce
    r_values = [0.2, 0.5, 0.8]
    colors = ['green', 'red', 'orange']
    labels = ['Immunità Lunga (r=0.2)', 'Immunità Media (r=0.5)', 'Immunità Breve (r=0.8)']

    plt.figure(figsize=(10, 6))

    for i, r_val in enumerate(r_values):
        # Ricalcola simulazione variando solo r
        sol = run_simulation(beta, vax_rate, n_stages, t, gamma, r_val, I0)
        plt.plot(t, sol[:, 1], label=labels[i], color=colors[i], linewidth=2)

    plt.title("Analisi di Sensitività: Effetto del decadimento dell'immunità (r)")
    plt.xlabel("Mesi")
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_sensitivity_r.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_vaccine_scenarios(t, beta, gamma, r, n_stages, vax_real, I0):
    # Scenari: 0%, Reale (2017), Target Alto (30%)
    vax_scenarios = [0.0, vax_real, 0.30]
    labels = ['Nessun Vaccino (0%)', f'Reale 2017 ({vax_real * 100:.1f}%)', 'Target Ottimale (30%)']
    styles = ['--', '-', '-.']

    plt.figure(figsize=(10, 6))

    for i, v_rate in enumerate(vax_scenarios):
        # Ricalcola simulazione variando tasso vaccinale
        sol = run_simulation(beta, v_rate, n_stages, t, gamma, r, I0)

        # Calcolo picco per mostrarlo in legenda
        peak = sol[:, 1].max()
        plt.plot(t, sol[:, 1], linestyle=styles[i], linewidth=2, label=f'{labels[i]} - Picco: {peak:.3f}')

    plt.title("Analisi Scenari: Impatto della Copertura Vaccinale")
    plt.xlabel("Mesi")
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_vaccine.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rt(t, sol, beta, gamma, n_stages):
    # Estrazione variabili di stato S e R
    S_t = sol[:, 0]
    R_t = sol[:, 2:]

    # Pesi suscettibilità (gli stessi definiti nel modello)
    s = [0.1 * (i + 1) for i in range(n_stages)]

    # Calcolo suscettibilità media della popolazione nel tempo
    # Somma pesata: S vale 1, R[i] vale s[i]
    susceptibility = S_t + sum(s[i] * R_t[:, i] for i in range(n_stages))

    # Rt = R0_base * suscettibilità_media
    Rt_curve = (beta / gamma) * susceptibility

    plt.figure(figsize=(10, 6))
    plt.plot(t, Rt_curve, color='purple', linewidth=2, label='Rt Effettivo')

    # Linea di soglia a 1 (dove l'epidemia inizia a decrescere)
    plt.axhline(1.0, color='black', linestyle='--', label='Soglia Rt=1')

    plt.title("Andamento del Numero di Riproduzione (Rt) nel tempo")
    plt.xlabel("Mesi")
    plt.ylabel("Valore Rt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_rt.png", dpi=300, bbox_inches="tight")
    plt.close()


# --- ESECUZIONE DEL CODICE ---

df_clean = clean_flu_data(df_flu)
t = np.linspace(0, 5, 100)

print("Inizio Analisi...")

# 1. Recupero dati Stagione 2017-2018 (Il nostro Caso Studio)
season_17 = get_flu_season(df_clean, 2017)
vax_17 = get_vax_rate(df_vax, 2017)

# 2. Simulazione Principale (Calibrazione)
print(f"Simulazione Stagione 2017 (Vax Rate: {vax_17:.3f})...")
sol_17 = run_simulation(beta, vax_17, n_stages, t, gamma, r, I0)
plot_results(t, sol_17, season_17, 2017, vax_17, "sirnv_2017_2018.png")

# 3. Analisi Avanzate (Basate sui parametri del 2017)
print("Generazione Analisi di Sensitività (r)...")
plot_sensitivity_r(t, beta, gamma, vax_17, n_stages, I0)

print("Generazione Scenari Vaccinali...")
plot_vaccine_scenarios(t, beta, gamma, r, n_stages, vax_17, I0)

print("Calcolo e Grafico Rt...")
plot_rt(t, sol_17, beta, gamma, n_stages)

print("Tutto completato. Grafici salvati.")