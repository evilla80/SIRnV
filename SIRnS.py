import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Caricamento dati
df_flu = pd.read_csv('flunet.csv')
df_vax = pd.read_csv('vaccini.csv', sep=';')

# Parametri globali
n_stages = 3
gamma = 4
r = 0.5
beta = 8
I0 = 0.001

s_min = 0.01
s_max = 0.7

t_vax = 2 #durata della campagna attiva vaccinale
pop_ita = 60e6


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
    season['Num'] = np.arange(len(season))
    return season


def sirnv_model(y, t, beta, gamma, r, vax_rate, n):
    S = y[0]
    I = y[1]
    R = y[2:]

    #creo n valori distribuiti tra 0.01 e 0.7, in base al numero di stage
    s = np.linspace(s_min, s_max, n)

    infezioniR = 0
    #somma degli infetti che provengono da qualsiasi Ri
    for i in range(n):
        nuovi_infetti = s[i] * beta * R[i] * I
        infezioniR = infezioniR + nuovi_infetti

    if t <= t_vax:
        v = vax_rate * np.exp(-t)
    else:
        v = 0

    dsdt = -beta * S * I - v * S + r * R[n - 1]

    didt = beta * S * I + infezioniR - gamma * I

    drdt = np.zeros(n)
    # R1 riceve i guariti e i vaccinati
    drdt[0] = gamma * I + v * S - (s[0] * beta * R[0] * I) - (r * R[0])
    # Passaggio tra stadi R_i -> R_{i+1} (invecchiamento immunità)
    for i in range(1, n):
        drdt[i] = r * R[i - 1] - (s[i] * beta * R[i] * I) - r * R[i]

    ris = [dsdt, didt] + drdt.tolist()

    return ris


def run_simulation(beta, vax_rate_target, n_stages, t_grid, gamma, r, I0):
    vax_october = 0.15
    initial_immune = vax_rate_target * vax_october

    vax_rate_rim = vax_rate_target - initial_immune

    R0 = [0.0] * n_stages
    S0 = 1 - I0 - sum(R0)
    R0[0] = initial_immune
    y0 = [S0, I0] + R0

    # Affinché l'integrale della vaccinazione nel tempo sia pari al tasso target
    integral_factor = 1 - np.exp(-t_vax)
    vax_rate = vax_rate_rim / integral_factor

    sol = odeint(sirnv_model, y0, t_grid,
                 args=(beta, gamma, r, vax_rate, n_stages))
    return sol


def plot_results(t, sol, real_data, year, vax_rate, filename, title_suffix=""):
    plt.figure(figsize=(10, 6))

    plt.plot(t, sol[:, 1], 'r-', linewidth=2, label='Modello SIRnS (Infetti)')

    # Scalatura dati reali sul picco della simulazione
    if not real_data.empty:
        # Calcolo fattore dinamico per allineare il picco visivamente
        peak_model = sol[:, 1].max()
        peak = real_data['INF_ALL'] .max()

        factor = peak_model / peak
        real_scaled = real_data['INF_ALL'] * factor
        plt.scatter(real_data['Num'], real_scaled, color='blue', label='Dati Reali (Scalati)')



    plt.title(f"Stagione {year}-{year + 1}: Modello vs Realtà (Vax: {vax_rate * 100:.1f}%) {title_suffix}")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
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
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
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
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_vaccine.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rt(t, sol, beta, gamma, n_stages):
    # Estrazione variabili di stato S e R
    S_t = sol[:, 0]
    R_t = sol[:, 2:]

    s = np.linspace(s_min, s_max, n_stages)  # Da 0.01 a 0.7

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
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Valore Rt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_rt.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_sir_dynamics(t, sol, year):
    # 1. Estrazione dei dati
    S = sol[:, 0]
    I = sol[:, 1]

    # Sommiamo tutte le colonne da 2 in poi (tutti gli stadi R0, R1, R2...)
    # axis=1 significa "somma per ogni riga (istante di tempo)"
    R_total = np.sum(sol[:, 2:], axis=1)

    # 2. Creazione del Grafico
    plt.figure(figsize=(10, 6))

    # Curva Blu: Suscettibili (Scende man mano che la gente si ammala o si vaccina)
    plt.plot(t, S, color='blue', linewidth=2.5, label='Suscettibili (S)')

    # Curva Verde: Immuni (Sale grazie a guarigioni e vaccini)
    plt.plot(t, R_total, color='green', linewidth=2.5, label='Immuni/Guariti (R_tot)')

    # Curva Rossa: Infetti (Fa il picco e poi scende)
    plt.plot(t, I, color='red', linewidth=2.5, label='Infetti (I)')

    # 3. Formattazione Estetica
    plt.title(f"Dinamica Completa SIR - Stagione {year}")
    plt.xlabel("Mesi")
    plt.ylabel("Popolazione")

    # Etichette Mesi (Nov, Dic...) come negli altri grafici
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)

    # Griglia e Legenda
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=11, framealpha=1, shadow=True)

    # Salvataggio
    plt.savefig("analysis_full_dynamics_SIR.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Grafico S-I-R salvato: analysis_full_dynamics_SIR.png")

def plot_counterfactual_2021_rates(t, beta, gamma, r, n_stages, I0, season_data):
    rate_2017 = get_vax_rate(df_vax, 2017)

    other_year = 2021
    rate_other = get_vax_rate(df_vax, other_year)

    # Eseguiamo le due simulazioni
    sol_real = run_simulation(beta, rate_2017, n_stages, t, gamma, r, I0)
    sol_hypo = run_simulation(beta, rate_other, n_stages, t, gamma, r, I0)

    # Calcolo riduzione picco
    peak_real = sol_real[:, 1].max()
    peak_hypo = sol_hypo[:, 1].max()
    reduction = (peak_real - peak_hypo) / peak_real * 100

    plt.figure(figsize=(10, 6))

    plt.plot(t, sol_real[:, 1], 'r-', linewidth=2, label=f'Scenario 2017')
    plt.plot(t, sol_hypo[:, 1], 'g--', linewidth=2, label=f'Scenario Post-COVID')


    plt.title(f"Impatto campagna vaccinale Post-COVID sul 2017\nRiduzione del Picco: {reduction:.1f}%")

    # Etichette mesi
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)

    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis_counterfactual_covid_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Analisi Controfattuale salvata. Riduzione picco stimata: {reduction:.1f}%")


# --- ESECUZIONE DEL CODICE ---

df_clean = clean_flu_data(df_flu)
t = np.linspace(0, 5, 100)

print("Inizio Analisi...")

# 1. Recupero dati Stagione 2017-2018 (Il nostro Caso Studio)
season_17 = get_flu_season(df_clean, 2017)
vax_17 = get_vax_rate(df_vax, 2017)

print(season_17)

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

plot_sir_dynamics(t, sol_17, 2017)

plot_counterfactual_2021_rates(t, beta, gamma, r, n_stages, I0, season_17)

print("Tutto completato. Grafici salvati.")