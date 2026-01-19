import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Caricamento dati
df_flu = pd.read_csv('flunet.csv')
df_vax = pd.read_csv('vaccini.csv', sep=';')

# Parametri globali
n_stages = 3
g = 4
r = 0.5
b = 8
I0 = 0.001

s_min = 0.01
s_max = 0.7

t_vax = 2 #durata della campagna attiva vaccinale
pop_ita = 60e6


#FUNZIONI

#Funzione per prendere i dati sui contagi italiani e aggregarli mensilmente
def clean_flu_data(df_flu_clean):
    italy_df = df_flu_clean[df_flu_clean['COUNTRY_AREA_TERRITORY'] == 'Italy'].copy()
    italy_df['ISO_WEEKSTARTDATE'] = pd.to_datetime(italy_df['ISO_WEEKSTARTDATE'])

    italy_df['Year'] = italy_df['ISO_WEEKSTARTDATE'].dt.year
    italy_df['Month'] = italy_df['ISO_WEEKSTARTDATE'].dt.month

    monthly_flu = italy_df.groupby(['Year', 'Month'])['INF_ALL'].sum().reset_index()
    return monthly_flu

#Funzione per estrarre i contagi da Novembre ad Aprile
def get_flu_season(df, year_start):
    season = df[((df['Year'] == year_start) & (df['Month'] >= 11)) | \
                ((df['Year'] == year_start + 1) & (df['Month'] <= 4))].copy()
    season = season.sort_values(['Year', 'Month'])

    # numero mese nella stagione
    season['Num'] = np.arange(len(season))
    return season

#Funzione per estrarre il tasso di vaccinazioni di un anno specifico
def get_vax_rate(df_vax_clean, year_start):
    anno = f"{year_start}-{str(year_start + 1)[2:]}"

    riga = df_vax_clean[df_vax['anno'] == anno]
    if not riga.empty:
        return riga['tasso'].values[0] / 100
    return 0

#Funzione che definisce il SIR model standard
def sir_standard(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

#Funzione che definisce il SIRnS model
def sirns_model(y, t, beta, gamma, r, vax_rate, n):
    S = y[0]
    I = y[1]
    R = y[2:]

    #creo n valori distribuiti tra la suscettibilità minima e la suscettibilità massima,
    #in base al numero di stage n
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

    #Equazion differenziali
    dsdt = -beta * S * I - v * S + r * R[n - 1]
    didt = beta * S * I + infezioniR - gamma * I
    drdt = np.zeros(n)
    # R1 riceve i guariti e i vaccinati
    drdt[0] = (gamma * I) + (v * S) - (s[0] * beta * R[0] * I) - (r * R[0])
    # Passaggio tra stadi R_i a R_{i+1}
    for i in range(1, n):
        drdt[i] = (r * R[i - 1]) - (s[i] * beta * R[i] * I) - (r * R[i])

    ris = [dsdt, didt] + drdt.tolist()
    return ris

# Simulazione modello SIR standard
def run_sir_standard(beta, gamma, I0, t_grid):
    S0 = 1 - I0
    R0 = 0.0
    y0 = [S0, I0, R0]

    sol = odeint(sir_standard, y0, t_grid, args=(beta, gamma))
    return sol


def run_simulation(beta, vax_rate_target, n_stages, t_grid, gamma, r, I0):
    # Il 15% dei vaccini viene fatto a Ottobre
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

    sol = odeint(sirns_model, y0, t_grid, args=(beta, gamma, r, vax_rate, n_stages))
    return sol

#GRAFICI

#Modello SIRnV in confronto ai dati reali
def plot_sirnv_realdata(t, sol, real_data, year, vax_rate, filename, title_suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(t, sol[:, 1], 'r-', linewidth=2, label='Modello SIRnS (Infetti)')
    # Scalatura dati reali sul picco della simulazione
    if not real_data.empty:
        peak_model = sol[:, 1].max()
        peak = real_data['INF_ALL'] .max()

        factor = peak_model / peak
        real_scaled = real_data['INF_ALL'] * factor
        plt.scatter(real_data['Num'], real_scaled, color='blue', label='Dati Reali (Scalati)')

    plt.title(f"Stagione {year}-{year + 1} (Vax: {vax_rate * 100:.1f}%) {title_suffix}")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")

#Confronto di vari tassi di decadimento dell'immunità
def plot_sensitivity_r(t, beta, gamma, vax_rate, n_stages, I0):
    r_values = [0.2, 0.5, 0.8]
    labels = ['Immunità Lunga (r=0.2)', 'Immunità Media (r=0.5)', 'Immunità Breve (r=0.8)']
    plt.figure(figsize=(10, 6))

    for i, r_val in enumerate(r_values):
        sol = np.array(run_simulation(beta, vax_rate, n_stages, t, gamma, r_val, I0))
        plt.plot(t, sol[:, 1], label=labels[i], linewidth=2)

    plt.title("Effetto del decadimento dell'immunità (r)")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sensitività.png", dpi=300, bbox_inches="tight")

#Contronto tra vari tassi di vaccinazione
def plot_vaccine_scenarios(t, beta, gamma, r, n_stages, vax_real, I0):
    vax_scenarios = [0.0, vax_real, 0.30]
    labels = ['Nessun Vaccino (0%)', f'Reale 2017 ({vax_real * 100:.1f}%)', 'Target Ottimale (30%)']
    plt.figure(figsize=(10, 6))

    for i, v_rate in enumerate(vax_scenarios):
        sol = np.array(run_simulation(beta, v_rate, n_stages, t, gamma, r, I0))
        plt.plot(t, sol[:, 1], linewidth=2, label=f'{labels[i]}')

    plt.title("Impatto della Copertura Vaccinale")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vaccinazioni.png", dpi=300, bbox_inches="tight")

#Variazione del valore Rt nel tempo
def plot_rt(t, sol, beta, gamma, n_stages):
    S_t = sol[:, 0]
    R_t = sol[:, 2:]
    s = np.linspace(s_min, s_max, n_stages)

    # Calcolo il numero di riproduzione (rip) nel tempo
    susceptibility = S_t + sum(s[i] * R_t[:, i] for i in range(n_stages))
    rip_base = beta / gamma
    rip_t =  rip_base * susceptibility

    plt.figure(figsize=(10, 6))
    plt.plot(t, rip_t, color='purple', linewidth=2, label='Rip Effettivo')
    plt.axhline(1.0, color='black', linestyle='--', label='Soglia Rip=1')
    plt.title("Andamento del Numero di Riproduzione nel tempo")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Valore Rip")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analisi_rip.png", dpi=300, bbox_inches="tight")

#Grafico andamento di S, I e R nel tempo
def plot_dynamics(t, sol, year):
    S = sol[:, 0]
    I = sol[:, 1]
    # Sommiamo tutte gli stadi di R (R0, R1, R2...)
    R_total = np.sum(sol[:, 2:], axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, color='blue', linewidth=2.5, label='Suscettibili (S)')
    plt.plot(t, R_total, color='green', linewidth=2.5, label='Immuni (R_tot)')
    plt.plot(t, I, color='red', linewidth=2.5, label='Infetti (I)')

    plt.title(f"Stagione {year}-{year + 1}")
    plt.ylabel("Popolazione")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center right', fontsize=11, framealpha=1, shadow=True)
    plt.savefig("SIRnV_fasi.png", dpi=300, bbox_inches="tight")

#Compara la curva del modello con i dati vaccinali del 2017 e quelli del 2021
def plot_comparison_2021_rates(t, beta, gamma, r, n_stages, I0, season_data):
    rate_2017 = get_vax_rate(df_vax, 2017)

    other_year = 2021
    rate_other = get_vax_rate(df_vax, other_year)

    # Eseguiamo le due simulazioni
    sol_real = np.array(run_simulation(beta, rate_2017, n_stages, t, gamma, r, I0))
    sol_hypo = np.array(run_simulation(beta, rate_other, n_stages, t, gamma, r, I0))

    peak_real = sol_real[:, 1].max()
    peak_hypo = sol_hypo[:, 1].max()
    reduction = (peak_real - peak_hypo) / peak_real * 100
    print(f"Riduzione del picco usando dati vaccinali 2021 {reduction}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, sol_real[:, 1], 'r-', linewidth=2, label=f'Scenario 2017')
    plt.plot(t, sol_hypo[:, 1], 'g--', linewidth=2, label=f'Scenario Post-COVID')
    plt.title(f"Impatto campagna vaccinale Post-COVID sul 2017")
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vaccini_postCovid.png", dpi=300, bbox_inches="tight")

#Comparazione dell'andamento della curva degli infetti usando il modello SIR e il SIRnS
#scalo i dati in modo da confrontare la forma dell'andamento
def plot_sir_comparison(t, sol_sir, sol_sirns, real_data):
    plt.figure(figsize=(10, 6))
    I_sir = sol_sir[:, 1]
    I_sirns = sol_sirns[:, 1]

    I_sir_norm = I_sir / I_sir.max()
    I_sirns_norm = I_sirns / I_sirns.max()

    if not real_data.empty:
        real_norm = real_data['INF_ALL'] / real_data['INF_ALL'].max()
        plt.scatter(real_data['Num'], real_norm, color='blue', label='Dati Reali (Scalati)')

    plt.plot(t, I_sir_norm, '--', linewidth=2.5, label='SIR standard')
    plt.plot(t, I_sirns_norm, '-', linewidth=2.5, label='SIRnS (immunità graduale)')
    mesi_labels = ['Nov', 'Dic', 'Gen', 'Feb', 'Mar', 'Apr']
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=mesi_labels)
    plt.ylabel("Proporzione Infetti")
    plt.title("Confronto tra SIR standard e SIRnS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("SIR_vs_SIRnS.png", dpi=300, bbox_inches="tight")

#ESECUZIONE
df_clean = clean_flu_data(df_flu)
t = np.linspace(0, 5, 100)

print("Inizio Analisi...")

#Recupero dei dati Stagione 2017-2018
season_17 = get_flu_season(df_clean, 2017)
vax_17 = get_vax_rate(df_vax, 2017)
print(season_17)

# Simulazione del modello SIRnS
print(f"Simulazione Stagione 2017 (Vax Rate: {vax_17:.3f})...")
sol_17 = run_simulation(b, vax_17, n_stages, t, g, r, I0)
plot_sirnv_realdata(t, sol_17, season_17, 2017, vax_17, "sirnv_2017_2018.png")
plot_dynamics(t, sol_17, 2017)

# Simulazione del modello SIR standard
print("Simulazione SIR standard (baseline)...")
sol_sir = run_sir_standard(b, g, I0, t)

# Analisi della sensibilità dei parametri
print("Generazione Analisi di Sensitività (r)...")
plot_sensitivity_r(t, b, g, vax_17, n_stages, I0)

print("Generazione Scenari Vaccinali...")
plot_vaccine_scenarios(t, b, g, r, n_stages, vax_17, I0)
plot_comparison_2021_rates(t, b, g, r, n_stages, I0, season_17)

print("Calcolo e Grafico Rt...")
plot_rt(t, sol_17, b, g, n_stages)

#Confronto tra SIR e SIRnS sui dati reali
print("Confronto SIR standard vs SIRnS...")
plot_sir_comparison(t, sol_sir, sol_17, season_17)

