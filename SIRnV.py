import sys

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime


df_flu = pd.read_csv('flunet.csv')
df_vax = pd.read_csv('vaccini.csv')

def clean_flu_data(df):
    italy_df = df[df['COUNTRY_AREA_TERRITORY'] == 'Italy'].copy()
    italy_df['ISO_WEEKSTARTDATE'] = pd.to_datetime(italy_df['ISO_WEEKSTARTDATE'])

    # crea colonne Year e Month
    italy_df['Year'] = italy_df['ISO_WEEKSTARTDATE'].dt.year
    italy_df['Month'] = italy_df['ISO_WEEKSTARTDATE'].dt.month

    # Aggrega mensilmente
    monthly_flu = italy_df.groupby(['Year', 'Month'])['INF_ALL'].sum().reset_index()
    return monthly_flu

def get_flu_season(df, year_start):
    season = df[((df['Year'] == year_start) & (df['Month'] >= 11)) | \
           ((df['Year'] == year_start + 1) & (df['Month'] <= 4))].copy()
    season = season.sort_values(['Year', 'Month'])

    # numero mese nella stagione
    season['Num'] = np.arange(len(season)) + 1
    return season

def sirnv_model(y, t, params):
    S = y[0]
    I = y[1]
    R = y[2:]
    n = len(R)




df_clean = clean_flu_data(df_flu)

season_17_18 = get_flu_season(df_clean, 2017)
season_18_19 = get_flu_season(df_clean, 2018)

print("Dati Stagione 2017-18 pronti!")
print(season_17_18)




