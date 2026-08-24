import numpy as np
import xarray as xr
import pandas as pd
import os

#Chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'


SWIO = (32, 50, -28, -12)

seuil_panache = 34.0 # seuil definissant le panache
salinite_ambiante = 35.0 # valeur de reference (au large)

sortie_csv = 'gradient_salinite_annuel.csv'

# Grilles et donnees

g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
mask = g['mask_rho'].values
pm = g['pm'].values
pn = g['pn'].values
g.close()

d = xr.open_dataset(os.path.join(simu, file))
salt = d.salt
if 's_rho' in salt.dims:
   salt = salt.isel(s_rho=-1)

fill_value = 9.96921e+36
salt = salt.where((salt != fill_value), np.nan)
time = salt.time
d.close()

n_days = len(time)
print(f"{n_days} jours a analyser.")

dates = []
grad_min = []
grad_max = []
grad_moy = []
lon_grad_max = []
lat_grad_max = []
amplitudes = []
long_car = []

for date in range (n_days):
    champ = salt.isel(time=date).values
    champ = np.where(mask == 1, champ, np.nan) # selection des points en mer
    masque_zone = (lon >= SWIO[0]) & (lon <= SWIO[1]) & (lat >= SWIO[2]) & (lat <= SWIO[3])
    champ = np.where(masque_zone, champ, np.nan) # selection des points dans le cadre SWIO considéré

    # Calcul du gradient (d'abord par indice, puis conversion pour l'avoir en psu/m)
    dS_dx = np.gradient(champ, axis=1) * pm
    dS_dy = np.gradient(champ, axis=0) * pn

    gradient_magnitude = np.sqrt(dS_dx**2 + dS_dy**2) # psu/m
    gradient_magnitude_km = gradient_magnitude * 1000 # psu/km
    gradient_magnitude_km = np.where(mask == 1, gradient_magnitude_km, np.nan) #au cas ou un point terre est entoure de points mer

    masque_panache = champ <= seuil_panache # masque du panache pour les calculs suivants

    if np.any(masque_panache):
        grad_min_jour = np.nanmin(gradient_magnitude_km[masque_panache])
        grad_moy_jour = np.nanmean(gradient_magnitude_km[masque_panache])
        grad_max_jour = np.nanmax(gradient_magnitude_km[masque_panache])

        # Position du max
        gradient_dans_panache = np.where(masque_panache, gradient_magnitude_km, np.nan)
        idx_max = np.unravel_index(np.nanargmax(gradient_dans_panache), gradient_dans_panache.shape)
        j_max, i_max = idx_max
        lon_grad_max_day = lon[j_max, i_max]
        lat_grad_max_day = lat[j_max, i_max]

        amplitude_panache = np.nanmax(salinite_ambiante - champ[masque_panache])
    else :
        grad_min_jour = np.nan
        grad_moy_jour = np.nan
        grad_max_jour = np.nan
        lon_grad_max_day = np.nan
        lat_grad_max_day = np.nan
        amplitude_panache = np.nan

    if grad_moy_jour and grad_moy_jour > 0:
        longueur_caracteristique_km = amplitude_panache / grad_moy_jour
    else:
        longueur_caracteristique_km = np.nan

    dates.append(str(time.values[date])[:10])
    grad_min.append(grad_min_jour)
    grad_moy.append(grad_moy_jour)
    grad_max.append(grad_max_jour)
    amplitudes.append(amplitude_panache)
    long_car.append(longueur_caracteristique_km)
    lon_grad_max.append(lon_grad_max_day)
    lat_grad_max.append(lat_grad_max_day)


# Statistiques sur l'annee

def stats_et_dates(valeurs, dates, nom):
    valeurs = np.array(valeurs, dtype=float)
    moyenne = np.nanmean(valeurs)
    idx_max_val = np.nanargmax(valeurs)
    idx_min_val = np.nanargmin(valeurs)

    print(f"\n ------ {nom} ------")
    print(f" Moyenne sur l'annee {moyenne:.4f}")
    print(f" Maximum de l'annee : {valeurs[idx_max_val]:.4f} (le {dates[idx_max_val]})")
    print(f" Minimum de l'annee : {valeurs[idx_min_val]:.4f} (le {dates[idx_min_val]})")

    return moyenne, valeurs[idx_max_val], dates[idx_max_val], valeurs[idx_min_val], dates[idx_min_val]

# Affichage des stats

print("\n" + "=" * 50)
print("Statistiques annuelles")
print("=" * 50)

stats_et_dates(grad_min, dates, "Gradient minimum, zone du panache [psu/km]")
stats_et_dates(grad_moy, dates, "Gradient moyen, zone du panache [psu/km]")
stats_et_dates(grad_max, dates, "Gradient maximum, zone du panache [psu/km]")
stats_et_dates(amplitudes, dates, "Amplitude de la variation de salinite [psu]")
stats_et_dates(long_car, dates, "Longueur caracteristique [km]")
# Creation du tableau csv

tableau = pd.DataFrame({
    'date': dates,
    'gradient_min_psu_km': grad_min,
    'gradient_moyen_psu_km': grad_moy,
    'gradient_max_psu_km': grad_max,
    'lon_gradient_max': lon_grad_max,
    'lat_gradient_max': lat_grad_max,
    'amplitude_panache_psu': amplitudes,
    'longueur_caracteristique_km': long_car,
})

tableau.to_csv(sortie_csv, index=False)
print(f"\nTableau exporte en CSV : {sortie_csv}")
                                                  
