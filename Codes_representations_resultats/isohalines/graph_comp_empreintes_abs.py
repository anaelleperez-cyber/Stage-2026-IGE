import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

#Paths 
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
file_1psu = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_1PSU_25deg/001swiose_avg.nc'
file_15psu = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_15PSU_25deg/001swiose_avg.nc'

seuil_panache = 35 #en psu, critère de définition du panache 
fill_value = 9.96921e+36

#chargement grille 
g = xr.open_dataset(grid)
pm = g['pm'].values
pn = g['pn'].values
lon = g['lon_rho'].values
lat = g['lat_rho'].values
mask_rho = g['mask_rho'].values
g.close()

#calcul de la surface de chaque cellule 
cell_area_km2 = 1.0 / (pm*pn) / 1e6
print (f"La surface totale de la grille est : {np.sum(cell_area_km2):,.0f} km2 ")

#masque geographique
SWIO = (33, 43, -26.5, -16)
masque_zone = (lon >= SWIO[0]) & (lon <= SWIO[1]) & (lat >= SWIO[2]) & (lat <= SWIO[3])


def calculer_empreinte(diff_file, nom):
    """Calcule l'empreinte du panache (km2) pour chaque jour d'un fichier de difference."""
    d = xr.open_dataset(diff_file)
    salt = d.salt
    if 's_rho' in salt.dims:
        salt = salt.isel(s_rho=-1)
    salt = salt.where((salt != fill_value), np.nan)
    time = salt.time
    d.close()

    n_days = len(time)
    print(f"[{nom}] {n_days} jours a analyser")

    dates = []
    surfaces_km2 = []

    for t in range(n_days):
        surface_salinity = salt.isel(time=t).values
        surface_salinity = np.where(mask_rho == 1, surface_salinity, np.nan) #exclusion explicite de la terre
        masque_panache = (surface_salinity <= seuil_panache) & masque_zone
        surface_jour = np.sum(cell_area_km2[masque_panache])
        date_str = str(time.values[t])[:10]
        dates.append(date_str)
        surfaces_km2.append(surface_jour)

    print(f"[{nom}] Empreinte moyenne sur la periode : {np.mean(surfaces_km2):,.1f} km2")
    print(f"[{nom}] Empreinte maximale : {np.max(surfaces_km2):,.1f} km2 (le {dates[np.argmax(surfaces_km2)]})")
    print(f"[{nom}] Empreinte minimale : {np.min(surfaces_km2):,.1f} km2 (le {dates[np.argmin(surfaces_km2)]})\n")

    return dates, surfaces_km2

#calcul pour les deux fichiers
dates_1psu, surfaces_1psu = calculer_empreinte(file_1psu, "1 PSU")
dates_15psu, surfaces_15psu = calculer_empreinte(file_15psu, "15 PSU")

#figure
plt.style.use('seaborn-v0_8-whitegrid')   # style global de la figure

fig, ax = plt.subplots(figsize=(13, 5.5))

# Aire remplie sous chaque courbe
ax.fill_between(dates_1psu, surfaces_1psu, alpha=0.15, color='#1f77b4')
ax.fill_between(dates_15psu, surfaces_15psu, alpha=0.15, color='#ff7f0e')

# Courbes
ax.plot(dates_1psu, surfaces_1psu, linewidth=2, marker='o', markersize=4,
        color='#1f77b4', label='Zambèze - 1 PSU')
ax.plot(dates_15psu, surfaces_15psu, linewidth=2, marker='o', markersize=4,
        color='#ff7f0e', label='Zambèze - 15 PSU')

# Marquages du maximum de chaque courbe
idx_max_1 = np.argmax(surfaces_1psu)
idx_max_15 = np.argmax(surfaces_15psu)
ax.annotate(f'{surfaces_1psu[idx_max_1]:,.0f} km²\n{dates_1psu[idx_max_1]}',
            xy=(idx_max_1, surfaces_1psu[idx_max_1]),
            xytext=(60, -5), textcoords='offset points',
            ha='center', fontsize=9, color='#1f77b4', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#1f77b4', linewidth=1.5),
            arrowprops=dict(arrowstyle='-', color='#1f77b4', lw=1))
ax.annotate(f'{surfaces_15psu[idx_max_15]:,.0f} km²\n{dates_15psu[idx_max_15]}',
            xy=(idx_max_15, surfaces_15psu[idx_max_15]),
            xytext=(-60, 5), textcoords='offset points',
            ha='center', fontsize=9, color='#ff7f0e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#ff7f0e', linewidth=1.5),
            arrowprops=dict(arrowstyle='-', color='#ff7f0e', lw=1))

# Titres et labels
ax.set_ylabel(f"Empreinte du panache [km²]\n(SSS ≤ {seuil_panache} psu)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title("Évolution temporelle de l'empreinte spatiale du panache du Zambèze",
             fontsize=13, fontweight='bold', pad=15)

# Legende 
ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc='upper right')

# Suppression des bordures
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grille horizontale seulement
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.grid(False, axis='x')

n_ticks = min(15, len(dates_1psu))
step = max(1, len(dates_1psu) // n_ticks)
ax.set_xticks(range(0, len(dates_1psu), step))
ax.set_xticklabels([dates_1psu[k] for k in range(0, len(dates_1psu), step)],
                     rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelsize=9)

fig.tight_layout()
plt.show()
fig.savefig('empreinte_panache_Zambeze_1psu_15psu.png', dpi=300)

                                                
