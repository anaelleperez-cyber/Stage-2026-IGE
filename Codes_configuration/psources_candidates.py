# Code qui permet de sortir une liste CSV (utilisée dans le code river_psource.py) de tous les points de grille à l'interface terre-mer

import xarray as xr
import numpy as np
import csv

# Chemin vers la grille
grid_file = '/lus/scratch/CT1/c1601279/aperez/SWIO/test_avec_psource/swiose_grid.nc'


# Si on veut restreindre à unz zone géograpghique
lon_min, lon_max = None, None
lat_min, lat_max = None, None

output_csv = 'psource_candidates.csv'

# Chargement de la grille 
g = xr.open_dataset(grid_file)
mask = g['mask_rho'].values  
lon = g['lon_rho'].values
lat = g['lat_rho'].values
g.close()

nJ, nI = mask.shape
print(f"Dimensions de la grille (eta_rho, xi_rho) : {mask.shape}")

# Détection des faces u (cas Dsrc=0)
candidates = []

for j in range(nJ):
    for i in range(1, nI):
        m1 = mask[j, i - 1]
        m2 = mask[j, i]
        if m1 != m2:
            lon_pt = lon[j, i]
            lat_pt = lat[j, i]

            if lon_min is not None and not (lon_min <= lon_pt <= lon_max):
                continue
            if lat_min is not None and not (lat_min <= lat_pt <= lat_max):
                continue

            candidates.append({
                'Isrc': i, 'Jsrc': j, 'Dsrc': 0,
                'lon': lon_pt, 'lat': lat_pt,
                'side_terre': 'gauche' if m1 == 0 else 'droite'
            })

# Détection des faces v (cas Dsrc=1)
for j in range(1, nJ):
    for i in range(nI):
        m1 = mask[j - 1, i]
        m2 = mask[j, i]
        if m1 != m2:
            lon_pt = lon[j, i]
            lat_pt = lat[j, i]

            if lon_min is not None and not (lon_min <= lon_pt <= lon_max):
                continue
            if lat_min is not None and not (lat_min <= lat_pt <= lat_max):
                continue

            candidates.append({
                'Isrc': i, 'Jsrc': j, 'Dsrc': 1,
                'lon': lon_pt, 'lat': lat_pt,
                'side_terre': 'bas' if m1 == 0 else 'haut'
            })

print(f"\n{len(candidates)} points d'interface terre-mer trouves.")

# Sauvegarde en csv
if candidates:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Isrc', 'Jsrc', 'Dsrc', 'lon', 'lat', 'side_terre'])
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Liste complete sauvegardee dans : {output_csv}")

# Affichage des resultats 

print("\nApercu des 10 premiers points trouves :")
print(f"{'Isrc':>6} {'Jsrc':>6} {'Dsrc':>6} {'lon':>10} {'lat':>10}  cote terre") #en-tete du tableau de sortie
for c in candidates[:]:
    print(f"{c['Isrc']:>6} {c['Jsrc']:>6} {c['Dsrc']:>6} {c['lon']:>10.3f} {c['lat']:>10.3f}  {c['side_terre']}")
else:
    print("Aucun point trouve. Verifiez le fichier de grille ou la zone de filtrage (lon_min/lat_min...).")
