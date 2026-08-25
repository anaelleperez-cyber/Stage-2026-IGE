#Pour une rivière/un fleuve donné, code qui permet de lister les n points à l'interface terre-mer les plus proches de l'embouchure 

import csv
import numpy as np

input_csv = '/lus/home/CT1/c1601279/aperez/SCRIPTS_ANAELLE/psource_candidates.csv'

# Coordonnees approximatives de l'embouchure de la riviere
lon = 44.07
lat = -25.03

# Rayon de recherche autour de l'embouchure, en degres
search_radius_deg = 0.5

n_closest = 4 # nombre de points les plus proches a afficher

# Lecture du CSV 

candidates = []
with open(input_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['Isrc'] = int(row['Isrc'])
        row['Jsrc'] = int(row['Jsrc'])
        row['Dsrc'] = int(row['Dsrc'])
        row['lon'] = float(row['lon'])
        row['lat'] = float(row['lat'])
        candidates.append(row)

print(f"{len(candidates)} points charges depuis {input_csv}")

# Calcul de la distance à l'embouchure 
for c in candidates:
    dlon = c['lon'] - lon
    dlat = c['lat'] - lat
    c['distance_deg'] = np.sqrt(dlon**2 + dlat**2)

# filtrage : ne garder que les points dans le rayon de recherche
near_river = [c for c in candidates if c['distance_deg'] <= search_radius_deg]

print(f"{len(near_river)} points trouves dans un rayon de {search_radius_deg} degre "
      f"autour de l'embouchure de la riviere ({lon}, {lat}).")

# Tri par distante croissante (le plus proche en premier)
near_river.sort(key=lambda c: c['distance_deg'])

# Affichage des n plus proches
print(f"\nLes {min(n_closest, len(near_river))} points les plus proches :")
print(f"{'Isrc':>6} {'Jsrc':>6} {'Dsrc':>6} {'lon':>10} {'lat':>10} {'dist(deg)':>10}  cote terre") #en-tete du tableau

for c in near_river[:n_closest]:
    print(f"{c['Isrc']:>6} {c['Jsrc']:>6} {c['Dsrc']:>6} "
          f"{c['lon']:>10.3f} {c['lat']:>10.3f} {c['distance_deg']:>10.3f}  {c['side_terre']}")


