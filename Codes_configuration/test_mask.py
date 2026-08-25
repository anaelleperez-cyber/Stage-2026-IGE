#Code permettant de tester si les 12 PSOURCES sélectionnés sont effectivement à l'interface terre-mer (double vérification par rapport à psources_candidates.py)

import xarray as xr
import numpy as np

grid_file = '/lus/scratch/CT1/c1601279/aperez/SWIO/test_avec_psource/swiose_grid.nc'

# Points sources a tester : (nom, Isrc, Jsrc, Dsrc)
sources = [
    ("Zambeze", 96, 200, 0),
    ("Tsiribihinha", 188, 192, 0),
    ("Bestiboka", 208, 240, 1),
    ("Mangoki", 178, 171, 0),
    ("Manambolo", 187, 197, 0),
    ("Mahavy", 238, 272, 1),
    ("Sambriano", 233, 265, 0),
    ("Limpopo", 70, 119, 1),
    ("Save", 83, 173, 0),
    ("Buzi et Pungoe", 79, 185, 0),
    ("Morondova", 186, 184, 0),
    ("Linta", 185, 128, 0)
]

# Chargelent du masque
g = xr.open_dataset(grid_file)
mask = g['mask_rho'].values  
g.close()

print(f"Dimensions de la grille (eta_rho, xi_rho) : {mask.shape}")
print()

# Fonction de vérification 

def check_source(name, Isrc, Jsrc, Dsrc, mask):
    print(f"--- {name} : Isrc={Isrc}, Jsrc={Jsrc}, Dsrc={Dsrc} ---")

    nJ, nI = mask.shape

    if Dsrc == 0:
        # Face u : entre le point rho (Isrc-1, Jsrc) et (Isrc, Jsrc)
        if Isrc - 1 < 0 or Isrc >= nI or Jsrc >= nJ:
            print("  ERREUR : indices hors de la grille.")
            return
        m1 = mask[Jsrc, Isrc - 1]
        m2 = mask[Jsrc, Isrc]
        print(f"  Point rho gauche (I={Isrc-1}, J={Jsrc}) : mask={m1} ({'mer' if m1==1 else 'terre'})")
        print(f"  Point rho droite (I={Isrc},   J={Jsrc}) : mask={m2} ({'mer' if m2==1 else 'terre'})")

    elif Dsrc == 1:
        # Face v : entre le point rho (Isrc, Jsrc-1) et (Isrc, Jsrc)
        if Jsrc - 1 < 0 or Jsrc >= nJ or Isrc >= nI:
            print("  ERREUR : indices hors de la grille.")
            return
        m1 = mask[Jsrc - 1, Isrc]
        m2 = mask[Jsrc, Isrc]
        print(f"  Point rho bas  (I={Isrc}, J={Jsrc-1}) : mask={m1} ({'mer' if m1==1 else 'terre'})")
        print(f"  Point rho haut (I={Isrc}, J={Jsrc})   : mask={m2} ({'mer' if m2==1 else 'terre'})")

    else:
        print(f"  Dsrc={Dsrc} non reconnu (doit etre 0 ou 1).")
        return

    if m1 != m2:
        print("  --> PSOURCE VALIDE : un cote est terre, l'autre est mer.")
    elif m1 == 1 and m2 == 1:
        print("  --> PSOURCE NON VALIDE : les deux cotes sont en MER.")
    else:
        print("  --> ATTENTION : les deux cotes sont en TERRE. Aucune eau")
    print()


for name, Isrc, Jsrc, Dsrc in sources:
    check_source(name, Isrc, Jsrc, Dsrc, mask)
