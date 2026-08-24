import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CHEMINS ET PARAMETRES
# ---------------------------------------------------------------
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'

latitude_coupe = -11.0     # latitude le long de laquelle on coupe
lon_min, lon_max = 40, 51  # bornes de longitude affichees

# Parametres de la coordonnee sigma (rappel de croco.in)
theta_s = 8.0
theta_b = 2.0
hc = 100.0
N = 50   # nombre de niveaux s_rho

# ---------------------------------------------------------------
# CHARGEMENT DE LA GRILLE
# ---------------------------------------------------------------
g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
h = g['h'].values
mask_rho = g['mask_rho'].values
g.close()

# ---------------------------------------------------------------
# FONCTION DE TRANSFORMATION SIGMA -> PROFONDEUR REELLE
# (formulation CROCO/ROMS standard, Vtransform=2, Vstretching=4)
# ---------------------------------------------------------------
def stretching(sc, theta_s, theta_b):
    if theta_s > 0:
        csrf = (1 - np.cosh(theta_s * sc)) / (np.cosh(theta_s) - 1)
    else:
        csrf = -sc**2
    if theta_b > 0:
        Cs = (np.exp(theta_b * csrf) - 1) / (1 - np.exp(-theta_b))
    else:
        Cs = csrf
    return Cs

def zlevs(h, theta_s, theta_b, hc, N):
    """Calcule la profondeur reelle z (m, negative) de chaque niveau s_rho,
    pour un vecteur/tableau de bathymetrie h. zeta (surface libre) est
    suppose nul (pas de forcage instantane, juste la geometrie de la grille)."""
    ds = 1.0 / N
    sc = ds * (np.arange(1, N + 1) - N - 0.5)   # niveaux rho, centres des couches
    Cs = stretching(sc, theta_s, theta_b)

    z = np.zeros((N,) + h.shape)
    for k in range(N):
        z0 = (hc * sc[k] + h * Cs[k]) / (hc + h)
        z[k] = h * z0   # zeta = 0
    return z

# ---------------------------------------------------------------
# EXTRACTION DE LA COUPE, A LA LATITUDE DEMANDEE
# ---------------------------------------------------------------
# Indice de ligne (eta) le plus proche de la latitude voulue, sur la colonne centrale
j_coupe = np.argmin(np.abs(lat[:, lat.shape[1] // 2] - latitude_coupe))

lon_coupe = lon[j_coupe, :]
h_coupe = h[j_coupe, :]
mask_coupe = mask_rho[j_coupe, :]

# Restriction a la fenetre de longitude voulue
sel = (lon_coupe >= lon_min) & (lon_coupe <= lon_max)
lon_coupe = lon_coupe[sel]
h_coupe = h_coupe[sel]
mask_coupe = mask_coupe[sel]

# Calcul des profondeurs de chaque niveau sigma, le long de cette coupe
z_coupe = zlevs(h_coupe, theta_s, theta_b, hc, N)   # shape (N, n_lon)

# Bathymetrie affichee en negatif (fond de l'ocean)
bathy_coupe = np.where(mask_coupe == 1, -h_coupe, 0)

# ---------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

# Remplissage gris = terre/fond marin (sous la bathymetrie)
ax.fill_between(lon_coupe, bathy_coupe, -np.nanmax(h_coupe) * 1.05,
                  color='lightgray', zorder=1)

# Chaque niveau sigma, en fine ligne grise
for k in range(N):
    ax.plot(lon_coupe, np.where(mask_coupe == 1, z_coupe[k], np.nan),
            color='gray', linewidth=0.3, zorder=2)

# Le fond bathymetrique, en trait noir epais
ax.plot(lon_coupe, bathy_coupe, color='black', linewidth=1.2, zorder=3)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(-np.nanmax(h_coupe) * 1.05, 0)
ax.set_xlabel(f"Longitudes le long de la latitude {latitude_coupe:.1f}°")
ax.set_ylabel("Depth [m]")

fig.tight_layout()
fig.savefig('coupe_sigma.png', dpi=300)
print("Figure sauvegardee : coupe_sigma.png")
plt.show()
