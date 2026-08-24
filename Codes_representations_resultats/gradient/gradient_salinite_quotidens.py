import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import os

# Chemins
grid = '/lus/store/CT1/c1601279/aperez/resultats/ref_1an_sans_psource/swiose_grid.nc'
simu = '/lus/store/CT1/c1601279/aperez/resultats/exp_1PSOURCE_1PSU_25deg/'
file = '001swiose_avg.nc'
date = '2017-02-11'

SWIO = (33, 43, -26.5, -16)

seuil_panache = 34.0 # seuil definissant le panache
salinite_ambiante = 35.0 # valeur de reference (au large)

#Grille et données

g = xr.open_dataset(grid)
lon = g['lon_rho'].values
lat = g['lat_rho'].values
pm = g['pm'].values
pn = g['pn'].values
mask_rho = g['mask_rho'].values   # 0 = terre, 1 = mer
g.close()

d = xr.open_dataset(os.path.join(simu, file))
salt = d.salt
if 's_rho' in salt.dims:
    salt = salt.isel(s_rho=-1)

fill_value = 9.96921e+36
salt = salt.where((salt != fill_value), np.nan)
time = salt.time
d.close()

if np.datetime64(date) not in time.values.astype('datetime64[D]'):
    date = np.datetime_as_string(time.values.astype('datetime64[D]')[0])
    print(f'Date non trouvee, utilisation de {date} a la place.')

champ = salt.sel(time=date).mean(dim='time').values
champ = np.where(mask_rho == 1, champ, np.nan) # selection des points en mer 
masque_zone = (lon >= SWIO[0]) & (lon <= SWIO[1]) & (lat >= SWIO[2]) & (lat <= SWIO[3])
champ = np.where(masque_zone, champ, np.nan) # selection des points dans le cadre SWIO considéré

# Calcul du gradient (d'abord par indice, puis conversion pour l'avoir en psu/m)
dS_dx = np.gradient(champ, axis=1) * pm
dS_dy = np.gradient(champ, axis=0) * pn

gradient_magnitude = np.sqrt(dS_dx**2 + dS_dy**2) # psu/m
gradient_magnitude_km = gradient_magnitude * 1000 # psu/km

gradient_magnitude_km = np.where(mask_rho == 1, gradient_magnitude_km, np.nan) #au cas ou un point terre est entoure de points mer

print(f"Gradient max : {np.nanmax(gradient_magnitude_km):.4e} psu/km")
print(f"Gradient moyen (tout le domaine) : {np.nanmean(gradient_magnitude_km):.4e} psu/km")

# Calcul de la longueur caractéristique dans la zone du panache
# L_car = amplitude / gradient_moyen
masque_panache = champ <= seuil_panache # calcul dans la zone du panache (terre deja exclue via champ)

amplitude_panache = np.nanmax(salinite_ambiante - champ[masque_panache]) if np.any(masque_panache) else np.nan
gradient_moyen_panache = np.nanmean(gradient_magnitude_km[masque_panache]) if np.any(masque_panache) else np.nan

if gradient_moyen_panache and gradient_moyen_panache > 0:
    longueur_caracteristique_km = amplitude_panache / gradient_moyen_panache
else:
    longueur_caracteristique_km = np.nan

print(f"\nAmplitude du panache (zone salinite <= {seuil_panache}) : {amplitude_panache:.3f} psu")
print(f"Gradient moyen dans cette zone : {gradient_moyen_panache:.4e} psu/km")
print(f"Longueur caracteristique : {longueur_caracteristique_km:.1f} km")


# Figure : representation du gradient de salinite
fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle(f"Gradient de salinité de surface - {date}")

ax.set_extent(SWIO)
ax.coastlines(resolution='50m')
ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='black', zorder=3)
ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, zorder=3)

# Palette INFERNO classique,  avec une echelle non-lineaire (PowerNorm) pour etaler davantage les nuances sur les faibles valeurs de gradient
cmap = plt.colormaps['inferno']
norm = mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=0.1)

pcm = ax.pcolormesh(lon, lat, gradient_magnitude_km, cmap=cmap, norm=norm,
                      transform=ccrs.PlateCarree(), zorder=1)
cb = plt.colorbar(pcm, ax=ax, label='Gradient SSS [psu/km]', orientation='vertical')

gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.4)
gl.top_labels = False
gl.right_labels = False

fig.tight_layout()
fig.savefig('gradient_salinite.png', dpi=300)
print("\nFigure sauvegardee : gradient_salinite.png")
plt.show()

