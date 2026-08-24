import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chemins des csv contenant les stat sur le gradient de salinite
csv_1psu = '/lus/home/CT1/c1601279/aperez/SCRIPTS_ANAELLE/codes_fonctionnels/gradient/1PSU_gradient_salinite_annuel.csv'
csv_15psu = '/lus/home/CT1/c1601279/aperez/SCRIPTS_ANAELLE/codes_fonctionnels/gradient/15PSU_gradient_salinite_annuel.csv'

df_1psu = pd.read_csv(csv_1psu) #liste csv en tableau dataframe
df_15psu = pd.read_csv(csv_15psu)

# Figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 5.5))

col = 'longueur_caracteristique_km'

ax.fill_between(df_1psu['date'], df_1psu[col], alpha=0.15, color='#1f77b4')
ax.fill_between(df_15psu['date'], df_15psu[col], alpha=0.15, color='#ff7f0e')

ax.plot(df_1psu['date'], df_1psu[col], linewidth=2, marker='o', markersize=3,
        color='#1f77b4', label='Zambèze - 1 PSU')
ax.plot(df_15psu['date'], df_15psu[col], linewidth=2, marker='o', markersize=3,
        color='#ff7f0e', label='Zambèze - 15 PSU')

# Moyenne et mediane annuelles : lignes horizontales de reference, pour chaque cas
moy_1psu = np.nanmean(df_1psu[col])
moy_15psu = np.nanmean(df_15psu[col])
med_1psu = np.nanmedian(df_1psu[col])
med_15psu = np.nanmedian(df_15psu[col])

# Lignes pointillees (:) pour la MOYENNE, tiret-point (-.) pour la MEDIANE
# label= permet a ces lignes d'apparaitre automatiquement dans la legende
ax.axhline(moy_1psu, color='#1f77b4', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'Moyenne 1 PSU : {moy_1psu:.1f} km')
ax.axhline(moy_15psu, color='#ff7f0e', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'Moyenne 15 PSU : {moy_15psu:.1f} km')
ax.axhline(med_1psu, color='#1f77b4', linestyle='-.', linewidth=1.2, alpha=0.7,
           label=f'Médiane 1 PSU : {med_1psu:.1f} km')
ax.axhline(med_15psu, color='#ff7f0e', linestyle='-.', linewidth=1.2, alpha=0.7,
           label=f'Médiane 15 PSU : {med_15psu:.1f} km')

# Annotations : max et min de chaque courbe
def annoter_extreme(df, col, couleur, offset_max, offset_min):
    idx_max = df[col].idxmax()
    idx_min = df[col].idxmin()

    ax.annotate(f'Max: {df[col][idx_max]:,.1f} km\n{df["date"][idx_max]}',
                xy=(idx_max, df[col][idx_max]),
                xytext=offset_max, textcoords='offset points',
                ha='center', fontsize=8, color=couleur, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=couleur, linewidth=1.2),
                arrowprops=dict(arrowstyle='-', color=couleur, lw=0.8))

    ax.annotate(f'Min: {df[col][idx_min]:,.1f} km\n{df["date"][idx_min]}',
                xy=(idx_min, df[col][idx_min]),
                xytext=offset_min, textcoords='offset points',
                ha='center', va='top', fontsize=8, color=couleur, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=couleur, linewidth=1.2),
                arrowprops=dict(arrowstyle='-', color=couleur, lw=0.8))

annoter_extreme(df_1psu, col, '#1f77b4', offset_max=(-50, 0), offset_min=(20, -20))
annoter_extreme(df_15psu, col, '#ff7f0e', offset_max=(50, 0), offset_min=(-20, -7))

# Habillage
ax.set_ylabel("Longueur caractéristique [km]", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title("Évolution temporelle de la longueur caractéristique du gradient de salinité de surface - Zambèze",
             fontsize=12, fontweight='bold', pad=15)

ax.legend(fontsize=8, frameon=True, framealpha=0.9, loc='upper right', ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.grid(False, axis='x')

n_ticks = min(15, len(df_1psu))
step = max(1, len(df_1psu) // n_ticks)
ax.set_xticks(range(0, len(df_1psu), step))
ax.set_xticklabels([df_1psu['date'][k] for k in range(0, len(df_1psu), step)],
                     rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelsize=9)

fig.tight_layout()
fig.savefig('longueur_caracteristique_comparaison.png', dpi=300)
print("Figure sauvegardee : longueur_caracteristique_comparaison.png")
plt.show()
