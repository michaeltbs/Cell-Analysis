# --- 1. Installation und Laden der benötigten Bibliotheken ---
# Führen Sie diese Zeilen einmal in Ihrem Terminal oder Ihrer Konsole aus,
# falls Sie die Bibliotheken noch nicht installiert haben:
# pip install pandas seaborn matplotlib scikit-learn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 2. Konfiguration & Daten laden ---
input_file = Path("/mnt/c/Users/micha/Desktop/project_cellanalysis/Output/all_final_summaries.csv")
output_base_dir = Path("/mnt/c/Users/micha/Desktop/project_cellanalysis/Output/Plots")

# Ausgabeordner erstellen
output_base_dir.mkdir(exist_ok=True, parents=True)
dir_part1 = output_base_dir / "1_Individual_Analyses_vs_Age"
dir_part2 = output_base_dir / "2_Individual_Analyses_vs_Region"
dir_part3 = output_base_dir / "3_Advanced_Analyses"
dir_part4 = output_base_dir / "4_Proportional_Analyses"
dir_part5 = output_base_dir / "5_Feature_Importance"
dir_part1.mkdir(exist_ok=True)
dir_part2.mkdir(exist_ok=True)
dir_part3.mkdir(exist_ok=True)
dir_part4.mkdir(exist_ok=True)
dir_part5.mkdir(exist_ok=True)

# Daten laden
print(f"Lade Daten von: {input_file}")
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"FEHLER: Die Datei '{input_file}' wurde nicht gefunden. Bitte überprüfen Sie den Pfad.")
    exit()

# Automatische Erkennung der Kanal-Spalte
if 'channel_key' in df.columns:
    channel_col_name = 'channel_key'
elif 'channel' in df.columns:
    channel_col_name = 'channel'
else:
    print("\nFEHLER: Konnte die Kanal-Spalte ('channel_key' oder 'channel') nicht finden.")
    exit()

print(f"\nVerwende '{channel_col_name}' als Kanal-Spalte.")
df = df.dropna(subset=['region', 'age', channel_col_name])
df = df[df['region'] != ""].copy()
df['age'] = pd.Categorical(df['age'], categories=['Adult', 'Old'], ordered=True)

metrics_map = { "cell_count": "Cell Count", "mean_area_per_cell": "Mean Area per Cell", "mean_intensity_per_cell": "Mean Intensity per Cell", "mean_integrated_density_per_cell": "Mean Integrated Density per Cell", "total_image_int_den": "Total Image Integrated Density" }
metrics = list(metrics_map.keys())
channels = df[channel_col_name].unique()

# --- 3. Hilfsfunktionen ---

def p_to_stars(p):
    """Konvertiert einen p-Wert in Signifikanz-Sterne."""
    if p < 0.0001: return "****"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def iqr_outlier_removal(data, metric, group_vars):
    """Entfernt Ausreißer basierend auf dem 1.5*IQR-Kriterium pro Gruppe."""
    if metric not in data.columns: return data
    valid_group_vars = [var for var in group_vars if var in data.columns]
    if not valid_group_vars: return data
    # Warnung behoben durch Hinzufügen von observed=False
    df_grouped = data.groupby(valid_group_vars, observed=False)
    q1 = df_grouped[metric].transform('quantile', 0.25)
    q3 = df_grouped[metric].transform('quantile', 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    initial_rows = len(data)
    filtered_data = data[(data[metric] >= lower_bound) & (data[metric] <= upper_bound)]
    removed_count = initial_rows - len(filtered_data)
    print(f"IQC für '{metric}': {removed_count} Ausreißer entfernt.")
    return filtered_data

def add_stat_annotation(ax, data, x_var, y_var, pairs):
    """Fügt p-Wert-Sterne hinzu und passt bei Bedarf die Y-Achse an."""
    print(f"\n--- Statistische Analyse für: {y_var} ---")
    
    y_max_overall = data[y_var].max()
    y_min_overall = data[y_var].min()
    y_range = y_max_overall - y_min_overall
    # Setzt einen Puffer von 30% der Daten-Reichweite obendrauf, um Abschneiden zu verhindern
    puffer = y_range * 0.3 if y_range > 0 else 1
    ax.set_ylim(top=y_max_overall + puffer)
    
    for pair in pairs:
        group1, group2 = pair
        data1 = data[data[x_var] == group1][y_var].dropna()
        data2 = data[data[x_var] == group2][y_var].dropna()
        if len(data1) < 2 or len(data2) < 2:
            print(f"  Statistik für {group1} vs {group2} übersprungen: Nicht genügend Daten.")
            continue
        try:
            U, p = mannwhitneyu(data1, data2, alternative='two-sided')
            n1, n2 = len(data1), len(data2); effect_size_r = 1 - (2 * U) / (n1 * n2) if (n1*n2)>0 else 0
            print(f"  Vergleich {group1} vs {group2}: p={p:.4f}, Effektstärke (r)={effect_size_r:.4f}")
            stars = p_to_stars(p)
            x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            if group1 not in x_labels or group2 not in x_labels: continue
            x1, x2 = x_labels.index(group1), x_labels.index(group2)
            y_max = max(data1.max(), data2.max())
            y_offset = y_range * 0.05 if y_range > 0 else 0.05
            line_y = y_max + y_offset
            
            ax.plot([x1, x1, x2, x2], [line_y, line_y + y_offset, line_y + y_offset, line_y], lw=1.5, c='k')
            ax.text((x1 + x2) * 0.5, line_y + y_offset*1.2, stars, ha='center', va='bottom', color='k', fontsize=14)
        except Exception as e:
            print(f"  Konnte Statistik für Paar {pair} nicht berechnen: {e}")

# --- 4. Ausführung der Analyseteile ---

# Teil 1 & 2: Individualanalysen
print("\n--- TEIL 1 & 2: Starte Einzelanalysen für Alter und Regionen ---")
for channel in channels:
    print(f"\n===== Verarbeite Kanal: {channel.upper()} =====")
    df_channel = df[df[channel_col_name] == channel].copy()
    
    dir_age_out = dir_part1 / channel
    dir_region_out = dir_part2 / channel
    dir_age_out.mkdir(exist_ok=True, parents=True)
    dir_region_out.mkdir(exist_ok=True, parents=True)
    
    for metric, y_lab in metrics_map.items():
        if metric not in df_channel.columns: continue
        
        df_clean = iqr_outlier_removal(df_channel, metric, ['region', 'age'])
        if df_clean.empty: continue

        # Altersvergleich (Adult vs. Old, pro Region)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_clean, x='region', y=metric, hue='age', palette='pastel', ax=ax)
        ax.set_title(f"{channel.upper()}: {y_lab}\n(Vergleich Adult vs. Old)")
        ax.set_ylabel(f"{y_lab} (IQC)")
        ax.set_xlabel("Region")
        plt.tight_layout()
        plt.savefig(dir_age_out / f"1_Altersvergleich_{metric}.tiff", dpi=300) # Geändert zu .tiff
        plt.close(fig)

        # Regionenvergleich (ARC vs. DMH, pro Altersgruppe)
        if len(df_clean['region'].unique()) > 1:
            # Warnung behoben durch Zuweisung von hue und Deaktivierung der Legende
            g = sns.catplot(data=df_clean, x='region', y=metric, col='age', kind='box', hue='region', palette='muted', legend=False)
            g.fig.suptitle(f"{channel.upper()}: {y_lab}\n(Vergleich ARC vs. DMH)", y=1.03)
            g.set_axis_labels("Region", f"{y_lab} (IQC)")
            for i, age_group in enumerate(g.col_names):
                ax = g.axes[0][i]
                add_stat_annotation(ax, df_clean[df_clean['age'] == age_group], x_var='region', y_var=metric, pairs=[('ARC', 'DMH')])
            plt.savefig(dir_region_out / f"2_Regionenvergleich_{metric}.tiff", dpi=300) # Geändert zu .tiff
            plt.close('all')

# Teil 3: Erweiterte Analysen
print("\n--- TEIL 3: Starte erweiterte Gesamtanalysen ---")
corr = df[metrics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f")
plt.title("Korrelations-Heatmap der Messgrößen")
plt.tight_layout()
plt.savefig(dir_part3 / "3_Korrelations_Heatmap.tiff", dpi=300) # Geändert zu .tiff
plt.close()

df_pca_clean = df.dropna(subset=metrics)
X_pca = df_pca_clean[metrics]
grouping_vars_pca = df_pca_clean[['age', 'region', channel_col_name]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, grouping_vars_pca.reset_index(drop=True)], axis=1)
for group in ['age', 'region', channel_col_name]:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=group, alpha=0.7, s=80, palette='deep')
    title_group = group.title() if group != channel_col_name else 'Channel'
    plt.title(f"PCA der Messgrößen, gruppiert nach {title_group}")
    plt.xlabel(f"Hauptkomponente 1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"Hauptkomponente 2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend(title=title_group, bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(dir_part3 / f"3_PCA_Plot_nach_{group}.tiff", dpi=300) # Geändert zu .tiff
    plt.close()

# Teil 4: Proportionale Analyse
print("\n--- TEIL 4: Starte proportionale Analyse ---")
df_pivot = df.pivot_table(index=['filename', 'age', 'region'], columns=channel_col_name, values='cell_count', fill_value=0).reset_index()
proportions_to_calc = { 'pct_gal_pomc_in_gal': ('gal_pomc', 'gal', 'Anteil von Gal+Pomc in Gal'), 'pct_gal_pomc_in_pomc': ('gal_pomc', 'pomc', 'Anteil von Gal+Pomc in Pomc'), 'pct_gal_glp1r_in_gal': ('gal_glp1r', 'gal', 'Anteil von Gal+Glp1r in Gal'), 'pct_gal_glp1r_in_glp1r': ('gal_glp1r', 'glp1r', 'Anteil von Gal+Glp1r in Glp1r'), 'pct_pomc_glp1r_in_pomc': ('pomc_glp1r', 'pomc', 'Anteil von Pomc+Glp1r in Pomc'), 'pct_pomc_glp1r_in_glp1r': ('pomc_glp1r', 'glp1r', 'Anteil von Pomc+Glp1r in Glp1r'), 'pct_triple_in_gal': ('gal_pomc_glp1r', 'gal', 'Anteil von Triple+ in Gal'), 'pct_triple_in_pomc': ('gal_pomc_glp1r', 'pomc', 'Anteil von Triple+ in Pomc'), 'pct_triple_in_glp1r': ('gal_pomc_glp1r', 'glp1r', 'Anteil von Triple+ in Glp1r'), 'pct_triple_in_gal_pomc': ('gal_pomc_glp1r', 'gal_pomc', 'Anteil von Triple+ in Gal+Pomc')}
for new_col, (numerator, denominator, _) in proportions_to_calc.items():
    if numerator in df_pivot.columns and denominator in df_pivot.columns:
        df_pivot[new_col] = np.divide(df_pivot[numerator], df_pivot[denominator], out=np.zeros_like(df_pivot[numerator], dtype=float), where=(df_pivot[denominator] != 0)) * 100
for col_name, (_, _, plot_title_part) in proportions_to_calc.items():
    if col_name in df_pivot.columns:
        df_prop_clean = df_pivot.dropna(subset=[col_name])
        if df_prop_clean.empty: continue
        if len(df_prop_clean['region'].unique()) > 1:
            g = sns.catplot(data=df_prop_clean, x='region', y=col_name, col='age', kind='box', hue='region', palette='muted', legend=False, height=5, aspect=0.8)
            g.fig.suptitle(f"{plot_title_part}\n(Vergleich ARC vs. DMH)", y=1.03)
            g.set_axis_labels("Region", "Anteil (%)")
            for i, age_group in enumerate(g.col_names):
                ax = g.axes[0][i]
                add_stat_annotation(ax, df_prop_clean[df_prop_clean['age'] == age_group], x_var='region', y_var=col_name, pairs=[('ARC', 'DMH')])
            plt.savefig(dir_part4 / f"4_Anteil_{col_name}_vs_Region.tiff", dpi=300) # Geändert zu .tiff
            plt.close('all')
        if len(df_prop_clean['age'].unique()) > 1:
            g = sns.catplot(data=df_prop_clean, x='age', y=col_name, col='region', kind='box', hue='age', palette='pastel', legend=False, height=5, aspect=0.8)
            g.fig.suptitle(f"{plot_title_part}\n(Vergleich Adult vs. Old)", y=1.03)
            g.set_axis_labels("Altersgruppe", "Anteil (%)")
            for i, region_group in enumerate(g.col_names):
                ax = g.axes[0][i]
                add_stat_annotation(ax, df_prop_clean[df_prop_clean['region'] == region_group], x_var='age', y_var=col_name, pairs=[('Adult', 'Old')])
            plt.savefig(dir_part4 / f"4_Anteil_{col_name}_vs_Alter.tiff", dpi=300) # Geändert zu .tiff
            plt.close('all')

# Teil 5: Feature Importance Analyse
print("\n--- TEIL 5: Starte Feature Importance Analyse ---")
for channel in channels:
    channel_dir_out = dir_part5 / channel
    channel_dir_out.mkdir(exist_ok=True, parents=True)
    df_channel_filtered = df[df[channel_col_name] == channel]
    for target_col in ['age', 'region']:
        df_model = df_channel_filtered.dropna(subset=metrics + [target_col]).copy()
        if df_model[target_col].nunique() < 2 or len(df_model) < 10: continue
        X = df_model[metrics]
        y = df_model[target_col]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        importances = pd.DataFrame({'Feature': [metrics_map.get(f, f) for f in X.columns], 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        # Warnung behoben durch Zuweisung von hue und Deaktivierung der Legende
        sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', palette='viridis', legend=False)
        plt.title(f"Wichtigkeit der Merkmale zur Vorhersage von '{target_col.title()}'\nKanal: {channel.upper()} (Genauigkeit: {accuracy:.1%})")
        plt.xlabel("Relative Wichtigkeit")
        plt.ylabel("Merkmal (Metrik)")
        plt.tight_layout()
        plt.savefig(channel_dir_out / f"5_FeatureImportance_{target_col}_{channel}.tiff", dpi=300) # Geändert zu .tiff
        plt.close()

print("\n\nAnalyse erfolgreich abgeschlossen!")
print(f"Alle Ergebnisse wurden im Ordner '{output_base_dir}' gespeichert.")
