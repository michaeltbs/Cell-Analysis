import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

def run_anova_analysis(file_path: str, output_dir: str = None):
    """
    Führt eine Univariate Varianzanalyse (ANOVA) durch und erstellt Boxplots.
    
    Args:
        file_path (str): Pfad zur CSV-Datei (z.B. All_Counts_Master.csv)
        output_dir (str, optional): Verzeichnis zum Speichern der Ergebnisse. 
                                    Wenn None, wird das Verzeichnis der Eingabedatei verwendet.
    """
    print(f"--- Starte ANOVA Analyse für: {file_path} ---")
    
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    
    # 1) DATEN LADEN
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return

    # Leere Regionen entfernen
    if 'region' not in df.columns:
        print("Spalte 'region' nicht gefunden.")
        return
        
    df = df.dropna(subset=['region'])

    # Optional: Leerzeichen in Strings bereinigen
    if 'region' in df.columns and df['region'].dtype == object:
        df['region'] = df['region'].str.strip()
    if 'condition' in df.columns and df['condition'].dtype == object:
        df['condition'] = df['condition'].str.strip()

    # 2) SCHLEIFE: ANOVA FÜR JEDE REGION
    results_text = []
    results_text.append("--- ANOVA ERGEBNISSE (Vergleich mit PDF) ---")
    header = f"{'Region':<10} | {'F-Wert':<8} | {'p-Wert':<8} | {'Eta^2 (part.)':<12}"
    results_text.append(header)
    results_text.append("-" * 50)
    
    print(header)
    print("-" * 50)

    regionen = df['region'].unique()

    for region in regionen:
        # Daten filtern
        df_sub = df[df['region'] == region]
        
        # Sicherstellen, dass wir genug Daten für Statistik haben
        if len(df_sub['condition'].unique()) < 2:
            msg = f"{region:<10} | Zu wenig Gruppen für ANOVA."
            print(msg)
            results_text.append(msg)
            continue

        try:
            # ANOVA Modell definieren: cell_count hängt ab von condition
            # Typ 3 Quadratsummen (typ='III') wie im SPSS Skript
            model = ols('cell_count ~ C(condition)', data=df_sub).fit()
            anova_table = sm.stats.anova_lm(model, typ=3)
            
            # Werte extrahieren
            row = anova_table.loc['C(condition)']
            f_val = row['F']
            p_val = row['PR(>F)']
            ss_cond = row['sum_sq']
            
            # Partielles Eta-Quadrat berechnen
            ss_error = anova_table.loc['Residual', 'sum_sq']
            eta_sq = ss_cond / (ss_cond + ss_error)
            
            # Signifikanz-Sternchen
            sig = "*" if p_val < 0.05 else ""
            
            line = f"{region:<10} | {f_val:.3f}    | {p_val:.3f} {sig:<2} | {eta_sq:.3f}"
            print(line)
            results_text.append(line)
            
        except Exception as e:
            print(f"Fehler bei ANOVA für Region {region}: {e}")

    # Ergebnisse in Textdatei speichern
    results_file = os.path.join(output_dir, "anova_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))
    print(f"ANOVA Ergebnisse gespeichert in: {results_file}")

    # 3) GRAFIK (Boxplots)
    try:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="region", y="cell_count", hue="condition", palette="pastel")

        # Titel und Labels
        plt.title("Boxplot: Fos-positive Zellen pro Region (Vergleich Pos vs Neg)")
        plt.xlabel("Region")
        plt.ylabel("Anzahl Zellen (cell_count)")
        plt.legend(title="Bedingung")

        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, "anova_boxplot.png")
        plt.savefig(plot_file)
        print(f"Boxplot gespeichert in: {plot_file}")
        # plt.show() # Im Workflow eher nicht anzeigen, sondern speichern
        plt.close()
    except Exception as e:
        print(f"Fehler beim Erstellen des Plots: {e}")
