import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path


def _infer_animal_from_filename(filename: str) -> str:
    """Extract animal ID from filename like M000_1.tiff, A2_1.tiff, WT01.tiff."""
    stem = Path(str(filename)).stem
    m = re.match(r"([A-Za-z]{1,4}\d+)", stem)
    return m.group(1) if m else stem


def _ensure_animal_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'animal' column if missing by inferring from filename."""
    if "animal" in df.columns and df["animal"].notna().any():
        return df
    if "filename" in df.columns:
        df = df.copy()
        df["animal"] = df["filename"].apply(_infer_animal_from_filename)
    else:
        df = df.copy()
        df["animal"] = "A1"
    return df

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


def run_mixed_effects_analysis(file_path: str, output_dir: str = None):
    """
    Linear mixed-effects model: cell_count ~ condition + (1 | animal/region).

    Treats animal as a random effect, which is statistically more correct than
    plain ANOVA when multiple sections come from the same animal (repeated measures).
    """
    print(f"--- Starte Mixed-Effects Analyse für: {file_path} ---")

    if output_dir is None:
        output_dir = os.path.dirname(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return

    if "region" not in df.columns or "condition" not in df.columns or "cell_count" not in df.columns:
        print("Spalten 'region', 'condition' oder 'cell_count' nicht gefunden.")
        return

    df = df.dropna(subset=["region", "condition", "cell_count"])
    df["region"] = df["region"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    df = _ensure_animal_col(df)

    results_text = ["--- MIXED-EFFECTS ERGEBNISSE (animal als Random Effect) ---"]
    header = f"{'Region':<10} | {'F-Wert':<8} | {'p-Wert':<8} | {'Beta_cond':<10} | {'N_Tiere':<8}"
    results_text.append(header)
    results_text.append("-" * 60)
    print(header)
    print("-" * 60)

    regions = df["region"].unique()
    for region in regions:
        df_sub = df[df["region"] == region]
        if len(df_sub["condition"].unique()) < 2:
            msg = f"{region:<10} | Zu wenig Gruppen für Mixed-Effects."
            print(msg)
            results_text.append(msg)
            continue
        n_animals = df_sub["animal"].nunique()
        if n_animals < 2:
            msg = f"{region:<10} | Zu wenige Tiere ({n_animals}) für Random-Effekt. Fallback ANOVA."
            print(msg)
            results_text.append(msg)
            continue
        try:
            model = mixedlm("cell_count ~ C(condition)", data=df_sub, groups=df_sub["animal"])
            fit = model.fit(reml=True)
            # Extract via Wald test approximation: compare to t-value
            params = fit.params
            p_cols = [c for c in params.index if c.startswith("C(condition)")]
            p_vals = [fit.pvalues[c] for c in p_cols] if len(p_cols) else []
            beta_cond = float(params[p_cols[0]]) if p_cols else float("nan")
            p_min = min(p_vals) if p_vals else float("nan")

            # Wald F-Statistic from t-values (approx for single coefficient)
            t_cols = [c for c in params.index if c.startswith("C(condition)")]
            t_val = float(fit.tvalues[t_cols[0]]) if t_cols else float("nan")
            f_wald = t_val ** 2

            sig = "*" if p_min < 0.05 else ""
            line = f"{region:<10} | {f_wald:.3f}    | {p_min:.3f} {sig:<2} | {beta_cond:>8.3f} | {n_animals:<8}"
            print(line)
            results_text.append(line)

        except Exception as e:
            msg = f"{region:<10} | Fehler bei Mixed-Effects: {e}"
            print(msg)
            results_text.append(msg)

    results_file = os.path.join(output_dir, "mixed_effects_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))
    print(f"Mixed-Effects Ergebnisse gespeichert in: {results_file}")
