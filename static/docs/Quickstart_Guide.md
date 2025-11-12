# Quickstart Guide - Cell Analysis Pipeline

Ausfuehrlichere Schritt-fuer-Schritt-Anleitung fuer die Remote-Nutzung. (Umlaute bewusst als ASCII geschrieben, damit das Dokument ueberall problemlos lesbar bleibt.)

## 1. Vorbereitung
1. **Repository aktualisieren:** Per `git pull` sicherstellen, dass UI und Konfig-Dateien aktuell sind.
2. **Docker-Compose pruefen:** `docker compose ps` muss laufen (bei GPU-Laeufen zusaetzlich `nvidia-smi` kontrollieren).
3. **Remote-Zugriff & Daten:** Google-Drive-Ordner synchronisieren, Remote Desktop (z. B. Google Remote Desktop) verbinden und den Datenpfad pruefen.

## 2. CZI -> TIFF Konvertierung
1. Tab **"CZI Conversion"** oeffnen.
2. *Input Directory* auf den CZI-Ordner setzen (z. B. `D:\Cell\Drive\CZI`).
3. *Output Directory* auf den gewuenschten TIFF-Ablageort setzen.
4. Kanaele, Zielgroesse, RGB-Kombination etc. nach Bedarf anpassen.
5. Konfiguration speichern und den Lauf starten. TIFFs erscheinen automatisch im Zielordner.

## 3. Detection (Cellpose / CPSAM)
1. Tab **"Detection"** oeffnen.
2. Eingabe-/Ausgabepfade zu den TIFFs setzen.
3. Kanaele aktivieren, globale oder per-Kanal-Sensitivitaeten definieren.
4. Nur relevante Optionen werden eingeblendet:
   - **Pre-Processing:** Tophat-Radius und Stretch-Percentiles erscheinen nur, wenn die Funktion aktiv ist.
   - **Split Cells:** Zusätzliche Parameter sind sichtbar, wenn "Split Touching Cells" aktiviert ist.
   - **Advanced Filtering:** SNR/Floor/Foreground-Felder werden nur eingeblendet, wenn "Enable Advanced" aktiv ist.
5. Overlays koennen farblich per Color-Picker angepasst werden (Konturfarben + Linienstaerken).
6. Config speichern und Detection starten – oder zuerst den Testmodus verwenden.

## 4. Co-Expression Analyse
1. Tab **"Co-Expression"** oeffnen.
2. *Use Detection Masks* aktiv lassen, wenn Masken existieren; nur bei Bedarf auf Overlay-Threshold zurueckfallen.
3. **Modus waehlen:** Overlap, Centroid, Intersection, Union oder **Blend (Heatmap)**.
   - **Centroid Max Distance** wird nur bei Modi angezeigt, die centroid-basierte Abstaende nutzen (Centroid, Union, Intersection ...).
   - **Overlap Threshold (%):** Regelt, wie viel Prozent der Basiszellflaeche mit allen Partnern ueberlappen muessen, damit die Zelle als mehrfach positiv zaehlt. Gilt fuer alle Modi mit Maskenueberlappung (Overlap, Either/Union, Intersection, Blend, All).
   - **Blend-Heatmap-Farben:** Base/Partner/Overlap-Farbpicker erscheinen nur, wenn der Blend-Modus aktiv ist.
4. Optional: Sweep-Modus aktivieren, gewuenschte Methoden anhaken und einen Sample-Key (per "Browse") auswaehlen.
5. Analyse starten; Ergebnisse enthalten CSVs, Overlays sowie bei Blend zusaetzliche PNG/TIFF-Heatmaps.

## 5. Ergebnisse pruefen & exportieren
1. In jedem Haupt-Tab zum Sub-Tab **"Results"** wechseln.
2. Mit *Refresh* aktualisieren; einzelne Dateien oder ZIP-Pakete (CSVs/Overlays/Heatmaps) herunterladen.
3. Die Master-CSV (`coexpr_sweep_summary.csv`) fasst alle Methoden-Laeufe zusammen – inklusive Condition, Region, Kanal-Kombination und Heatmap-Statistiken.

## 6. Parameter-Tipps
1. **Centroid Distance:** Niedrige Werte (z. B. 5–8 px) setzen voraus, dass Zellen nahezu ueberlappen. Groessere Werte erlauben lockerere "Nachbarschaften".
2. **Overlap Threshold (%):**
   - 0–2 % = minimaler Kontakt reicht (sehr sensitiv).
   - 5–10 % = nur deutliche Ueberlappung zaehlt (robuster gegen Rauschen).
   - Wird automatisch fuer alle Modi genutzt, die Maskenueberlappung auswerten (Overlap, Union/Either, Intersection, Blend, All/Both).
3. **Heatmap-Farben:** Base/Partner/Overlap lassen sich frei per Picker anpassen – ideal fuer Publikationen oder interne Farbcodes.
4. **Detection-Overlays:** Konturfarben und Linienstärken sind ebenfalls ueber Color-Picker anwählbar, so dass pro Kanal eindeutige Farben genutzt werden koennen.

## 7. Troubleshooting / Hinweise
1. **Logs beobachten:** In jedem Tab gibt es ein Konsolenfeld. Bei Fehlermeldungen Eintraege kopieren und im Team teilen.
2. **Konfigurationen sichern:** Nach relevanten Aenderungen Config speichern (Snapshots erleichtern Rueckverfolgbarkeit).
3. **Remote-Performance:** Bei GPU-Jobs regelmaessig `nvidia-smi` pruefen; bei Problemen `docker compose logs -f` ansehen.
4. **Heatmap-Kontrolle:** Blend-PNGs zeigen Grau/Cyan/Magenta, das zugehoerige TIFF enthaelt Label 1–3 (Base-only, Partner-only, Overlap) und kann in FIJI/ImageJ weiterverarbeitet werden.

> Dieses Dokument kann beliebig erweitert werden (z. B. mit Screenshots oder institutsinternen SOPs). Bei Bedarf einfach anpassen.
