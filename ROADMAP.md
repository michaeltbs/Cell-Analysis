# Cell Analysis — Roadmap

Liste von Wünschen aus dem Labor, priorisiert nach Aufwand.

## ✅ Erledigt / In Arbeit

- [x] HF GPU Space Verfügbarkeit
- [x] FastAPI Backend + asynchrone Jobs
- [x] Datei-Upload (CZI/TIFF)
- [x] Batch-Upload mehrerer Bilder auf einmal
- [x] Download-Links für Job-Ergebnisse
- [x] Prozentsatz der Expression (Pixel + per Zelle)
- [x] Expression-Prozentsatz als CSV exportieren (per Bedingung + kombiniert)
- [x] Distance-Map für Rezeptoren
- [x] Distance-Map als Statistik-CSV + Heatmap-PNG exportieren
- [x] Universelle Benennung für Kanäle/Bedingungen/Regionen
- [x] Naming-Config in UI / API-Requests erlauben
- [x] Kalibrierungsmodus: Intensitäts-Profil + Parameter-Optimierung (F1/MAE)
- [x] Cellpose Fine-Tuning Wrapper (eigenes Modell trainieren)

## 🟢 Leicht (1–3 Tage)

- [ ] Bessere Fehlermeldungen + Logging im UI

## 🟡 Mittel (3–7 Tage)

- [ ] Converter erweitern: ND2 (Nikon), LIF (Leica), OME-TIFF
- [ ] Umgang mit TIFF-Z-Stacks (mehrere Ebenen laden)
- [ ] Asynchrone Jobs mit Redis/Queue für HF Spaces (statt in-memory)
- [ ] HF Space authentifizierter Upload/Download

## 🔴 Schwer (1–4 Wochen)

- [ ] 3D-Segmentierung und 3D-Co-Expression
- [ ] Axon von Soma trennen (Morphologie oder spezialisiertes Modell)

## Geplante nächste Schritte

1. Rest der Mittel-Liste abarbeiten (ND2/LIF/OME-TIFF-Converter, Z-Stacks, Redis-Queue, HF-Auth).
2. Upload + Jobs auf HF Space deployen und mit GPU (T4 on-demand) testen.
3. Distance-Map-Werte pro Zelle (nicht nur aggregierte Stats pro Bild) — Rückfrage ans Labor.

## Notizen

- Cellpose 4.2+ ist aktuell auf HF GPU Space eingestellt.
- Alle Modelle CC-BY-NC (nicht-kommerziell).
- 3D und Axon/Soma sind separate Projekte — nicht in einem Sprint machbar.
- Export von Expression-CSV, Distance-Map-Statistik und Heatmaps läuft automatisch in jedem Upload-Job (Seit 2026-08-30).
