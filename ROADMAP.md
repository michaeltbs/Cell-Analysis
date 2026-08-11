# Cell Analysis — Roadmap

Liste von Wünschen aus dem Labor, priorisiert nach Aufwand.

## ✅ Erledigt / In Arbeit

- [x] HF GPU Space Verfügbarkeit
- [x] FastAPI Backend + asynchrone Jobs
- [x] Datei-Upload (CZI/TIFF)
- [x] Prozentsatz der Expression (Pixel + per Zelle)
- [x] Universelle Benennung für Kanäle/Bedingungen/Regionen
- [x] Distance-Map für Rezeptoren
- [x] Kalibrierungsmodus: Intensitäts-Profil + Parameter-Optimierung (F1/MAE)
- [x] Cellpose Fine-Tuning Wrapper (eigenes Modell trainieren)

## 🟢 Leicht (1–3 Tage)

- [ ] Expression-Prozentsatz in CSV/Report exportieren
- [ ] Distance-Map als Overlay/Heatmap exportieren
- [ ] Naming-Config in UI / API-Requests erlauben
- [ ] Batch-Upload mehrerer Bilder auf einmal
- [ ] Download-Links für Job-Ergebnisse

## 🟡 Mittel (3–7 Tage)

- [ ] Converter erweitern: ND2 (Nikon), LIF (Leica), OME-TIFF
- [ ] Umgang mit TIFF-Z-Stacks (mehrere Ebenen laden)
- [ ] Asynchrone Jobs mit Redis/Queue für HF Spaces (statt in-memory)
- [ ] Bessere Fehlermeldungen + Logging im UI
- [ ] HF Space authentifizierter Upload/Download

## 🔴 Schwer (1–4 Wochen)

- [ ] 3D-Segmentierung und 3D-Co-Expression
- [ ] Axon von Soma trennen (Morphologie oder spezialisiertes Modell)
- [ ] Eigene Cellpose-Modelle trainieren/fine-tunen

## Geplante nächste Schritte

1. Upload + Jobs auf HF Space deployen und testen.
2. Expression-Metriken in den API-Output integrieren.
3. Naming-Config in FastAPI-Upload-Endpoint einbauen.
4. Distance-Map als PNG/CSV ausgeben.

## Notizen

- Cellpose 4.2+ ist aktuell auf HF GPU Space eingestellt.
- Alle Modelle CC-BY-NC (nicht-kommerziell).
- 3D und Axon/Soma sind separate Projekte — nicht in einem Sprint machbar.
