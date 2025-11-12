# Remote-Setup - Cell-Analysis

Kurzanleitung, um das Projekt remote zu betreiben und Daten ueber Google Drive bereitzustellen.

## Ziele
- Host-Datenverzeichnis sauber in den Container mounten (`/data`).
- Schreibrechte auf gemounteten Ordnern erhalten.
- Analysen per Google Remote Desktop starten und ueber einen synchronisierten Drive-Ordner versorgen.

## Vorbereitung
1. Docker und docker compose (V2) auf dem Host installieren.
2. Optional ein dediziertes Datenverzeichnis anlegen, z. B. `D:\CellAnalysisData`.

## Container starten

### 1) Build (einmalig oder nach Dockerfile-Aenderungen)
```bash
# UID/GID-Anpassung (falls Host-User nicht 1000:1000 ist)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build --build-arg USER_ID=${HOST_UID} --build-arg GROUP_ID=${HOST_GID}

# Ohne UID/GID-Anpassung
docker compose build
```

### 2) Start mit gemountetem Host-Datenverzeichnis
```bash
# Beispiel: Host-Pfad /srv/cell_data wird als /data gemountet
HOST_DATA=/srv/cell_data HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d
```

### Hinweise
- `HOST_DATA` wird als `/data` im Container sichtbar. Die Web-App verwendet diesen Pfad im Dateibrowser.
- Wenn `HOST_DATA` fehlt, wird standardmaessig das aktuelle Projektverzeichnis gemountet.
- `HOST_UID` und `HOST_GID` sorgen dafuer, dass der Container-User dieselbe UID/GID wie der Host-User besitzt und damit Schreibrechte behaelt.

## GPU (NVIDIA)
Wenn der Remote-Rechner eine NVIDIA-GPU besitzt, kann die GPU-Variante genutzt werden.

1. NVIDIA-Treiber auf dem Host installieren (`nvidia-smi` sollte funktionieren).
2. `nvidia-container-toolkit` einrichten: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

## Quickstart Guide
1. **CZI → TIFF:** Im Tab *CZI Conversion* Eingabe- und Ausgabeordner setzen, optional Kanäle festlegen und den Lauf starten.  
2. **Detection:** Im Tab *Detection* Kanäle/Sensitivitäten wählen, Config speichern und `Start Detection` ausführen.  
3. **Co-Expression:** Im Tab *Co-Expression* den gewünschten Modus (z. B. Blend-Heatmap) wählen und Analysen bzw. Sweeps starten.  
4. **Review & Export:** Resultate (CSV, Overlays, Heatmaps) über die jeweiligen *Results*-Subtabs herunterladen.

## Impressum
Cell Analysis Pipeline (Remote)  
Forschungscampus Datenwerk  
Musterstraße 12  
10999 Berlin, Deutschland

Kontakt: remote-support@cell-analysis.example  
Telefon: +49 30 123456-0  
Verantwortlich gem. §5 TMG: Projektleitung Cell Analysis  
USt-IdNr.: DE999999999

> **Hinweis:** Bitte ergänzen/ersetzen Sie diese Angaben bei Bedarf mit den offiziellen Daten Ihres Teams oder Instituts.
3. GPU-Compose-Datei verwenden:
   ```bash
   HOST_DATA=/srv/cell_data HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   ```
4. Alternativ das Hilfsskript mit GPU-Flag nutzen:
   ```bash
   ./start_remote.sh --gpu /srv/cell_data
   ```

## Google Drive Synchronisierung

1. Google Drive for Desktop auf Leistungs-PC, Mac und weiteren Clients installieren.
2. Einen Projektordner (z. B. `CellAnalysisData`) anlegen und als "Spiegeln" konfigurieren, damit die Dateien lokal vorliegen.
3. Auf Windows erscheint der Ordner als Laufwerk (z. B. `G:\CellAnalysisData`). Diesen Pfad in `HOST_DATA` setzen:
   ```powershell
   set HOST_DATA=G:\CellAnalysisData
   docker compose up -d --build
   ```
4. Auf dem Mac liegt der Ordner unter `/Volumes/GoogleDrive`. Dateien werden dort automatisch synchronisiert und stehen dem Container bereit.
5. Input- und Output-Unterordner trennen (`input/`, `output/`), um Konflikte zu vermeiden.
6. Bei sensiblen Daten optional clientseitige Verschluesselung einsetzen (z. B. Cryptomator).

### Web-App Hinweise
- `DATA_UPLOAD_SUBDIR`: legt den Unterordner innerhalb von `HOST_DATA` fest, den die Upload-Oberflaeche nutzt (`Input` als Standard).
- `GOOGLE_DRIVE_SYNC_PATH`: Pfadangabe (z. B. `G:\CellAnalysisData\Input`), die im Dashboard angezeigt wird, damit alle User denselben Drive-Ordner sehen.
- `GOOGLE_DRIVE_EMBED_URL`: Optionaler Embed-Link (`https://drive.google.com/embeddedfolderview?id=<FOLDER_ID>#list`), um den Drive-Ordner direkt im Upload-Tab einzubetten.
- Nach dem Setzen neuer Variablen den Container neu starten, damit Flask die Werte uebernimmt.
- Im Detection-Tab steuert 'Current Magnification' die automatische Skalierung der zellgroessenabhaengigen Parameter (Durchmesser, Min/Max-Area usw.).
- Der Button `Upload folder...` (Chrome/Edge) ermoeglicht das Hochladen kompletter Ordner samt Unterstruktur in einem Schritt.

### Co-Expression Einstellungen
- **Modus (Overlap/Centroid/Union/Intersection)**: Im Co-Expression-Tab per Dropdown waehlen. 
  - `Overlap` (Standard) liefert die robustesten Ergebnisse, solange die Masken je Kanal präzise auf derselben Position liegen.
  - `Union` (Overlap OR Centroid) ist tolerant gegen leichte Masken-Offsets und eignet sich fuer Serien mit minimalen Kanalverschiebungen.
  - `Centroid` oder `Intersection` nur einsetzen, wenn explizit Zentroid-Abstaende bzw. sehr strenge Uebereinstimmungen gefordert sind.
- **Centroid Max Distance**: Einflussradius fuer die centroid-basierten Modi. 
  - Für typische 10x-Aufnahmen hat sich ein Wert von **8 px** bewaehrt.
  - Bei stark vergroesserten Bildern (&gt;10x) auf 10–12 px erhoehen, bei 5x-Aufnahmen ggf. auf 5–6 px reduzieren.
- **Test Mode**: `Simple` waehlt pro Lauf eine kleine Zufallsstichprobe (Anzahl/Seed konfigurierbar); `Sweep` fixiert einen Sample Key und prueft mehrere Methoden hintereinander.
- **Co-Expression Figuren**: `Figure Type = mask` zeigt reine Maskenpanels, `overlay` kombiniert Original-Overlay plus farbige Konturen/Füllungen.
- **Sweep Test**: Im Test Mode `Sweep` koennen mehrere Methoden parallel angehakt werden (z. B. `overlap`, `either`, `intersection`) und laufen nacheinander auf dem gewaehlten Sample Key. Ergebnisse landen unter `__sweep__/<modus>` im Output.

## Google Remote Desktop
- Google Remote Desktop fuer den Leistungs-PC einrichten und in der Sitzung angemeldet bleiben.
- Aufgaben via Remote Desktop starten und ggf. per Aufgabenplanung oder Skripte automatisieren.
- Remote Desktop beeinflusst den Containerbetrieb nicht; GPU-Beschleunigung bleibt aktiv.

## Zugriff
- Web-UI: `http://<host>:5000` (Portweiterleitung oder VPN beachten).

## Fehlersuche
- Schreibrechte: `docker compose exec cellanalysis ls -ln /data` und UID/GID pruefen.
- Container-Logs: `docker compose logs -f`.
- GPU-Probleme: `nvidia-smi` und `docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi`.

## Weitere Schritte
- Optional systemd-Unit oder `docker-compose.override.yml` erstellen, um Container automatisch zu starten.
- Bei Bedarf ein Skript schreiben, das Analysen nach dem Drive-Sync automatisch anstoesst.
