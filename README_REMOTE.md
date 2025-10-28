# Remote-Setup — Cell-Analysis

Kurzanleitung, um das Projekt remote (Host-Daten-Verzeichnis gemountet) laufen zu lassen.

Ziele:
- Host-Datenverzeichnis einfach in den Container mounten (Pfad `/data`).
- Vermeiden von Datei-/Rechteproblemen beim Schreiben in gemountete Ordner.

Vorbereitung
1. Stelle sicher, dass Docker und docker-compose (V2) auf dem Host installiert sind.
2. Optional: Wähle ein Host-Verzeichnis für die Daten, z.B. `/srv/cell_data`.

Empfohlene Startschritte

1) Build (einmalig oder nach Änderungen an der Dockerfile):

```bash
# falls UID/GID Angleichung gewünscht (z.B. Host-User ist uid 1001 gid 1001)
HOST_UID=$(id -u) HOST_GID=$(id -g) docker-compose build --build-arg USER_ID=${HOST_UID} --build-arg GROUP_ID=${HOST_GID}

# ohne explizite UID/GID:
docker-compose build
```

2) Start mit Host-Datenverzeichnis gemountet

```bash
# Beispiel: Host-Pfad /srv/cell_data wird in den Container nach /data gemountet
HOST_DATA=/srv/cell_data HOST_UID=$(id -u) HOST_GID=$(id -g) docker-compose up -d
```

Hinweise:
- Das Projekt mountet `HOST_DATA` nach `/data` im Container. App-Schnittstellen (z.B. der Date-Browser) listen `/data` als möglichen Anker.
- Wenn `HOST_DATA` nicht gesetzt ist, wird standardmäßig das Projektverzeichnis (.) als `/data` gemountet.
- `HOST_UID` / `HOST_GID` helfen, Schreibrechte auf dem gemounteten Verzeichnis zu erhalten, indem der Containerbenutzer dieselbe UID/GID verwendet.

GPU (NVIDIA) Hinweis
---------------------------------
Wenn dein Remote-Rechner eine NVIDIA-GPU hat (z.B. deine 1080Ti), kannst du die Analyse mit GPU-Beschleunigung laufen lassen. Kurze Schritte:

1) Host vorbereiten
  - NVIDIA-Treiber auf dem Host installieren (treiber passend zur GPU, z.B. 470/510/... je nach Kernel).
  - `nvidia-smi` sollte auf dem Host funktionieren und deine GPU zeigen.
  - Installiere `nvidia-container-toolkit` (früher nvidia-docker2). Siehe: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2) Compose GPU-Start
  - Ich habe `docker-compose.gpu.yml` angelegt. Damit baust und startest du die GPU-Variante:

```bash
# Beispiel: Build + Start GPU-Variante
# HOST_DATA auf den Pfad zu Daten setzen
HOST_DATA=/srv/cell_data HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

  - Alternativ kannst du das Hilfsskript mit dem Flag `--gpu` verwenden:

```bash
# macht build+start mit GPU-Compose
./start_remote.sh --gpu /srv/cell_data
```

3) Versionen/Kompatibilität
  - Die GPU-Docker-Stage verwendet CUDA 11.8 und installiert PyTorch mit CUDA 11.8 wheels (`torch==2.3.1+cu118`). Falls dein Host-Treiber zu alt ist, kann ein anderer CUDA-Tag nötig sein.
  - Wenn beim Start Fehler auftauchen (z.B. `Failed to initialize NVML` oder `CUDA library not found`), prüfe `nvidia-smi` auf dem Host und ggf. installierte `nvidia-container-toolkit` Version.

4) Debugging
  - Prüfe auf dem Host:
    - `nvidia-smi`
    - `docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi`
  - Logs anschauen: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f`

Zugriff
- Web-UI: http://<host>:5000 (bei lokalen/remote-Forwarding entsprechend Port freigeben)

Tipps zur Fehlersuche
- Dateirechte: Wenn der Container keine Dateien schreiben kann, prüfe `ls -ln /srv/cell_data` auf dem Host und vergleiche UID/GID mit der im Container verwendeten UID.
- Logs: `docker-compose logs -f` zeigt die Container-Ausgabe.

Weiteres / nächste Schritte
- Falls benötigt, kann ich:
  - Eine systemd-Unit oder ein docker-compose.override.yml mit persistenten Bind-Mounts vorbereiten.
  - Ein kurzes Start-Skript `start_remote.sh` erstellen, das ENV-Variablen abfragt und `docker-compose` startet.

Viel Erfolg — sag mir, wenn ich noch ein Startskript oder zusätzliche Automatisierung hinzufügen soll.
