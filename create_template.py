"""
Erstellt die vollständige templates/index.html Datei neu.
Ausführen:
  python create_template.py
Danach:
  python app.py  -> http://localhost:5000
"""
import os

HTML_CONTENT = """<!-- filepath: /c:/Users/Admin/Desktop/project_cellanalysis/templates/index.html -->
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Cell Analysis - Co-Expression Analyse</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; background: #f5f7fb; margin: 0; }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .card { background: #fff; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); padding: 18px 22px; margin: 16px 0; }
    h1 { margin: 0 0 6px 0; font-size: 22px; }
    h2 { font-size: 18px; margin: 8px 0 12px; color: #333; }
    .tabs { display: flex; gap: 8px; margin: 12px 0 16px; }
    .tab { padding: 10px 14px; border: 1px solid #e1e5ef; background: #fff; border-radius: 8px; cursor: pointer; }
    .tab.active { background: #eef2ff; color: #3344aa; border-color: #cfd8ff; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    label { display: block; font-weight: 600; margin: 10px 0 6px; }
    input[type="text"], input[type="number"], select {
      width: 100%; padding: 10px; border: 1px solid #d8deea; border-radius: 8px; background: #fbfcff;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .btn { padding: 10px 14px; border-radius: 8px; border: 1px solid #d8deea; background: #fff; cursor: pointer; }
    .btn.primary { background: #4f6cff; color: #fff; border-color: #4f6cff; }
    .btn.primary:disabled { background: #a0abff; cursor: not-allowed; }
    .btn.block { width: 100%; }
    .muted { color: #666; font-size: 12px; }
    .log { background: #111827; color: #d1d5db; font-family: Consolas, monospace; border-radius: 8px; padding: 10px; height: 260px; overflow: auto; }
    .results .item { display: flex; align-items: center; justify-content: space-between; padding: 10px; border: 1px solid #e8ecf5; border-radius: 8px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>🔬 Cell Analysis - Co-Expression Analyse</h1>
      <div class="muted">Web-Interface zur Konfiguration, Ausführung und Ergebnis-Ansicht</div>
    </div>

    <div class="card">
      <div class="tabs">
        <button class="tab active" onclick="switchTab('config', event)">⚙️ Konfiguration</button>
        <button class="tab" onclick="switchTab('analysis', event)">▶️ Analyse</button>
        <button class="tab" onclick="switchTab('results', event)">📊 Ergebnisse</button>
      </div>

      <!-- Konfiguration -->
      <div id="config" class="tab-content active">
        <h2>Pfad- und Analyse-Parameter</h2>
        <label>Input-Verzeichnis (mit ch0/ch1/.../overlays)</label>
        <input id="input_dir" type="text" placeholder="z.B. results_lea2" />
        <label>Output-Verzeichnis (Speicherort für CSV/Overlays)</label>
        <input id="output_dir" type="text" placeholder="z.B. coexpression_lea2_new" />

        <div class="row">
          <div>
            <label>Datensatz-Name</label>
            <input id="dataset_name" type="text" placeholder="z.B. lea"/>
          </div>
          <div>
            <label>Aktiver HSV-Bereich</label>
            <select id="active_hsv_range">
              <option value="0">Lime Grün [50,100,100]–[70,255,255]</option>
              <option value="1">Erweitert Grün [35,50,50]–[85,255,255]</option>
              <option value="2">Helles Lime [45,80,150]–[75,255,255]</option>
              <option value="3">Breites Grün [30,30,30]–[90,255,255]</option>
            </select>
          </div>
        </div>

        <div class="row">
          <div>
            <label>Auto-Thresholding</label>
            <select id="use_auto_threshold">
              <option value="true">Ja (empfohlen)</option>
              <option value="false">Nein (fester Wert)</option>
            </select>
          </div>
          <div>
            <label>Distanz-Schwelle (Pixel)</label>
            <input id="distance_threshold" type="number" value="30" />
          </div>
        </div>

        <div class="row">
          <div>
            <label>K_First (Kalibrierungsbilder)</label>
            <input id="calibration_k_first" type="number" value="8" />
          </div>
          <div>
            <label>Safety Multiplier</label>
            <input id="safety_multiplier" type="number" step="0.05" value="1.25" />
          </div>
        </div>

        <div class="row">
          <div>
            <label>Max Factor Diameter</label>
            <input id="max_factor_diameter" type="number" step="0.1" value="2.0" />
          </div>
          <div>
            <label>Min Pairs</label>
            <input id="min_pairs" type="number" value="10" />
          </div>
        </div>

        <div class="row">
          <div>
            <label>Smooth Window</label>
            <input id="smooth_window" type="number" value="5" />
          </div>
          <div>
            <label>Analyse aktiv</label>
            <select id="enable_analysis">
              <option value="true">Ja</option>
              <option value="false">Nein (nur Übersicht)</option>
            </select>
          </div>
        </div>

        <div style="margin-top: 12px; display:flex; gap:10px;">
          <button class="btn" onclick="loadConfig()">🔄 Laden</button>
          <button class="btn primary" onclick="saveConfig()">💾 Speichern</button>
        </div>
      </div>

      <!-- Analyse -->
      <div id="analysis" class="tab-content">
        <h2>Analyse starten</h2>
        <div class="muted" style="margin-bottom:8px;">Logs</div>
        <div id="log" class="log">Warte auf Analyse-Start…</div>
        <div style="margin-top: 12px;">
          <button id="start-btn" class="btn primary" onclick="startAnalysis()">▶️ Analyse starten</button>
        </div>
      </div>

      <!-- Ergebnisse -->
      <div id="results" class="tab-content">
        <h2>Ergebnisse</h2>
        <div style="margin-bottom: 8px;">
          <button class="btn" onclick="loadResults()">🔄 Aktualisieren</button>
        </div>
        <div id="results-list" class="results">
          <div class="muted">Keine Ergebnisse geladen.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(id, ev) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (ev && ev.target) ev.target.classList.add('active');
      document.getElementById(id).classList.add('active');
      if (id === 'results') loadResults();
    }

    async function loadConfig() {
      try {
        const res = await fetch('/api/config');
        const cfg = await res.json();

        document.getElementById('input_dir').value = cfg.paths?.base_results_dir ?? '';
        document.getElementById('output_dir').value = cfg.paths?.output_dir ?? '';
        document.getElementById('dataset_name').value = cfg.dataset_name ?? '';
        document.getElementById('active_hsv_range').value = cfg.detection?.active_hsv_range_index ?? 0;

        document.getElementById('enable_analysis').value =
          (cfg.enable_analysis === false) ? 'false' : 'true';

        const ap = cfg.analysis_params ?? {};
        document.getElementById('distance_threshold').value = ap.distance_threshold ?? 30;
        document.getElementById('use_auto_threshold').value = (ap.use_auto_threshold === false) ? 'false' : 'true';
        document.getElementById('calibration_k_first').value = ap.calibration_k_first ?? 8;
        document.getElementById('safety_multiplier').value = ap.safety_multiplier ?? 1.25;
        document.getElementById('max_factor_diameter').value = ap.max_factor_diameter ?? 2.0;
        document.getElementById('min_pairs').value = ap.min_pairs ?? 10;
        document.getElementById('smooth_window').value = ap.smooth_window ?? 5;
      } catch (e) {
        console.error(e);
        alert('Konfiguration konnte nicht geladen werden.');
      }
    }

    async function saveConfig() {
      const config = {
        paths: {
          base_results_dir: document.getElementById('input_dir').value.trim(),
          output_dir: document.getElementById('output_dir').value.trim()
        },
        dataset_name: document.getElementById('dataset_name').value.trim(),
        enable_analysis: document.getElementById('enable_analysis').value === 'true',
        analysis_params: {
          distance_threshold: parseInt(document.getElementById('distance_threshold').value, 10) || 30,
          use_auto_threshold: document.getElementById('use_auto_threshold').value === 'true',
          calibration_k_first: parseInt(document.getElementById('calibration_k_first').value, 10) || 8,
          safety_multiplier: parseFloat(document.getElementById('safety_multiplier').value) || 1.25,
          max_factor_diameter: parseFloat(document.getElementById('max_factor_diameter').value) || 2.0,
          min_pairs: parseInt(document.getElementById('min_pairs').value, 10) || 10,
          smooth_window: parseInt(document.getElementById('smooth_window').value, 10) || 5,
          region_fixed_thresholds: {}
        },
        detection: {
          active_hsv_range_index: parseInt(document.getElementById('active_hsv_range').value, 10) || 0,
          hsv_ranges: [
            ["Lime Grün (Standard)", [50,100,100],[70,255,255]],
            ["Erweitert Grün", [35,50,50],[85,255,255]],
            ["Helles Lime", [45,80,150],[75,255,255]],
            ["Breites Grün", [30,30,30],[90,255,255]]
          ]
        },
        region_mapping: {
          larc:"lArc", rarc:"rArc", arc:"Arc",
          ldmh:"lDMH", rdmh:"rDMH", dmh:"DMH",
          lpvh:"lPVH", rpvh:"rPVH", pvh:"PVH",
          "5x":"5x", "5x_pvh":"5x_PVH",
          negativecontrol:"NegativeControl", positivecontrol:"PositiveControl",
          lmbh:"lMBH", rmbh:"rMBH", mbh:"MBH"
        },
        filenames: { overlay_suffix:"_overlay.png", unknown_suffix:"_UNK" }
      };

      try {
        const res = await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        alert('✅ Konfiguration gespeichert');
      } catch (e) {
        console.error(e);
        alert('❌ Fehler beim Speichern');
      }
    }

    async function startAnalysis() {
      const btn = document.getElementById('start-btn');
      btn.disabled = true; btn.textContent = '⏳ Läuft...';
      const payload = {
        paths: {
          base_results_dir: document.getElementById('input_dir').value.trim(),
          output_dir: document.getElementById('output_dir').value.trim()
        },
        dataset_name: document.getElementById('dataset_name').value.trim(),
        enable_analysis: document.getElementById('enable_analysis').value === 'true',
        analysis_params: {
          distance_threshold: parseInt(document.getElementById('distance_threshold').value, 10) || 30,
          use_auto_threshold: document.getElementById('use_auto_threshold').value === 'true',
          calibration_k_first: parseInt(document.getElementById('calibration_k_first').value, 10) || 8,
          safety_multiplier: parseFloat(document.getElementById('safety_multiplier').value) || 1.25,
          max_factor_diameter: parseFloat(document.getElementById('max_factor_diameter').value) || 2.0,
          min_pairs: parseInt(document.getElementById('min_pairs').value, 10) || 10,
          smooth_window: parseInt(document.getElementById('smooth_window').value, 10) || 5
        },
        detection: { active_hsv_range_index: parseInt(document.getElementById('active_hsv_range').value, 10) || 0 }
      };

      try {
        const res = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        pollStatus();
      } catch (e) {
        alert('❌ Start fehlgeschlagen'); btn.disabled = false; btn.textContent = '▶️ Analyse starten';
      }
    }

    async function pollStatus() {
      const log = document.getElementById('log');
      const btn = document.getElementById('start-btn');
      const timer = setInterval(async () => {
        try {
          const res = await fetch('/api/status');
          const st = await res.json();
          log.innerHTML = (st.log || []).map(l => '<div>' + l + '</div>').join('');
          log.scrollTop = log.scrollHeight;
          if (!st.running) {
            clearInterval(timer);
            btn.disabled = false; btn.textContent = '▶️ Analyse starten';
          }
        } catch {}
      }, 1500);
    }

    async function loadResults() {
      const cont = document.getElementById('results-list');
      cont.innerHTML = '<div class="muted">Lade…</div>';
      try {
        const res = await fetch('/api/results');
        const arr = await res.json();
        if (!Array.isArray(arr) || arr.length === 0) {
          cont.innerHTML = '<div class="muted">Keine Ergebnisse gefunden.</div>';
          return;
        }
        cont.innerHTML = arr.map(f => (
          '<div class="item">' +
            '<div><strong>📄 ' + f.name + '</strong><div class="muted">' +
            (f.size/1024).toFixed(1) + ' KB • ' + f.modified + '</div></div>' +
            '<a class="btn" href="/api/download/' + encodeURIComponent(f.name) + '">⬇️ Download</a>' +
          '</div>'
        )).join('');
      } catch {
        cont.innerHTML = '<div class="muted">Fehler beim Laden.</div>';
      }
    }

    // Init
    loadConfig();
  </script>
</body>
</html>
"""

def main():
    tpl_dir = os.path.join(os.getcwd(), "templates")
    os.makedirs(tpl_dir, exist_ok=True)
    out = os.path.join(tpl_dir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HTML_CONTENT)
    # Sanity check
    ok = HTML_CONTENT.strip().endswith("</html>")
    size = os.path.getsize(out)
    print(f"✅ index.html geschrieben: {out} ({size} Bytes)")
    if not ok:
        print("⚠️ Warnung: Inhalt endet nicht mit </html> – bitte prüfen!")

if __name__ == "__main__":
    main()
