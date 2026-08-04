# thermal-panel-inspector

Detects and severity-classifies thermal anomalies (hotspots) in radiometric FLIR
images of electrical panels and systems — loose connections, overloaded circuits,
failing breakers, imbalanced phases, etc.

## How it works

1. Extracts the per-pixel temperature array embedded in a radiometric FLIR JPEG
   (via [`flyr`](https://pypi.org/project/flyr/)).
2. Estimates an ambient/reference temperature (25th percentile of the frame by
   default — override with `--ambient` if you have a known-good reference reading).
3. Flags connected regions whose temperature rise (ΔT) above ambient exceeds a
   threshold (`--min-delta`, default 8°C) and are larger than sensor noise
   (`--min-area`, default 25px).
4. Classifies each hotspot by ΔT using NETA/NFPA-70B-style severity tiers:

   | ΔT above ambient | Severity             | Typical guidance          |
   |-------------------|-----------------------|----------------------------|
   | 0–10°C            | `minor`               | Possible deficiency, monitor |
   | 10–20°C           | `serious`             | Probable deficiency, schedule repair |
   | 20–40°C           | `critical`            | Major discrepancy, repair ASAP |
   | >40°C             | `critical_immediate`  | Imminent failure risk, repair now |

5. Writes an annotated image (boxes + labels, color-coded by severity) and a
   CSV/JSON report per run.

These thresholds are a common convention, not a substitute for engineering
judgment — always corroborate with load current and component-to-component
(phase-to-phase) comparison where possible.

## Setup

```bash
pip install -r requirements.txt
```

Some FLIR camera models store radiometric data in a way `flyr` can only parse
with `exiftool` installed and on PATH. If extraction fails, install exiftool
and retry before assuming the file isn't radiometric. Also note: many cameras
save a paired non-radiometric visual-only photo alongside each thermal capture
(same filename series, no embedded temperature data) — that's expected to fail
extraction, it's not a bug.

## Usage

Single image:

```bash
python -m thermal_inspector.cli panel_001.jpg -o results
```

Folder (batch), with an explicit ambient reference:

```bash
python -m thermal_inspector.cli ./site_survey/ -o results --ambient 24.5
```

Output in `results/`:
- `<image>_annotated.png` per input image
- `report.csv` / `report.json` — one row per detected hotspot
- `summary.json` — counts by severity and the single worst hotspot found

### Excluding background with `--roi`

If the frame includes anything besides the panel itself — a wall, an open
door, a neighboring enclosure — that background can be warmer than the panel
and get flagged as a false-positive "hotspot," or skew the ambient estimate.
Crop detection to the panel enclosure with `--roi X,Y,W,H` (pixel coordinates):

```bash
python -m thermal_inspector.cli panel_001.jpg -o results --roi 0,0,120,120
```

The annotated output still shows the full frame, with a thin white outline
marking the searched region. For a folder, the same ROI is applied to every
image, so only use it when all images share the same framing (e.g. a fixed
camera position across a series). If images in the same batch need different
crops, call the library functions directly per image instead (see below) and
pass the combined results to `generate_pdf_report`.

### PDF report

Add `--pdf` to also write a client-facing `report.pdf` alongside the CSV/JSON,
with the annotated image and hotspot table for each analyzed file:

```bash
python -m thermal_inspector.cli ./site_survey/ -o results --pdf --pdf-title "Q3 Site Survey"
```

A detector can't tell a real fault from a camera artifact (a reflection off
bare metal, a warm tool left in frame, background scenery). To record that
kind of field judgment in the report, pass `--notes-file notes.json` with a
filename-to-note mapping:

```json
{
  "panel_003.jpg": "Hotspot near the door hinge is likely a reflection off bare metal, not a fault - verify from another angle."
}
```

The note is appended under that image's table.

## Library use

```python
from thermal_inspector import load_radiometric, find_hotspots, annotate_image

thermogram = load_radiometric("panel_001.jpg")
hotspots, ambient_c = find_hotspots(
    thermogram.temperature_c, min_delta_c=8.0, roi=(0, 0, 120, 120)
)

for h in hotspots:
    print(h.severity, h.delta_t_c, h.bbox)
```

Building a PDF report programmatically (e.g. with a different ROI per image,
which the CLI's single `--roi` flag can't express for a batch):

```python
from thermal_inspector import (
    load_radiometric, find_hotspots, annotate_image,
    hotspots_to_rows, ImageReportEntry, generate_pdf_report,
)
import cv2

entries = []
for filename, roi in [("panel_a.jpg", (0, 0, 120, 120)), ("panel_b.jpg", None)]:
    thermogram = load_radiometric(filename)
    hotspots, ambient_c = find_hotspots(thermogram.temperature_c, roi=roi)
    annotated_path = f"{filename}.annotated.png"
    cv2.imwrite(annotated_path, annotate_image(thermogram, hotspots, roi=roi))
    entries.append(ImageReportEntry(
        image_name=filename,
        annotated_image=annotated_path,  # a Path works; raw PNG bytes work too
        hotspot_rows=hotspots_to_rows(filename, hotspots),
        ambient_c=ambient_c,
    ))

generate_pdf_report(entries, "report.pdf", title="Site Survey")
```

## Web UI

```bash
pip install -r webapp/requirements.txt
```

The web UI is multi-user: every request requires login, and what an account
can do depends on its role.

| Role | Can do |
|---|---|
| `admin` | Everything below, plus create/deactivate users and change roles at `/admin` |
| `inspector` | Run analyses, generate/download PDF reports |
| `viewer` | Browse history and download PDF reports — cannot run new analyses |

No account exists until you create one. Bootstrap the first admin:

```bash
python -m webapp.manage create-user <username> --role admin
```

(prompts for a password; add `--password <pw>` instead for non-interactive/scripted
use, e.g. a container entrypoint — but prefer the prompt when a human is typing it,
since `--password` leaves the value in shell history). Manage users from the
command line any time with `list-users`, `set-role`, `deactivate`, `activate`
(run `python -m webapp.manage --help`), or from `/admin` in the browser once
logged in as an admin.

Then run the server:

```bash
uvicorn webapp.server:app --reload
```

Open http://127.0.0.1:8000 and log in. Select one or more radiometric FLIR
JPEGs — drag-and-drop multiple files, click "Choose files" for a multi-select
dialog, or "Choose folder" to grab every JPEG in a directory at once
(non-.jpg/.jpeg files in a selected folder are silently filtered out
client-side). Adjust ambient/ΔT/area/ROI parameters in the form — ROI, if
set, applies to every selected file, so only use it when they share the same
framing. Run once with no ROI first — the result shows the first image's
pixel dimensions, which you need to pick sensible ROI coordinates for a
follow-up run.

Every analysis is saved automatically — browse past runs at `/history`,
where any logged-in role (including viewer) can view results and download a
PDF. After analyzing (or from a history entry), fill in the optional "Report
details" fields (client, site/location, audit date, inspector, report ID)
and pick a report style (Full or Audit), then click **Download PDF**. Any
field left blank is simply omitted from the report header — audit date
defaults to today if not set.

### Data and security notes

- Accounts and analysis history live in a local SQLite database at
  `webapp/data/app.db`; annotated images are stored under `webapp/data/runs/`.
  Nothing here is committed to git (`webapp/data/` is gitignored) — back it up
  yourself if the history matters to you.
- Sessions are signed cookies (Starlette's `SessionMiddleware`). The signing
  key is read from the `SESSION_SECRET_KEY` environment variable if set;
  otherwise one is generated and persisted to `webapp/data/.session_secret`
  so sessions survive a restart. Set `SESSION_SECRET_KEY` explicitly for any
  deployment that might redeploy to a fresh working directory, or that runs
  more than one server process.
- Set `SESSION_HTTPS_ONLY=true` once the app is actually served over HTTPS
  (it isn't inferred automatically — the app itself sees plain HTTP even when
  a proxy/edge terminates TLS in front of it). The `fly.toml` below sets this
  for you.
- Login attempts are rate-limited per client IP (5 failures / 5 minutes,
  in-memory — resets on restart, which is fine for a single small-team
  instance). There's still no CSRF token beyond `SameSite=Lax` cookies, which
  is adequate for this app's threat model but worth knowing.

## Deployment (Fly.io + Supabase)

This app is a Python process that needs somewhere to persist data — it
doesn't run as serverless functions, which rules out Netlify/Vercel-style
platforms without a full rewrite (Netlify Functions specifically don't
support Python at all). `Dockerfile`, `.dockerignore`, and `fly.toml` are
included for [Fly.io](https://fly.io), which runs a normal container.

Storage has two modes, and which one is active is decided entirely by
environment variables — no code changes either way:

- **Supabase** (recommended for anything beyond solo/local use): set
  `DATABASE_URL` to a Supabase Postgres connection string, and `SUPABASE_URL`
  / `SUPABASE_SERVICE_KEY` for Storage. The app then has no local state at
  all, which also means it's safe to run more than one machine if you ever
  need to (not configured here, since a small team doesn't need it, but
  nothing stops you).
- **Local fallback** (no Supabase env vars set): SQLite + images live on a
  Fly volume. Simpler to set up, but **single-instance only** — a Fly volume
  attaches to one machine, so `fly scale count 2` would split your data
  across two empty-ish volumes instead of sharing it.

### Supabase setup

I can't create the Supabase project for you (account creation isn't
something I can do), but once you have one:

1. In your Supabase project: **Settings → Database** for the connection
   string (`DATABASE_URL` — use the pooler connection string, not the direct
   one, since Fly machines connect/disconnect on scale-to-zero), and
   **Settings → API** for the project URL and `service_role` key (`SUPABASE_URL`
   / `SUPABASE_SERVICE_KEY` — the service role key bypasses row-level
   security, which is what a trusted backend needs; never expose it to a
   frontend).
2. In **Storage**, create a bucket (default name the app expects:
   `thermal-images`, or set `SUPABASE_STORAGE_BUCKET` to whatever you name it).
3. Tables are created automatically on first request (`init_db()` calls
   `Base.metadata.create_all`) — no manual migration step needed for this
   app's simple schema.

### Deploy steps

```bash
# 1. Install flyctl, then authenticate (opens a browser)
fly auth login

# 2. Edit fly.toml: set `app` to something globally unique on Fly, and
#    `primary_region` to whatever's nearest your team (see: fly platform regions)

# 3. Create the app
fly apps create <your-app-name>

# --- Supabase mode: skip the volume, set these instead ---
fly secrets set \
  DATABASE_URL="<supabase-pooler-connection-string>" \
  SUPABASE_URL="https://<project>.supabase.co" \
  SUPABASE_SERVICE_KEY="<service-role-key>" \
  -a <your-app-name>

# --- Local-fallback mode: create the volume instead of the above ---
# fly volumes create thermal_data --size 1 --region <your-region> -a <your-app-name>

# 4. Set the session signing key as a real secret (generate it locally first)
python -c "import secrets; print(secrets.token_hex(32))"
fly secrets set SESSION_SECRET_KEY=<paste-the-generated-value> -a <your-app-name>

# 5. Deploy (Fly builds the image remotely, so local Docker isn't required)
fly deploy

# 6. Create the first admin account on the running instance
fly ssh console -a <your-app-name> -C "python -m webapp.manage create-user admin --role admin --password <a-strong-password>"
```

Then visit `https://<your-app-name>.fly.dev` and log in. Manage further users
the same way (`fly ssh console -a <your-app-name>`, then use the `webapp.manage`
CLI as documented above) or from `/admin` once logged in.

**A note on testing this specific path**: I verified the Supabase-mode code
(engine selection, upload/download calls) against a mocked client and
confirmed the local fallback still works end-to-end, but I don't have a real
Supabase project to test against — I'd recommend a quick smoke test (log in,
run one analysis, download a report) right after your first deploy.
