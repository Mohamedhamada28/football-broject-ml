# Football Model Dashboard

## What is included

- `dashboard_server.py`
- `football_model_refresh.py`
- `webapp/`
- `outputs/model_refresh/` training outputs and exported model artifacts
- `top5-players.csv`
- `players_data-2024_2025.csv`
- `requirements.txt`
- `Procfile` and `render.yaml` for web deployment

## Run

1. Create a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
python dashboard_server.py --host 127.0.0.1 --port 8000
```

4. Open `http://127.0.0.1:8000`.http://localhost:8000

## Deploy Online

The server now binds to `0.0.0.0` by default and reads the `PORT` environment variable, so it can run behind a public hosting proxy.

Render-style deployment:

```powershell
pip install -r requirements.txt
python dashboard_server.py --host 0.0.0.0 --port $env:PORT
```

On Linux hosts or with the included `Procfile`:

```bash
python dashboard_server.py --host 0.0.0.0 --port "$PORT"
```

After deployment, use the public service URL shown by the host.

## Added Analytics

- Player similarity and top similar players
- Player recommendation system
- Player search
- Compare two players
- Best XI generator for 4-3-3 and 4-2-3-1, with existing extra formations kept
- Scout report generator
- Player role detection
- Player clustering with K-Means and DBSCAN
- Explainable AI panel using local SHAP-style feature attribution for the exported sklearn pipeline

## Market Value and Injury Risk

The current datasets do not include market value, transfer fee, wage, contract, injury outcome, medical history, or workload-history labels. The dashboard therefore shows a dataset-readiness explanation instead of fake market value or injury risk predictions.

## Notes

- The web app reads the saved CSV outputs and exported `.joblib` models in `outputs/model_refresh/`.
- `football_model_refresh.py` is included because the dashboard server imports it for shared dataset/model helpers.
- `xgboost` is listed in the requirements because that module imports it, even though the dashboard itself uses the exported sklearn models.
