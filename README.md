# Trend Energy ETF Scanner Starter

> **추세추종 전략 연구실 (백테스트·강건성 도구):** `추세추종 연구실 시작.cmd`를 실행하세요. 사용법은 [lab/README.md](lab/README.md).
> 아래 내용은 매일 자동으로 도는 ETF 스캐너(GitHub Pages)에 대한 설명입니다.

Static ETF scanner built with:

- GitHub Actions for scheduled calculation
- GitHub Pages for the static web UI
- Nasdaq Trader symbol directory for ETF symbol candidates
- Manual or semi-manual AUM CSV for ETF scale filtering
- yfinance for daily OHLCV data

## 1. Files to edit first

### `config/aum.csv`

Replace the sample rows with your ETF AUM universe.

Required columns:

```csv
symbol,aum,asset_group,category
SPY,650000000000,equity_broad,broad_us
QQQ,330000000000,equity_broad,nasdaq_100
```

AUM should be numeric USD. The scanner will keep `aum_top_n` from `config/universe.yml`.

### `config/universe.yml`

Controls AUM rank, dollar-volume rank, minimum history, and liquidity thresholds.

### `config/exclusions.yml`

Keyword-based exclusion rules for leveraged, inverse, volatility, single-stock, option-income, and buffer ETFs.

### `config/manual_overrides.csv`

Use this to restore false exclusions or force exclusions.

## 2. Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_daily_scan
```

Then open:

```text
web/index.html
```

If browser blocks local JSON fetch, run a tiny static server:

```bash
cd web
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## 3. GitHub Pages setup

1. Create a GitHub repository.
2. Upload these files.
3. Go to `Settings -> Pages`.
4. Under `Build and deployment`, choose `Deploy from a branch`.
5. Branch: `main`.
6. Folder: `/web`.
7. Save.

The site URL will usually be:

```text
https://<your-github-id>.github.io/<repo-name>/
```

## 4. GitHub Actions setup

1. Go to the repository `Actions` tab.
2. Select `Daily ETF Scan`.
3. Click `Run workflow` once manually.
4. Confirm that `web/data/latest.json` and `web/data/latest.csv` were updated.
5. The scheduled job runs on weekdays at 23:30 UTC.

## 5. Notes

- yfinance is suitable for research/MVP use, not a guaranteed institutional data feed.
- AUM sourcing is intentionally separated into `config/aum.csv` so you can swap the provider later.
- The signal logic is intentionally simple and configurable later. The important output is the full feature table.
