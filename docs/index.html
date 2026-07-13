<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trend Energy ETF Scanner</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <h1>Trend Energy ETF Scanner</h1>
    <p id="meta">Loading...</p>
  </header>

  <section class="tabs">
    <button id="scannerTabBtn" class="tab-btn active" data-tab="scanner">Scanner</button>
    <button id="activeSignalsTabBtn" class="tab-btn" data-tab="activeSignals">Active Signals</button>
  </section>

  <section class="controls">
    <label><input type="checkbox" id="eligibleOnly" checked /> Eligible universe only</label>
    <label><input type="checkbox" id="signalOnly" /> Surge signal only</label>

    <label>
      Preset
      <select id="presetFilter">
        <option value="basic" selected>기본형</option>
        <option value="exploratory">탐색형</option>
        <option value="strict">엄격형</option>
        <option value="custom">Custom</option>
      </select>
    </label>

    <label>
      Group
      <select id="groupFilter">
        <option value="All">All</option>
      </select>
    </label>

    <label>Min R63 <input type="number" id="minR63" step="0.01" value="0.03" /></label>
    <label>Min ER63 <input type="number" id="minER63" step="0.01" value="0.20" /></label>
    <label>Min SurgeRatio <input type="number" id="minSurge" step="0.05" value="1.10" /></label>
    <label>Max ATR20% <input type="number" id="maxATR" step="0.01" value="0.06" /></label>
    <label>DollarVol Rank <= <input type="number" id="maxDollarRank" step="50" value="500" /></label>

    <button id="resetBtn">Reset</button>
    <a href="data/latest.csv" download>Download CSV</a>
  </section>

  <section class="preset-help">
    <strong>Preset:</strong>
    탐색형 = 후보 넓게 보기,
    기본형 = 기본 추천값,
    엄격형 = 강한 후보만 보기.
    수치를 직접 바꾸면 Custom으로 전환됩니다.
  </section>

  <section id="scannerPanel" class="tab-panel active">
    <section class="table-wrap">
      <table id="scannerTable">
        <thead>
          <tr>
            <th data-key="symbol">Symbol</th>
            <th data-key="name">Name</th>
            <th data-key="asset_group">Group</th>
            <th data-key="aum">AUM</th>
            <th data-key="dollar_vol_rank">DV Rank</th>
            <th data-key="close">Close</th>
            <th data-key="r63">R63</th>
            <th data-key="r126">R126</th>
            <th data-key="er63">ER63</th>
            <th data-key="te63">TE63</th>
            <th data-key="te126">TE126</th>
            <th data-key="score">Score</th>
            <th data-key="surge_ratio">Surge</th>
            <th data-key="atr20_pct">ATR20%</th>
            <th data-key="signal_surge_v0">Signal</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>
  </section>

  <section id="activeSignalsPanel" class="tab-panel">
    <section class="active-help">
      <strong>Active Signals:</strong>
      현재 Signal TRUE인 ETF만 표시합니다.
      Signal Age는 현재 이어지는 TRUE streak 기준입니다.
      Suggested Stop은 최근 20거래일 저점 기준이며, 영웅문 자동감시 매도 가격으로 옮기기 쉬운 보조값입니다.
    </section>

    <section class="table-wrap">
      <table id="activeSignalsTable">
        <thead>
          <tr>
            <th data-key="symbol">Symbol</th>
            <th data-key="name">Name</th>
            <th data-key="asset_group">Group</th>
            <th data-key="signal_streak_start_date">First Signal</th>
            <th data-key="signal_streak_trading_days">Signal Days</th>
            <th data-key="signal_streak_calendar_days">Calendar Days</th>
            <th data-key="is_first_signal_today">New?</th>
            <th data-key="close">Close</th>
            <th data-key="suggested_stop">Suggested Stop</th>
            <th data-key="stop_distance_pct">Stop Dist.</th>
            <th data-key="r63">R63</th>
            <th data-key="er63">ER63</th>
            <th data-key="score">Score</th>
            <th data-key="surge_ratio">Surge</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </section>
  </section>

  <script src="app.js"></script>
</body>
</html>
