# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 사용자

- 연구자이며 Git/웹 개발 전문가는 아님. 한국어로, 변경 설명은 **무엇을 / 왜 / 어디에 영향** 순서로.
- 코드를 요청하면 부분 snippet보다 그대로 쓸 수 있는 완성 코드를 선호.
- 변경은 브랜치 → PR → 사용자가 GitHub에서 병합. `main`에 직접 커밋하지 않는다.

## 새 세션을 시작하면

세션끼리 대화 내용은 공유되지 않는다. 맥락은 이 파일과 아래 기록으로 이어진다.

1. **지금 어디까지 왔는지**: `docs/research/lab/`에서 날짜가 가장 최근인 메모. 끝의 "남은 판단" 절이 사용자 결정 대기 항목이다.
2. **왜 그렇게 정했는지**: `docs/research/trend_v2/DECISIONS.md` (번호순, 19번부터 lab).
3. 연구를 대신 수행하거나 기준을 바꿨다면 1·2를 같은 PR에서 갱신한다. 다음 세션이 읽는 곳이 여기뿐이다.
4. 자산배분(포트폴리오 비중·리밸런싱·레버리지) 연구는 `docs/research/allocation/`에 따로 쌓인다: `NOTES.md`(발견·경고·캠페인), `DECISIONS.md`(A번호), 날짜별 메모.

## 명령어 (Windows, 저장소 루트, `.venv`는 Python 3.13)

```bash
# 연구실 앱 실행 (사용자는 "추세추종 연구실 시작.cmd" 더블클릭)
.venv/Scripts/python.exe -m streamlit run lab/app.py        # http://127.0.0.1:8501
.venv/Scripts/python.exe -m streamlit run lab/aa_app.py     # 자산배분 연구실, http://127.0.0.1:8511 ("자산배분 연구실 시작.cmd")

# 테스트 — 반드시 `python -m pytest` (루트를 sys.path에 넣어 `src`, `lab` import가 됨. pytest 설정 파일 없음)
.venv/Scripts/python.exe -m pytest -q tests/test_lab.py                       # lab 변경 시
.venv/Scripts/python.exe -m pytest -q tests/test_lab.py::test_gates_and_loyo  # 단일 테스트
.venv/Scripts/python.exe -m pytest -q -W error::FutureWarning                 # 전체 (~1.5분, CI와 동일 플래그)

# 런처 자동 업데이트 (두 런처와 두 북마크가 앱을 켜기 전에 실행. 직접 돌려도 됨)
.venv/Scripts/python.exe scripts/lab_update.py

# 레거시 파이프라인 (스캔은 GitHub Actions가 매일 실행, 백테스트는 Actions 탭에서 수동 실행만; 로컬 실행은 docs/data를 덮어쓰므로 하지 말 것)
python -m src.run_daily_scan && python src/postprocess_groups.py
python -m src.run_backtest_only
```

CI(`.github/workflows/tests.yml`)는 PR마다 **Python 3.11**, pandas 2.2와 3.x 두 버전으로 `-W error::FutureWarning` 전체 스위트를 돈다. 로컬 venv는 3.13이므로 3.12+ 전용 문법과 pandas FutureWarning을 피할 것.
린터/포매터 설정은 없다.

## 저장소 구조: 여섯 개의 하위 시스템

| 시스템 | 위치 | 상태 |
|---|---|---|
| **연구실 (lab)** | `lab/`, `tests/test_lab.py` | **현재 주력 제품.** 새 기능은 여기에 |
| 자산배분 연구실 | `lab/aa_*.py`, `scripts/aa_link.py`, `scripts/aa_*_study.py`·`scripts/aa_broad_search.py`·`scripts/aa_global_search.py`·`scripts/aa_stage2.py`(재현 스크립트), `tests/test_aa.py`, `tests/test_aa_korea.py`, `tests/test_aa_tactical.py`, `tests/test_aa_search.py`, `tests/test_aa_search_results.py`, `tests/test_aa_real.py`, `tests/test_aa_local.py`, `tests/test_aa_trend_filter.py`, `docs/research/allocation/` | 별도 Streamlit 앱(포트 8511, 북마크 `aalab://`). 국내 상장 ETF(원화)로 자산 비중·리밸런싱·overlay·레버리지·전략 믹스 연구, ISA 리밸런싱 계산기. 전술 전략(HAA·BAA·VAA, `aa_tactical.py`)·검증 도구(`aa_validate.py`)·위기 회피 스위치(`aa_signals.py`)는 ‘전술 전략 · 검증’ 페이지, 전역 탐색 결과(A31)는 ‘전역 탐색 결과’ 페이지. 가격은 기본이 ‘실제 가격 우선’: `aa_real.py`가 상장 후 구간을 실제 국내 가격(`lab_data/kr/`, 앱이 20시간마다 자동으로 받음, `python -m lab.aa_real fetch`, 커밋 금지)으로 바꾸고 상장 전은 모형(미국 대용 × 원/달러, `aa_korea.py`)이다. 사이드바에서 ‘모형만’으로 바꿀 수 있다(A33·A34). 재현 스크립트는 `--prices real|model`(기본 model = 등록 당시) |
| Daily Alert | `lab/daily_alert.py`, `config/alert.yml`, `.github/workflows/daily_alert.yml` | 평일 장 마감 후 자동 실행. 이슈 "📊 일일 알림"에 댓글 |
| Daily ETF Scan | `src/run_daily_scan.py` → `universe`, `features`, `signal_history`, `update_aum`; 결과 `docs/data/`; 화면 `docs/index.html` (GitHub Pages) | 매일 자동 실행. 유지 |
| v1 Backtest Only | `src/run_backtest_only.py` → `src/backtest.py`, `src/portfolio.py`; 결과 `docs/data/backtest_*`; `docs/backtest_dashboard.js` | score-breakout 기반 레거시. 동결. 자동 실행 꺼짐(수동 실행만, DECISIONS 44) |
| v2 Foundation | `src/trend_v2_foundation/`, `src/trend_v2*.py`, `scripts/run_trend_v2_*.py`, `config/trend_v2/` | 보존만. Codex용 규칙은 `AGENTS.md` |

- `web/`은 초기 스타터의 오래된 사본(2026-07 이후 갱신 없음)이다. Pages 사이트와 워크플로가 쓰는 곳은 `docs/`. README의 "Folder: /web" 안내는 낡은 내용.
- `docs/data/`, `config/aum.csv`의 커밋은 GitHub Actions 봇("Update ETF scan data" / "Update backtest data")이 만든다. 손으로 고치지 않는다.

### 해시로 기록된 파일 (건드리지 않는다)

`config/oos_evaluation_manifest.json`이 PR #18 OOS 기준선 기록으로 다음 파일의 git blob 해시를 적어 두었고, `tests/test_oos_evaluation_manifest.py`는 이 기록값이 바뀌지 않았는지 검사한다(현재 파일의 해시를 다시 계산하지는 않는다). 파일을 고쳐도 테스트는 깨지지 않지만 기준선 기록과 어긋나므로, 고칠 때는 DECISIONS에 이유를 남긴다:
`src/backtest.py`, `src/features.py`, `src/portfolio.py`, `src/universe.py`, `src/prices.py`, `src/run_daily_scan.py`, `src/run_backtest_only.py`, `config/universe.yml`, `config/exclusions.yml`, `config/manual_overrides.csv`, `.github/workflows/daily_scan.yml`, `.github/workflows/backtest-only.yml`, `scripts/verify_data_publish_base.py`, 그리고 `docs/data`의 일부 백테스트 산출물.
`lab/data.py`는 `src/prices.py`의 함수를 **가져다 쓰기만** 한다. 필요한 변경은 `lab/` 안에서 할 것. `backtest-only.yml`은 자동 실행을 끄느라 한 번 고쳤다(DECISIONS 44).

## lab/ 아키텍처

데이터 흐름: `universe`(필터 깔때기) → `data.Panel` → `engine.run_backtest` (+ `engine.baselines`) → `metrics.all_metrics` → (`grid.run_grid` / `grid.strategy_checklist`) → `store`, `research_log` → `app.py` 화면.

- **`universe.py` / `etf_meta.py`**: ETF 필터 깔때기(현금성 제외 → 자산군 → AUM → 총보수 → 상관 ≥ 0.98 중복 정리). 각 단계 제외 사유를 기록해 화면에 보여준다.
  ETF 메타데이터(총보수·AUM·Morningstar 분류)는 `lab/etf_meta.csv`(커밋됨)에서 읽는다. 갱신: `.venv/Scripts/python.exe -m lab.etf_meta`.
  분류 → 자산군 매핑은 `asset_class()`의 키워드 목록이다. 새 분류가 '기타'로 빠지면 여기를 고친다.
- **`families.py`**: 사전 등록된 고정 메뉴(v3: 진입 11 × 청산 7 = 77개 계열, 각 6×6)로 `run_grid`를 계열마다 돌려 격자 전체 성적으로 비교한다(DECISIONS 33–34, 45, 59).
  200칸 이상이면 계열을 여러 프로세스(`default_workers`: 코어 수, 최대 8. 환경변수 `LAB_WORKERS=1`이면 끔)로 나눠 돌린다. 결과와 `on_result` 호출 순서(시험 기록 순서)는 한 프로세스로 돌릴 때와 같다.
  작업 프로세스는 spawn으로 뜨므로 앱 스크립트를 다시 실행하지 않도록 `_main_script_hidden`이 `__main__.__file__`을 잠시 숨긴다.
  메뉴 값은 결과를 본 뒤 고치지 않는다. 바꾸려면 `MENU_VERSION`을 올려 새 메뉴로 등록한다.
  공통 조건 시험(`run_addon_test`): 블록 하나를 기본값으로 모든 계열에 더해 계열별 파레토 판정을 센다(DECISIONS 47). `verdict`도 여기 있다.
  국면 전환(`kind="switch"`): 국면 진입 필터와 그 정반대 청산(`blocks.REGIME_PAIRS`)을 함께 더한다(DECISIONS 50–51).
- **`macro.py`**: 매크로 국면 블록(블록 분류 "매크로")이 읽는 FRED 일간 시리즈(T10Y3M, BAA10Y, VIXCLS). 연구는 커밋된 `lab/macro_snapshot.csv.gz`만 읽는다(갱신: Actions "Refresh macro snapshot" 수동 실행 또는 `python -m lab.macro`).
  t일 결정에는 t−1일 이전 값만 쓴다(공표 시차). 사후 수정되는 지표는 넣지 않는다(DECISIONS 49).
- **`single.py`**: 종목별 단독 시험(DECISIONS 67). 엔진을 `single=True`(슬롯 = 종목 수, 현금 제한 없음, numpy 경로)로 한 번 돌려 종목별 단독 계좌와 매매 기록 포트폴리오(sleeve)를 만든다. 한 종목 백테스트와 매매가 같다는 테스트가 있다.
- **`target.py`**: 사용자 목표 지표. 레버리지 없이 낙폭 15%에 맞춘 비중·수익, 같은 낙폭의 SPY+국채 혼합, 3년 롤링 구간 승률(DECISIONS 64–65). `explore.py`: Sobol 표본 탐색과 PBO(66).
- **`crsp.py` / `crsp_study.py`**: CRSP 일별 파일(사용자 드라이브 `H:\내 드라이브\CRSP`, 커밋 금지) → 시점별 시총 상위 N 패널(`Panel.eligible`, `delist_exit`) → 종목별 단독 시험·Sobol을 중간 저장하며 실행(DECISIONS 70–71). 실행은 노트북 RAM을 아끼려고 **Colab**(`colab/crsp_study.ipynb`, 드라이브 연결)에서 하고, 결과는 드라이브 `CRSP/lab_crsp/`로 돌아온다. `--dry`로 파이프라인만 점검할 수 있다.
  엔진의 `Panel.eligible`은 진입만 막는다(보유 종목은 청산 규칙대로). `delist_exit=True`면 가격 시계열이 기간 중 끝나는 종목을 마지막 봉에서 강제 청산한다(`DELIST_KEY`). 두 기능 모두 기존 패널(`None`/`False`)에서는 결과를 바꾸지 않는다(2026-10-23 비트 단위 확인).
- **`screens.py` / `screen_study.py`**: 종목 스크린(DECISIONS 77). 제외 필터(True = 사지 않음)와 선별 점수(높을수록 좋음, 평균 백분위 순위로 결합, 상위 K)를 매월 말 결정해 다음 달 `Panel.eligible`·`Panel.rank`로 넣는다. 엔진은 `rank`가 있으면 거래대금 대신 순위 순으로 산다. 시총·배당은 `Panel.extra["cap"|"dividend"]`(CRSP 패널만). `screen_study.py`는 제외 × 선별 조합을 계좌 단위로 비교한다(중간 저장, `--dry`). 새 스크린은 `HYGIENE_SCREENS`/`EXCLUDE_SCREENS`/`SELECT_SCORES`에 넣으면 되고, 등록 메뉴 값은 `screen_study.EXCLUDE_MENU`·`K_MENU`.
  2026-10-25 재구성: 위생(상장 기간·주가·거래대금·Amihud·시총) → 제외(Minervini 추세 템플릿 `tt_fail`, 단기 과열, 약한 업종, 고유 변동성 …) → 선별(IBD식 RS 등급 `s_rs`, 매집/분산, 업종 RS, 52주 고가 × 거래량 …). 실무자 조합은 `PROFILES`(프로필 하나가 시험 단위). 선별 점수는 42개이고 `tier`(1 가격·거래량 직접, 2 일차 파생, 3 규모·배당·업종)로 우선순위를 표시한다(`scores_by_tier()`). 스타일 점수(소형·대형·배당·반전·저변동성)는 `STYLE_TILTS`로 표시하되 선별 기준으로 유지한다(사용자: 소형주를 빼지 말 것). 위생 기본값에는 시총 하한이 없다. 업종 스크린은 `panel.info["sic"]`(CRSP 패널, 2026-10-25 이후 빌드)가 필요하다. 설계: `docs/research/lab/2026-10-25_screens_v2_design.md`.
  `score_study.py`(DECISIONS 79): 선별 점수 42개를 기본값으로 하나씩, (제외 없음/추세 템플릿) × K {20,50,100} × 진입 2 × 매매층 2(보유형 / 교체형 = 청산 블록 `left_list`) × 표본 2. 스크린은 한 번 계산해 `out/screens/`에 저장. 원격 실행은 `crsp_remote.py --study scores`.
  `left_list`는 `Panel.eligible`을 읽으므로 `SignalCache.PANEL_SPECIFIC`로 캐시끼리 공유하지 않는다.
  `combo_study.py`(DECISIONS 80, Stage B): 상위 점수 6개의 2·3개 조합 + 모멘텀 단독(36개) × 모든 조합에 같은 매매층 Sobol 32점(`point_settings`) × 표본 2. ATR 손절이 없는 점은 위험 기준 대신 동일 금액 슬롯으로 산다(초기 손절이 없으면 위험 기준은 아무것도 사지 않음). 끝나면 `figures/`에 메모용 PNG 4장을 만든다(`lab/figures.py`, matplotlib이 없으면 건너뜀). 원격 실행은 `crsp_remote.py --study combos`.
  `stage_c_study.py`(DECISIONS 81, Stage C): 일별 유니버스(`crsp.build_panel(daily=True)` / `python -m lab.crsp --daily`: 매일 21일 평균 시총 상위 N, `extra["cap_rank"]`, 드라이브 `panel_daily3000`) + 위생 필터 매일 적용. 매수 10개(E0 조건 없음, E1–E9는 `blocks.STAGE_C_ENTRIES`의 '첫날' 신호, `first_day`) × 매도 9개(새 블록 `chandelier`(ATR22, 엔진 경로 청산)·`psar_flip`·`rsi2_exit`와 기존 블록) × 3×3칸. C1 신호 전체(`single.evaluate(with_result=True)`, 거래 목록 저장) + C2 20종목 계좌(우선순위 거래대금 / 추세 효율+시총 / 무작위) + 놓친 신호 비교. 원격 실행은 `crsp_remote.py --study stagec`(일별 패널과 결과 폴더 `stagec_daily3000`, `--extra=--dry`면 `_dry`).
- **`scripts/crsp_remote.py` / `CRSP_remote_run.cmd`**: 연구실 컴퓨터용. 드라이브의 패널을 임시 폴더로 복사(없으면 생성)해 연구를 돌리고 결과를 드라이브 `CRSP/lab_crsp/<study>_top<N>`에 쓴 뒤 임시 폴더를 지운다. CRSP 데이터는 그 컴퓨터에 남지 않는다.
- **`batch.py`**: 앱 기본 유니버스·설정으로 공통 조건 시험을 명령줄에서 돌린다(`python -m lab.batch addon <kind> <key>`). 앱과 똑같이 시험 횟수에 기록하고 결과를 저장한다.
- **`daily_alert.py`**: 일일 알림(DECISIONS 53, 75). 네 시장 국면(SPY 200일선·금리차·신용 스프레드·VIX)의 상태, 지속 기간, 값의 뜻을 풀어 쓴 해설을 이슈 댓글로 보낸다. 모델 포트폴리오는 2026-10-24에 뺐다. 가격은 SPY만 받고, `trim_incomplete_tail`로 당일 미완성 봉을 잘라 낸다(yfinance 당일 봉은 시각이 섞여 `data.normalize_dates`로 날짜를 맞춘다).
- **`refine.py`**: 확인 조건 시험. 한 계열에 진입 블록을 하나씩(기본값으로만) 더해 같은 격자를 다시 돌리고, 네 기준의 파레토 개선 여부로 판정한다(DECISIONS 35–36).
- 연구 기록: `docs/research/lab/` (날짜별 메모). 연구를 대신 수행했다면 결과·규칙·한계를 여기에 남긴다.
  **사용자 규칙(2026-10-26): 모든 시험·실험은 (1) 왜 했는지 (2) 핵심 결과 요약 (3) 그림을 남긴다.** 메모 맨 위에 `## 왜 했는지` / `## 핵심 결과` / `## 그림`을 두고, 그림은 집계 결과만 PNG로 `docs/research/lab/figures/<study>/`에 커밋한다(CRSP 원자료 금지, `lab/figures.py`, 끝난 연구는 `scripts/make_study_figures.py`). 같은 내용을 `docs/research/lab/experiments.json`(맨 앞이 최신)에 넣으면 앱의 ‘실험 기록’ 페이지(`lab/experiments.py`)가 보여 준다. 새 연구 실행기는 끝날 때 그림을 직접 만든다(`combo_study.write_figures` 참고).
- **`research_log.py`**: `lab_results/trials.csv`(시험한 전략 로그 → DSR의 N), `trial_returns.npz`(시험별 월간 수익률 → 평균 상관 ρ → 유효 N)와 `lab_results/holdout.json`(보류 구간 설정·변경·평가 기록).
  앱에서 실행한 백테스트·격자 칸·점검 변형은 모두 여기에 기록된다. `lab.batch`로 돌린 시험도 기록된다. 그 밖의 스크립트로 돌린 실험은 기록되지 않는다.
  ρ는 상관 행렬을 만들지 않고 열 합의 내적으로 구한다(2만여 개 시험에서 수십 초 → 1초 미만, 값은 반올림 오차 1e-15 수준까지 같음). 앱은 매 화면 갱신마다 묻지만 파일이 바뀌었을 때만 다시 읽는다.
  사용자의 연구 기록이므로 테스트하면서 생긴 항목은 지울 것. 파일 경로는 호출 시점에 `RESULTS_DIR`에서 만들어지므로, 테스트는 `research_log.RESULTS_DIR`만 임시 폴더로 바꾸면 된다(테스트가 `lab_results/`에 쓰면 안 된다).

- **`data.py`**: `Panel` = 날짜 × 종목의 wide DataFrame(open/high/low/close/volume) + `tradable` 목록 + `cash`(현금 일간 수익률: BIL, 2007년 이전은 ^IRX/252) + `kind`(저장 결과 재구성용). 벤치마크 SPY는 항상 데이터에 포함되지만, 선택한 유니버스에 없으면 매매 대상이 아니다(`Panel.subset`). 출처는 두 가지:
  - ETF 스냅샷: `docs/research/trend_v2/phase_a2/prices/*.csv.gz`(466개 ETF, 2016-08-01 ~ 2026-07-30, 읽기 전용). 첫 로드 때 `lab_data/snapshot_panel.pkl`로 캐시.
  - yfinance: 종목별 전체 이력을 `lab_data/yf/`에 캐시. 장기 멀티에셋 프리셋은 `LONG_HISTORY_ETFS`(2004년 이전 상장 38개).
- **`blocks.py`**: 진입 블록(AND)과 청산 블록(OR). 블록당 숫자 파라미터 1개, 가격·거래량(과 `macro.py`의 매크로 국면)만 사용.
  - `compute(panel, value) -> bool DataFrame`을 가진 블록은 엔진과 화면에 자동으로 연결된다.
  - `compute=None`인 청산 블록(트레일링·손절)은 포지션 상태가 필요해서 `engine.py` 루프에서 키로 처리한다. 이런 블록을 추가하면 엔진도 함께 고쳐야 한다.
- **`engine.py`**: 신호는 t일 종가로 계산하고 t+1일 시가에 체결. 동일 금액 슬롯(`max_positions`), 리밸런싱 없음, 편도 비용 bp, 신호 초과 시 20일 평균 거래대금 순으로 선택.
  - `SignalCache`가 블록 출력을 `(key, value)`로 재사용한다. 격자 속도의 핵심이다.
  - `RULES_KO`는 화면에 그대로 보여주는 규칙 설명이다. 엔진 동작을 바꾸면 함께 고칠 것.
  - 포지션 크기 `sizing`: `equal` | `inverse_vol`(기본, 60일 변동성·2배 상한) | `risk`(위험 기준: 초기 손절폭으로 주식 수 결정, `risk_pct`·`max_weight`). `to_dict()`는 기본값 `equal`일 때 키를 생략해 예전 시험 키를 유지한다. 새 설정 필드를 더할 때도 같은 방식으로.
  - 리스크 관리층(DECISIONS 72): `atr_stop` 청산 블록(신호일 ATR 기준 초기 손절), `heat`(열 상한), `breakeven_r`(손익분기 이동, 의사 청산 키 `BREAKEVEN_KEY`). 모두 기본값에서 꺼져 있다. 초기 손절 거리 R은 `risk_per_share`가 `atr_stop` 또는 `stop_loss_pct`에서 계산한다. 피라미딩은 아직 없다.
  - `time_stop`은 `first_exit`의 경로 청산이다. `take_profit_pct`(PARTIAL_KEY)는 전량 청산이 아니라 `first_partial`이 정한 날 다음 시가에 50%를 판다. 거래 목록은 포지션당 한 줄(두 매도를 합친 수익률), 부분 매도 기록은 `BacktestResult.partial_exits`.
  - 종목이 `LIST_MAX`(1,500)개를 넘으면 계좌 모드에서도 일간 루프가 리스트 대신 numpy 행을 읽는다(결과 동일, 테스트 있음). 같은 가격의 다른 패널(스크린만 다름)은 `SignalCache.share_from`으로 블록 출력을 공유한다.
  - 큰 패널에서 여러 규칙을 차례로 돌릴 때는 규칙이 바뀔 때 `SignalCache.drop_rule_data(keep_window)`로 규칙별 배열을 비운다(안 그러면 CRSP 3,700종목에서 수십 GB가 쌓임).
  - 일간 루프는 numpy 스칼라 대신 파이썬 리스트를 쓴다(수 배 빠름, 결과는 비트 단위로 동일). 기간(window)별 리스트는 `SignalCache.memo`로 격자 칸끼리 재사용한다.
  - 청산은 매일 보유 종목을 검사하지 않고, 매수하는 순간 `first_exit`가 그 종목의 이후 종가 배열에서 첫 청산일과 사유를 찾는다(규칙은 같음: 종가 없는 날 건너뜀, 고점은 매수가와 그날 종가까지 포함, 사유는 `cfg.exits` 순서).
    청산일은 (종목, 매수일)에만 달려 있어 같은 청산 설정의 모든 칸이 `exit_at` 메모를 공유한다. 포지션 상태가 필요한 청산 블록을 추가하면 `first_exit`에 넣을 것.
    루프를 고치면 고치기 전 결과를 저장해 두고 동일한지 비교할 것(2026-10-22 최적화 때는 28계열 2,016칸의 셀 표·자산 곡선·거래 내역을 비트 단위로 비교).
- **`metrics.py`**: `DEFINITIONS`가 지표 이름·뜻·계산식의 **단일 출처**다. 화면 툴팁, 결과 표, 용어 설명 페이지가 모두 여기서 읽는다. 지표를 추가하면 계산과 정의를 같이 넣을 것.
- **`grid.py`**: 1~2개 파라미터 격자(축당 최대 12값) 다음 순서로 판정한다:
  1. Gate (`check_gates`)
  2. 이웃 생존율·연결 영역(상하좌우 4방향 인접)
  3. LOYO (한 해씩 빼고 수익·낙폭 Gate 재판정)
  4. 사전식 정렬: 영역 크기 → 이웃 생존율 → LOYO → Calmar
- **`store.py`**: `lab_results/<시각>_<종류>_<이름>/`에 `meta.json` + CSV를 저장한다. `lab_data/`, `lab_results/`는 git 제외.
- **`app.py`**: Streamlit, `st.navigation`으로 페이지 5개(전략 연구, 최종 검증, 저장된 결과, 실험 기록, 용어 설명).
  - 위젯 기본값은 `init(key, default)`로 session_state에 한 번만 넣는다(`value=`와 key를 같이 쓰면 경고가 난다).
  - 페이지를 옮겨도 설정이 유지되도록 `PERSIST_PREFIXES`의 키를 매 실행 재할당한다. 새 위젯 키를 만들면 여기에 접두어를 추가할 것.

### 앱 작업 시 주의

- 화면을 바꾼 뒤에는 앱을 실제로 띄워 눌러 본 다음 완료라고 말한다. 서버를 띄운 채 `lab/*.py`(app.py 제외)를 고치면 이전 모듈이 남아 있으니 서버를 재시작할 것.
  사용자가 런처로 8501을 열어 두었을 수 있으니 확인용 서버는 8502 등 다른 포트로 띄운다. 화면 시험으로 생긴 `lab_results/` 항목은 백업해 두었다가 되돌린다.
- Streamlit markdown에서 `~` 두 개는 취소선이 된다. 범위는 `–`로 쓴다.
- 폭 지정은 `width="stretch"`를 쓴다(`use_container_width`는 지원 종료 예정).
- `.streamlit/config.toml`: `magicEnabled = false`(단독 표현식이 화면에 출력되는 것 방지), `address = 127.0.0.1`(외부 노출 방지).
- `.cmd` 런처의 echo 문구는 ASCII로만 쓴다. 한글 echo는 cmd 인코딩 문제로 깨진다(PR #44).
- 런처(`추세추종 연구실 시작.cmd`, `자산배분 연구실 시작.cmd`)와 북마크 핸들러(`scripts/lab_link.py`, `scripts/aa_link.py`)는 앱을 켜기 전에 `scripts/lab_update.py`로 GitHub의 `main`을 받아 **현재 브랜치에 병합**한다(checkout 안 함). 충돌·겹치는 미저장 수정·오프라인·진행 중인 병합이면 아무것도 바꾸지 않고 그대로 켠다. requirements가 바뀌면 패키지를 다시 설치한다. 그래서 사용자는 PR을 병합한 뒤 수동 `git pull` 없이 연구실만 다시 켜면 된다.
  업데이트가 실행 중인 `.cmd` 파일 자체를 바꿀 수 있으므로(cmd는 배치 파일을 줄 단위로 디스크에서 읽음) 업데이트와 앱 시작은 `exit /b`로 끝나는 `( … )` 블록 하나에 둔다. `tests/test_lab_update.py`가 이 구조를 검사한다.
- 사용자 화면에 내부 구조 용어(StrategyRun, hash 등)를 노출하지 않는다. 모든 지표·옵션에 정의를 붙인다.

## 자산배분 연구실 (`lab/aa_*.py`)

추세추종 연구실과 **별도 앱**이다. `lab/data.py`(가격 패널)와 `lab/metrics.py`(지표 공식)만 가져다 쓰고 `lab/app.py`·`engine.py`·`store.py`·시험 기록(DSR 시험 수)은 건드리지 않는다. 방법론 결정은 `docs/research/allocation/DECISIONS.md`(A1, A2, …로 번호를 매겨 추세 DECISIONS와 겹치지 않음), 사용법은 같은 폴더의 `README.md`.

- **`aa_assets.py`**: 국내 상장 상품 `PRODUCTS` 28종(종목코드 → 이름·자산군·미국 대용 ETF·환헤지 여부·총보수·상장일·레버리지 배수, 2026-10-24 확인. 섹터 4종·다우·물가채·회사채·리츠(H) 8종은 넓은 탐색 때 추가, 헬스케어·필수소비재의 대용 XLV·XLP는 2016년 이후 스냅샷에만 있어 오프라인 캐시에서는 '대용 데이터 없음')와 프리셋(`PRESETS`, 상품 코드 → %). 사용자는 ISA라 국내 상품만 살 수 있다(DECISIONS A13). 프리셋은 결과를 보고 고치지 않는다. 새 배분은 새 프리셋으로 등록. 프리셋은 비중 외에 신호 `rule`(`PresetRule`: overlay 종류·값·안전자산·주식만 여부, A38 ‘필터 판’)을 가질 수 있고, 화면·리밸런싱 계산기·성적표는 `aa_engine.preset_config`로 설정을 만든다(정적 프리셋은 예전과 똑같은 분기 설정). 프리셋을 불러오면 리밸런싱·overlay도 그 프리셋의 등록 설정으로 맞춘다. 정적 탐색(A31 전역 탐색의 프리셋 점, A37 닷컴 비교)과 위기 스위치 페이지는 규칙 있는 프리셋을 건너뛴다.
- **`aa_korea.py`**: 미국 대용 ETF 패널(`lab/data.py`) + 원/달러(`lab/aa_usdkrw.csv.gz`, FRED DEXKOUS) + 한국은행 기준금리(`lab/aa_bok_rate.csv`)로 상품별 원화 수익률을 만든 `krw_panel`(열 = 종목코드, 현금 = 원화 단기금리, `panel.benchmark` = 379800). 환노출·환헤지·원화·레버리지 공식은 모듈 docstring. 원/달러 갱신: `python -m lab.aa_korea refresh`.
- **`aa_engine.py`**: `AllocConfig`(비중·주기·밴드·비용·`Overlay`·`Leverage`) → `run_allocation(panel, cfg)` → `AllocResult`. 결정일 종가로 목표를 정하고 다음 거래일 종가에 체결, 편도 회전율에 비용, 현금은 패널 현금 수익률, 기준선은 `benchmark_of(panel)`. 레버리지는 일일 리셋(`levered_returns`). overlay의 안전자산(`Overlay.safe`)은 기본 비중에 들어 있어도 꺼지지 않고(비중 합 1 유지), 리스크 패리티에서 연 변동성 < `RISK_PARITY_MIN_VOL`(1%)인 현금성 상품은 기본 비중을 유지한다(둘 다 2026-10-24 수정, 테스트 있음). `MixConfig`/`run_mix`는 슬리브를 따로 돌려 고정 비율로 섞는다(믹스의 믹스는 없음). `current_targets`는 오늘 기준 목표 비중(리밸런싱 계산기용).
- **`aa_rebalance.py`**: ISA 리밸런싱 계산기(`rebalance_orders`: 입금액만으로 맞추기 / 전체 리밸런싱 + 밴드, 주 수는 버림). 보유 내역은 `lab_results/aa_holdings.json`(git 제외).
- **`aa_score.py`**: 아홉 축(단순·일관·위기·꼬리·비용·샤프 전반/후반·최대낙폭·칼마)을 비교 집합 안의 백분위 순위(0–100)로 점수화한다. 합성 '견고'는 **참고용**이고 판정은 Gate(기준선 = S&P500 원화. 탐색: 칼마 ≥ 기준선·|MDD| ≤ 0.75×기준선, 최종: CAGR ≥ 0.7×기준선·MDD −15% 이내·날짜 견고 `timing_mdd_worst` ≥ −20%, 자산배분 DECISIONS A26·A30)로만 한다. 날짜 견고가 없는 지표(계산 안 함)는 최종 Gate를 통과하지 않으니, 새 화면에서 Gate를 보이려면 `aa_validate.timing_worst_mdd`로 채운다. 위기 구간은 `CRISIS_WINDOWS`에 등록.
- **`aa_notes.py`**: `docs/research/allocation/NOTES.md`의 세 절(실무에 바로 쓰는 발견 / 정직한 경고 / 여기까지 오게 된 캠페인)을 읽고 덧붙인다. 다른 스레드·세션도 이 파일을 직접 고쳐 공유한다. 실행 기록은 `lab_results/aa_runs.csv`(git 제외, `config` 열에 설정 JSON → '전략 비교' 페이지가 다시 돌림).
- **`aa_store.py`**: `lab_results/<시각>_alloc_<이름>/` (meta.json + equity.csv + weights.csv). 추세 연구실 `store.py`와 폴더를 같이 쓰되 종류 접두어가 다르다.
- **`aa_app.py`**: 페이지 10개(전략 연구, 전략 비교, 성적표·믹스, 전술 전략·검증, 전역 탐색 결과, 실험 기록, 리밸런싱 계산기, 발견·경고·캠페인, 저장된 전략, 용어·상품 설명). 상품은 체크박스, 비중은 정수 % 슬라이더. `init`/`PERSIST_PREFIXES`(`r_`, `sc_`, `mix_`, `cmp_`, `rb_`, `m_`, `tac_`, `val_`, `cv_`, `sw_`, `gs_`) 규칙은 `app.py`와 같다. 선택 이벤트를 받는 표·차트의 키(`tbl_…`, `chart_…`)는 이 접두어로 시작하지 않게 한다(위젯 상태는 재할당할 수 없음). 버튼 키는 이 접두어로 시작하지 않게 하고(`btn_…`), `_editor`가 들어간 data_editor 키는 재할당에서 뺀다. 캐시만 있고 네트워크가 없으면 캐시 최신 날짜로 내려가 동작한다(EWY 캐시가 없으면 KODEX 200·레버리지는 비활성).
- **`aa_experiments.py`**: ‘실험 기록’ 페이지의 목록(`EXPERIMENTS`: 실험마다 왜 했나 · 핵심 결과 · 그래프 키 · 메모)과 그래프 데이터(커밋된 결과 파일만 읽음). 사용자 규칙(2026-10-26): **모든 실험은 왜 했는지, 핵심 결과, 그래프를 남긴다.** 실험을 하면 같은 PR에서 항목을 추가하고, 새 종류의 그래프는 `CHART_KEYS`와 `aa_app.EXPERIMENT_CHARTS`에 같이 넣는다(테스트가 둘을 맞춰 봄).
- **`aa_search.py`**: 전역 정적 탐색(DECISIONS A31)의 도구. `StaticSim`은 고정 비중 수천 개를 한 번에 돌리는 numpy 시뮬레이터로 `run_allocation`(결정일 종가 → 다음 거래일 종가 체결, 편도 비용, 사이 기간 drift, `rebalance_offset`)과 1e-9 안에서 같다(테스트). overlay·밴드·레버리지 factor는 없다. 표본 추출(`sample_two_level`: 자산군 Sobol × 상품 무작위, `sample_sparse`, `neighbours`), `summarize`(CAGR·MDD·창 낙폭·반기·보류), `rolling_window_share`, `pbo_selection`(월간 수익률 CSCV, 규칙 = 훈련 구간 MDD −15% 안에서 CAGR 최대)도 여기. 실행은 `scripts/aa_global_search.py`(`--quick`으로 1분 점검, `--replot`으로 산점도만 다시). 유일한 필수 조건은 전체 기간 MDD ≥ −15%이고 나머지는 열·⚠ 표시다(사용자 규칙). 엔진을 고치면 이 시뮬레이터의 프리셋 자체 점검이 먼저 깨진다.
- **`aa_search_results.py`**: A31 탐색 유니버스(30종: 국내 28 + 가상 환노출 미국채 2, `SYMBOLS`·`CLASSES`·`SYNTHETIC`)의 단일 정의와 결과 읽기. 스크립트와 ‘전역 탐색 결과’ 페이지가 같이 쓴다. 결과는 `docs/research/allocation/2026-10-25_global_search/`의 `passing.csv.gz`(MDD −15% 안 40,992개 전부, 비중은 `코드:%;…`)·`cloud.csv.gz`(밖 12,000개 표본) 등이고, `load_results`가 비중 행렬·자산군 비중·⚠를 붙인다. `band_frontier`는 frontier.md와 같은 값을 낸다(테스트). 탐색을 다시 돌리면 스크립트가 두 앱 파일도 쓴다. 2단계(A32, `scripts/aa_stage2.py` → `2026-10-25_stage2/stage2.csv`, % 단위)는 `load_stage2`·`stage2_scorecard`(MDD 시험만 세고 이웃 비율은 따로 보임)·`stage2_picks`(각 행이 여는 탐색 행 또는 프리셋)로 같은 페이지 ⑥에 나온다. 실제 가격 우선으로 다시 돌린 1·2단계(A35: `2026-10-26_global_search_real/`, `2026-10-26_stage2_real/`, 새 대표 후보 `2026-10-26_stage2_real_newreps/`)가 페이지 기본이고, `RESULT_SETS`(‘기록된 결과’ 선택)로 모형 기록(A31·A32)을 고른다. 결과 폴더를 새로 만들면 `RESULT_SETS`에 등록한다.
- **`aa_local.py`**: B2·B3·B4 구조 검증(자산배분 DECISIONS A37)의 도구. 중심 `CENTRES`, 구조 규칙 `rules`(R1 주식 35–50%·나스닥 ≥ 주식의 40%, R2 금 10–35%, R3 30년(H) ≤ 6%·리츠 없음), `sample_family`·`sample_control`, 특징 빼기 `ablate`(X1–X4), `perturb`(L1 이웃), `block_bootstrap_mdd`, 닷컴 시험용 `extend_history`(2000-09까지 대용 종가를 `SUBSTITUTES` 시리즈로 이어 붙임). 실행은 `scripts/aa_local_study.py`(`--dotcom`, `--quick`). 유니버스는 탐색 유니버스에서 레버리지·가상·KODEX 200을 뺀 21종. 결과에 따라 탐색 B2·B4·A 프리셋 설명에 ‘2005년 이후에만 확인됨’이 붙었다(비중은 그대로). 후속 A38(`scripts/aa_trend_filter_study.py`)은 같은 패널·표본으로 주식에만 10개월 추세 필터를 얹어 본다(엔진 `Overlay.only`: 필터를 걸 상품, 비우면 예전처럼 모든 위험자산).
- **런처**: `자산배분 연구실 시작.cmd`(8511), `자산배분 연구실 북마크 등록.cmd` → `scripts/aa_link.py`가 `aalab://` 프로토콜을 등록해 Chrome 북마크(`aalab://research|compare|scorecard|tactical|search|experiments|rebalance|notes|saved|glossary`)로 앱을 켠다. 추세 연구실의 `trendlab://`(8501)과 독립.
- 확인용 서버는 8512 등 다른 포트로 띄운다. 테스트(`tests/test_aa.py`, `tests/test_aa_korea.py`)는 streamlit/yaml을 import하지 않고 `lab_results/`에 쓰지 않는다(임시 경로 사용).
- 모든 결과는 국내 상장 ETF의 원화 수익률이고, 상장 전은 대용 데이터다(미국 ETF + 원/달러 + 두 나라 단기금리 + 총보수, 헤지 추가 비용 기본 20bp). 괴리율·추적오차·국내 금 프리미엄·세금은 모형에 없으니 결과를 말할 때 이 한계를 붙인다. 2026-10-24 이전 노트의 달러 숫자와 직접 비교하지 않는다(원화 기준선 MDD가 훨씬 작다).

## 확정된 연구 방법론 (상세: `docs/research/trend_v2/DECISIONS.md` 19번 이후, 목표: `CHARTER.md`)

- 2026-10-22부터 탐색의 주 기준은 **종목별 단독 시험**(`lab/single.py`, DECISIONS 67)이다: 종목마다 전략 vs 그 종목 보유, 그리고 매매 기록 포트폴리오 vs SPY(+국채 혼합). 포트폴리오 단위 결과는 비교용으로 함께 본다. 거래 단위 지표는 보조.
- 개별 종목 유니버스는 CRSP 시점별 시총 상위 1,000(DECISIONS 70)이다. 현재 지수 구성 종목만 쓰면 생존 편향이 커지므로 쓰지 않는다.
- 강건성 판정에 **가중 합산 점수를 쓰지 않는다.** 판정은 Gate(통과/탈락)와 구조적 기준(영역, 이웃, LOYO), 사전식 정렬로만 한다.
- Gate 기본값은 CHARTER에서 왔다: CAGR ≥ 0.80×SPY, |MDD| ≤ 0.75×|SPY MDD|, 완료 거래 ≥ 30. 통과 후보가 없어도 기준을 자동으로 낮추지 않는다.
  탐색 단계(계열·구성 요소 비교)는 탐색용 Gate `grid.EXPLORE_GATES`(칼마 ≥ SPY 칼마, 수익 조건 없음)를 쓰고, 보류 구간 후보는 위 최종 Gate로 고른다(DECISIONS 58). 최종 Gate는 결과를 본 뒤 바꾸지 않는다.
- 보류 구간(기본 2024-01-01 이후)은 연구에 쓰지 않는다. 앱의 종료일 상한이 보류 구간 직전으로 묶인다. 연구 결과를 보여줄 때 보류 구간 데이터를 섞지 말 것.
  보류 구간 평가는 **사용자가 그 후보를 지정해 요청할 때만** 한다. 평가할 때마다 `holdout.json`에 남고 볼수록 검증력이 줄어든다. 판정은 수익·낙폭 Gate만 쓴다(거래 수는 참고).
- 모든 백테스트는 기준선 3개(SPY 보유, SPY 200일선, 유니버스 동일비중)와 같이 본다. 단일 전략 강건성은 7개 통과/미달 체크리스트(DECISIONS 31).
- 남는 현금은 단기국채 수익률을 받는다(기본). 0%로 두면 추세추종이 부당하게 불리해진다.
- 2차원 격자(신호 파라미터 × 청산 파라미터)는 허용, 그 이상의 전수 탐색은 금지. 여러 변수를 함께 볼 때는 고정 예산 Sobol 표본 탐색(`lab/explore.py`)과 PBO로 한다(DECISIONS 66).
- 사용자 목표는 "레버리지 없이 수익 최대·낙폭 최소"다. 탐색 기준은 낙폭 15% 맞춤 수익이 SPY+국채 혼합보다 높은지(`lab/target.py`, DECISIONS 64)와 3년 롤링 구간 승률(65)이다.
- 고정 보유기간 청산과 전량 고정 목표가 청산은 쓰지 않는다(CHARTER). 단, 사용자 결정으로 조건부 시간 손절(`time_stop`: N일 뒤 진입가 이하일 때만)과 1회 부분 익절(`take_profit_pct`: +X%에 50% 매도)은 허용한다(DECISIONS 73).
- 블록은 가격·거래량과 최소한의 파생 지표로 시작한다. ETF 우선, 개별 종목은 yfinance 경로로(생존 편향 경고 유지).
- 기존 trend-energy score(0.65·TE63 + 0.35·TE126)와 score breakout은 연구 경로에서 폐기됐다. 되살리지 않는다.
- 결과는 표본 내(in-sample) 연구다. 스냅샷 유니버스가 현재 AUM 기준이라 생존 편향이 있다. 결과를 "승인"이나 "실거래 가능"으로 표현하지 않는다.
