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

## 명령어 (Windows, 저장소 루트, `.venv`는 Python 3.13)

```bash
# 연구실 앱 실행 (사용자는 "추세추종 연구실 시작.cmd" 더블클릭)
.venv/Scripts/python.exe -m streamlit run lab/app.py        # http://127.0.0.1:8501

# 테스트 — 반드시 `python -m pytest` (루트를 sys.path에 넣어 `src`, `lab` import가 됨. pytest 설정 파일 없음)
.venv/Scripts/python.exe -m pytest -q tests/test_lab.py                       # lab 변경 시
.venv/Scripts/python.exe -m pytest -q tests/test_lab.py::test_gates_and_loyo  # 단일 테스트
.venv/Scripts/python.exe -m pytest -q -W error::FutureWarning                 # 전체 (~1.5분, CI와 동일 플래그)

# 레거시 파이프라인 (스캔은 GitHub Actions가 매일 실행, 백테스트는 Actions 탭에서 수동 실행만; 로컬 실행은 docs/data를 덮어쓰므로 하지 말 것)
python -m src.run_daily_scan && python src/postprocess_groups.py
python -m src.run_backtest_only
```

CI(`.github/workflows/tests.yml`)는 PR마다 **Python 3.11**, pandas 2.2와 3.x 두 버전으로 `-W error::FutureWarning` 전체 스위트를 돈다. 로컬 venv는 3.13이므로 3.12+ 전용 문법과 pandas FutureWarning을 피할 것.
린터/포매터 설정은 없다.

## 저장소 구조: 다섯 개의 하위 시스템

| 시스템 | 위치 | 상태 |
|---|---|---|
| **연구실 (lab)** | `lab/`, `tests/test_lab.py` | **현재 주력 제품.** 새 기능은 여기에 |
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
- **`batch.py`**: 앱 기본 유니버스·설정으로 공통 조건 시험을 명령줄에서 돌린다(`python -m lab.batch addon <kind> <key>`). 앱과 똑같이 시험 횟수에 기록하고 결과를 저장한다.
- **`daily_alert.py`**: 일일 알림(DECISIONS 53). `config/alert.yml` 전략의 모델 포트폴리오를 오늘까지 시뮬레이션해 다음 시가 주문(`BacktestResult.pending_orders`)과 국면 상태를 이슈 댓글로 보낸다. 성과는 보여주지 않는다.
- **`refine.py`**: 확인 조건 시험. 한 계열에 진입 블록을 하나씩(기본값으로만) 더해 같은 격자를 다시 돌리고, 네 기준의 파레토 개선 여부로 판정한다(DECISIONS 35–36).
- 연구 기록: `docs/research/lab/` (날짜별 메모). 연구를 대신 수행했다면 결과·규칙·한계를 여기에 남긴다.
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
  - 포지션 크기 `sizing`: `equal` | `inverse_vol`(기본, 60일 변동성·2배 상한). `to_dict()`는 기본값 `equal`일 때 키를 생략해 예전 시험 키를 유지한다. 새 설정 필드를 더할 때도 같은 방식으로.
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
- **`app.py`**: Streamlit, `st.navigation`으로 페이지 4개(전략 연구, 최종 검증, 저장된 결과, 용어 설명).
  - 위젯 기본값은 `init(key, default)`로 session_state에 한 번만 넣는다(`value=`와 key를 같이 쓰면 경고가 난다).
  - 페이지를 옮겨도 설정이 유지되도록 `PERSIST_PREFIXES`의 키를 매 실행 재할당한다. 새 위젯 키를 만들면 여기에 접두어를 추가할 것.

### 앱 작업 시 주의

- 화면을 바꾼 뒤에는 앱을 실제로 띄워 눌러 본 다음 완료라고 말한다. 서버를 띄운 채 `lab/*.py`(app.py 제외)를 고치면 이전 모듈이 남아 있으니 서버를 재시작할 것.
  사용자가 런처로 8501을 열어 두었을 수 있으니 확인용 서버는 8502 등 다른 포트로 띄운다. 화면 시험으로 생긴 `lab_results/` 항목은 백업해 두었다가 되돌린다.
- Streamlit markdown에서 `~` 두 개는 취소선이 된다. 범위는 `–`로 쓴다.
- 폭 지정은 `width="stretch"`를 쓴다(`use_container_width`는 지원 종료 예정).
- `.streamlit/config.toml`: `magicEnabled = false`(단독 표현식이 화면에 출력되는 것 방지), `address = 127.0.0.1`(외부 노출 방지).
- `.cmd` 런처의 echo 문구는 ASCII로만 쓴다. 한글 echo는 cmd 인코딩 문제로 깨진다(PR #44).
- 사용자 화면에 내부 구조 용어(StrategyRun, hash 등)를 노출하지 않는다. 모든 지표·옵션에 정의를 붙인다.

## 확정된 연구 방법론 (상세: `docs/research/trend_v2/DECISIONS.md` 19번 이후, 목표: `CHARTER.md`)

- **포트폴리오 단위 지표(같은 날짜의 SPY 대비)가 주 기준**이다. 거래 단위 지표(t-통계량, 손익비 등)는 거래당 통계적 우위를 보는 **보조**.
- 강건성 판정에 **가중 합산 점수를 쓰지 않는다.** 판정은 Gate(통과/탈락)와 구조적 기준(영역, 이웃, LOYO), 사전식 정렬로만 한다.
- Gate 기본값은 CHARTER에서 왔다: CAGR ≥ 0.80×SPY, |MDD| ≤ 0.75×|SPY MDD|, 완료 거래 ≥ 30. 통과 후보가 없어도 기준을 자동으로 낮추지 않는다.
  탐색 단계(계열·구성 요소 비교)는 탐색용 Gate `grid.EXPLORE_GATES`(칼마 ≥ SPY 칼마, 수익 조건 없음)를 쓰고, 보류 구간 후보는 위 최종 Gate로 고른다(DECISIONS 58). 최종 Gate는 결과를 본 뒤 바꾸지 않는다.
- 보류 구간(기본 2024-01-01 이후)은 연구에 쓰지 않는다. 앱의 종료일 상한이 보류 구간 직전으로 묶인다. 연구 결과를 보여줄 때 보류 구간 데이터를 섞지 말 것.
  보류 구간 평가는 **사용자가 그 후보를 지정해 요청할 때만** 한다. 평가할 때마다 `holdout.json`에 남고 볼수록 검증력이 줄어든다. 판정은 수익·낙폭 Gate만 쓴다(거래 수는 참고).
- 모든 백테스트는 기준선 3개(SPY 보유, SPY 200일선, 유니버스 동일비중)와 같이 본다. 단일 전략 강건성은 7개 통과/미달 체크리스트(DECISIONS 31).
- 남는 현금은 단기국채 수익률을 받는다(기본). 0%로 두면 추세추종이 부당하게 불리해진다.
- 2차원 격자(신호 파라미터 × 청산 파라미터)는 허용, 그 이상의 전수 탐색은 금지.
- 고정 보유기간 청산과 고정 목표가 청산은 쓰지 않는다(CHARTER).
- 블록은 가격·거래량과 최소한의 파생 지표로 시작한다. ETF 우선, 개별 종목은 yfinance 경로로(생존 편향 경고 유지).
- 기존 trend-energy score(0.65·TE63 + 0.35·TE126)와 score breakout은 연구 경로에서 폐기됐다. 되살리지 않는다.
- 결과는 표본 내(in-sample) 연구다. 스냅샷 유니버스가 현재 AUM 기준이라 생존 편향이 있다. 결과를 "승인"이나 "실거래 가능"으로 표현하지 않는다.
