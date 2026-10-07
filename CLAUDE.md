# CLAUDE.md — 프로젝트 안내 (Claude Code가 매 세션 자동으로 읽음)

## 사용자

- 연구자이며 Git/웹 개발 전문가는 아님. 변경 설명은 **무엇을 / 왜 / 어디에 영향** 순서로, 한국어로.
- 코드를 요청하면 부분 snippet보다 그대로 쓸 수 있는 완성 코드를 선호.

## 지금 주력 제품: `lab/` (추세추종 전략 연구실)

Streamlit 앱 하나로 **전략 정의 → 백테스트 → 강건성 격자 → 저장/비교**를 한다.

```
lab/
  app.py      화면 (Streamlit). 실행: 저장소 루트의 "추세추종 연구실 시작.cmd"
  data.py     가격 데이터: ETF 스냅샷(오프라인) + yfinance 다운로드(캐시: lab_data/)
  blocks.py   전략 블록 (진입 AND / 청산 OR, 블록당 파라미터 1개, 가격·거래량만)
  engine.py   포트폴리오 백테스트 (t일 종가 신호 → t+1일 시가 체결)
  metrics.py  지표 계산 + 정의(DEFINITIONS: 화면 툴팁·용어 설명의 단일 출처)
  grid.py     강건성: Gate → 이웃 생존율 → 연결 영역 → LOYO → 사전식 정렬
  store.py    결과 저장 (lab_results/, git 제외)
tests/test_lab.py
```

새 진입/청산 블록 추가: `lab/blocks.py`의 `ENTRY_BLOCKS`/`EXIT_BLOCKS`에 `Block` 하나를 추가하면
화면·격자·용어 설명에 자동으로 나타난다. 포지션 상태가 필요한 청산(`compute=None`)은 `engine.py`에서 처리.

## 확정된 결정 (2026-10-08, 상세: docs/research/trend_v2/DECISIONS.md 19–25)

1. 기존 v2 Foundation UI(`src/trend_v2_foundation/`)를 고치지 않고, `lab/`을 새 단일 화면으로 만든다 (계획 A).
2. **포트폴리오 단위 지표(SPY 대비 CAGR·MDD 등)가 주 기준**, 거래 단위 지표(t-통계량, 손익비 등)는 통계적 우위 확인용 **보조**로 함께 표시.
3. 강건성 분석용 **2차원 파라미터 격자 허용** (신호 파라미터 × 청산 파라미터). 축당 최대 12개 값.
4. 전략 블록은 **가격·거래량과 최소한의 파생 지표**(이동평균, 이전 N일 고저, 평균 거래량)로 시작하고, 이후 점진적으로 세분화.
5. ETF 우선, 이후 개별 종목으로 확장 (yfinance 경로는 이미 동작, 생존 편향 경고 표시).
6. 강건성 판정에 **가중 합산 점수를 쓰지 않는다.** Gate(통과/탈락) + 이웃 생존율 + 연결 영역 + LOYO + 사전식 정렬.
7. Gate 기본값은 CHARTER.md 목표에서 가져온다 (CAGR ≥ 0.8×SPY, |MDD| ≤ 0.75×|SPY MDD|, 거래 ≥ 30). 결과를 보고 기준을 낮추지 않는다.

## 작업 규칙

- **화면 변경 후에는 앱을 실제로 띄워 눌러 본 뒤** 완료라고 말한다. (`.claude/launch.json`의 `lab`, 또는 `.venv\Scripts\python.exe -m streamlit run lab/app.py`)
- 사용자 화면에 내부 구조 용어(StrategyRun, hash, attempt 등)를 노출하지 않는다. 모든 지표/옵션에는 정의(툴팁)를 붙인다.
- 지표를 추가하면 `metrics.DEFINITIONS`에 정의·계산식을 같이 넣는다 (정의와 계산의 출처를 하나로).
- Streamlit markdown에서 `~` 두 개는 취소선이 된다. 범위 표기는 `–`를 쓴다.
- 테스트: `.venv\Scripts\python.exe -m pytest -q tests/test_lab.py` (lab 변경 시). 전체 스위트는 공용 코드를 건드릴 때만.
- 변경은 브랜치 → PR. `main` 직접 커밋 금지.

## 건드리지 않는 것

- Daily ETF Scan (`src/run_daily_scan.py`, `web/`, `docs/` GitHub Pages, `.github/workflows/daily_scan.yml`).
- v1 Backtest Only (`src/backtest.py`, `.github/workflows/backtest-only.yml`): 동결. 워크플로 파일은 OOS 기준선
  매니페스트(PR #18)에 해시로 고정돼 있어 수정하면 기준선 테스트가 깨진다.
- `docs/data/` 생성 데이터, `docs/research/trend_v2/phase_a2/` 고정 스냅샷 (lab이 읽기 전용으로 사용).
- `src/trend_v2_foundation/` (v2 Foundation): 보존만 한다. Codex용 규칙은 `AGENTS.md`.

## 알려진 한계

- ETF 스냅샷 유니버스는 현재 시점 AUM 기준 → 생존 편향. 개별 종목은 더 크다.
- 결과는 표본 내(in-sample) 연구이며 실거래 승인이 아니다.
