# 로컬 전략 워크플로 안내

## 시작 전 확인

기존 로컬 ResultStore를 지정해 사전 점검을 실행합니다. 이 명령은 원격 호출,
패키지 설치, 시장 데이터 다운로드를 하지 않습니다.

```powershell
python scripts/run_trend_v2_web.py preflight --store <local-result-store>
```

`[통과]`는 준비 완료, `[경고]`는 첫 정상 시작에서 상태 디렉터리가 만들어지는
등의 안내이며 시작을 막지 않습니다. `[차단]`은 지원 Python, 필수 패키지,
설정·저장소·스냅샷 호환성 또는 루프백 포트 문제이므로 해결 전에는 시작되지
않습니다. 출력에는 점검 코드, 한국어 안내, 기술 진단, 권장 조치가 포함되며
로컬 절대 경로나 비밀 정보는 표시하지 않습니다.

## 기본 시작과 종료

유일한 로컬 시작 명령은 다음입니다.

```powershell
python scripts/run_trend_v2_web.py start --store <local-result-store>
```

정상 시작은 사전 점검 경고 수, 저장소 준비, 복구한 워크플로/차단 항목 수와
정확한 URL을 표시합니다. 기본 URL은 `http://127.0.0.1:8765/`이며 다른 포트는
`--port`로 지정할 수 있습니다. 이 서버는 기본적으로 루프백에만 바인딩됩니다.

종료하려면 같은 터미널에서 `Ctrl+C`를 누르세요. 새 로컬 작업 접수를 중단하고,
활성 경제 작업에는 기존 협력 취소 계약으로 중단을 요청한 뒤 append-only 시도와
워크플로 상태를 보존합니다. 완료된 결과나 근거는 삭제하지 않습니다. 다음 시작은
수동 복구 없이 같은 저장소를 다시 열 수 있습니다.

## 재시작과 재개

시작 시 저장된 요청, 실행 시도, 강건성 시나리오, 워크플로를 다시 읽습니다.
프로세스 소유를 신뢰할 수 없는 실행 중 항목은 live로 보이지 않으며 `중단` 또는
`차단`으로 명시됩니다. 완료된 StrategyRun과 유효한 강건성 근거는 재실행하지
않고 재사용합니다. 손상되었거나 누락된 의존성은 재사용하거나 재개하지 않습니다.

워크플로 화면은 저장된 단계와 현재 서비스 상태를 분리해 보이고, 서비스 재시작
복구 ID, 차단/손상 수, 마지막 복구 결과를 표시합니다. `재개 가능`일 때만 기존
실행 요청 또는 강건성 시도의 재개 동작을 사용하세요. 완료 단위는 건너뛰고 중단된
단위만 다시 대기 상태가 됩니다. 확인 내용이 오래되었거나 다른 요청을 가리키면
재개가 거부됩니다. 재개 불가 사유는 화면과 API의 한국어 메시지로 확인하세요.

## 상태 확인과 보관

서버나 작업자를 추가로 시작하지 않고 상태만 보려면 다음을 사용합니다.

```powershell
python scripts/run_trend_v2_web.py status --store <local-result-store>
```

이 명령은 저장소 준비, 저장된 워크플로 수, 활성 시도, 재개 가능 항목과 마지막
복구 요약을 보여 줍니다. 작업 전에는 ResultStore 전체를 다른 안전한 위치에
복사해 백업하세요. 정리는 서비스를 종료한 뒤, 백업을 확인하고, 더 이상 필요 없는
로컬 ResultStore 전체만 제거하는 방식이 안전합니다. 개별 append-only 이벤트,
완료된 결과 또는 객체 파일을 손으로 삭제하지 마세요.

## 알려진 제한

이 도구는 인증, 클라우드/원격 저장소, 원격·분산 작업자, 프로세스 감독자, 시장
데이터 다운로드를 제공하지 않습니다. 중단된 작업은 자동 무제한 재시도하지 않으며
명시적 재개가 필요합니다. 역사적 유니버스의 생존 편향과 활성 OOS 수집 부재도
변하지 않았습니다.
# 첫 실행 (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py init --store .\.trend_v2_store
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py preflight --store .\.trend_v2_store
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py start --store .\.trend_v2_store
```

PowerShell 활성화는 필요 없습니다. 이후 실행에서는 같은 ResultStore를 지정해
`preflight`와 `start`만 실행합니다. `init`은 반복해도 안전하지만, 비어 있지 않은
비초기화 디렉터리, 호환되지 않는 정책, 손상된 정책은 덮어쓰지 않고 차단합니다.
# ResultStore 첫 실행과 평가 프로필

첫 실행에서는 아래 순서로 실행합니다. PowerShell 활성화는 필요 없습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py init --store .\.trend_v2_store
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py preflight --store .\.trend_v2_store
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py start --store .\.trend_v2_store
```

`init`은 기본 평가 프로필 세 개를 저장합니다: `연구용 기본 평가`, `최종 적격성 평가`,
`탐색용 가중 평가 예시`. 같은 저장소에서 `init`을 다시 실행하면 일치하는 프로필은
재사용하고, 빠진 기본 프로필만 추가합니다. 같은 ID/이름의 내용이 다른 프로필은
덮어쓰지 않고 차단합니다.

기존 저장소에 기본 프로필이 없을 때도 같은 명령을 다시 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py init --store .\.trend_v2_store
```

전략 구성 화면은 기본으로 `연구용 기본 평가`를 선택합니다. 프로필을 사용할 수 없으면
제출이 비활성화되고 다음 안내가 표시됩니다: `사용 가능한 평가 프로필이 없습니다. ResultStore 기본 프로필을 초기화하세요.`

이후 실행에서는 초기화 없이 다음 순서만 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py preflight --store .\.trend_v2_store
.\.venv\Scripts\python.exe scripts\run_trend_v2_web.py start --store .\.trend_v2_store
```
