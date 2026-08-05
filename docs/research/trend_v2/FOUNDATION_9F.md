# Trend Strategy v2 Foundation 9F

## Windows에서 시작과 종료

저장소 루트의 `Trend Strategy V2 시작.cmd`를 두 번 클릭하면 저장소 로컬
`.venv\Scripts\python.exe`와 `.trend_v2_store`를 사용한다. ResultStore가 없는
첫 실행에는 기존 `run_trend_v2_web.py init`을 자동 실행한 뒤 결정적 preflight를
실행한다. 차단 항목이 없을 때만 기존 `run_trend_v2_web.py start` 서버를 숨겨진
프로세스로 시작하고, 정식 Trend Strategy v2 identity와 readiness를 확인한 뒤
Windows 기본 브라우저로 `http://127.0.0.1:8765/`를 연다.

이미 정식 서비스가 응답하면 그 서비스를 재사용한다. 같은 포트의 다른 서비스는
정식 health/identity 응답으로 오인하지 않으며 종료하지도 않는다. 시작 중의 두 번째
요청은 원래 요청의 bounded readiness를 기다린 후 같은 서비스를 연다. preflight가
차단되거나 시작이 실패하면 브라우저를 열지 않고 한국어 오류와 다음 조치를 표시한다.

정상 종료는 `Trend Strategy V2 종료.cmd`를 두 번 클릭한다. 종료 파일은 런처가
원자적으로 기록한 PID, 프로세스 생성 표식, 저장소/저장소-root 범위, 인스턴스 identity,
무작위 로컬 토큰을 모두 확인한 뒤 loopback 제어 endpoint로 graceful shutdown을
요청한다. 그러면 기존 서버의 intake 중지, 활성 실행-attempt 취소 요청,
`ControlledExecutionService.close()`, append-only 상태 보존 경로가 그대로 실행된다.
Python 전체, 포트 점유 프로세스, 또는 검증되지 않은 PID를 강제 종료하지 않는다.

## 운영 파일과 복구

로그는 항상 `.trend_v2_store\launcher\launcher.log`에 기록된다. 런처 소유권 상태는
같은 디렉터리의 `runtime.json`, 시작 직렬화는 `start.lock`, 종료 비밀은
`shutdown.token`에 저장된다. 이 파일들은 경제 근거가 아닌 로컬 운영 상태이며 Git에
포함되지 않는다. runtime 기록이 오래되었고 PID가 없거나 생성 표식이 달라진 경우에만
해당 운영 기록을 정리한다. 프로세스 부재를 경제 작업 완료로 해석하지 않는다.
검증된 프로세스가 응답 identity와 일치하지 않거나 정상 종료가 시간 안에 끝나지 않으면
상태를 보존하고 아무 프로세스도 강제 종료하지 않는다.

`.venv`가 없거나 사용할 수 없으면 네트워크 설치를 시도하지 않고 예상 경로와 로그
경로를 표시한다. 고급 복구와 테스트에만 `TREND_V2_PYTHON`, `TREND_V2_STORE`,
`TREND_V2_PORT`를 사용할 수 있으며 정상 사용에는 설정이 필요 없다.

## 바탕 화면 바로가기

각 `.cmd` 파일을 마우스 오른쪽 단추로 누른 뒤 **더 많은 옵션 표시 → 보내기 →
바탕 화면에 바로 가기 만들기**를 선택한다. 생성된 바로가기는 저장소를 이동하면 다시
만들어야 한다. `.cmd`는 `%~dp0`에서 저장소 루트를 계산하므로 저장소 경로에 공백,
한글, 괄호가 있어도 동작한다.

## 제한

Windows 로컬 단일 사용자·단일 호스트 운용만 지원한다. 서버는 loopback에만 bind하며
원격 접속, CORS 확대, 인증 우회, 원격 worker, cloud storage를 제공하지 않는다.
강제 종료나 운영체제 장애 중 실행되던 경제 작업은 기존 Foundation 9B 복구 규칙에 따라
다음 시작에서 stale/interrupted/blocked로 명시되며 자동 완료 처리되지 않는다.

다음 작업은 `Foundation 10 -- separately scoped product workflow evolution, without unrestricted optimization`이다.
