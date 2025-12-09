# SNMP Device Discovery

DHCP IP 리스트에서 SNMP를 통해 장비명을 조회하는 Spring Boot 애플리케이션입니다.

## 주요 기능

1. **빠른 필터링**: SNMP 설정이 안된 장비를 빠르게 스킵
   - 포트 체크 (161 포트): 빠른 1차 필터링
   - SNMP 설정 확인: 간단한 GET 요청으로 2차 필터링
   - 장비명 조회: 필터링 통과한 장비만 상세 조회

2. **병렬 처리**: 여러 IP를 동시에 조회하여 성능 최적화

3. **REST API**: HTTP API를 통한 장비 조회

## 빌드 및 실행

### Maven 사용
```bash
mvn clean package
java -jar target/snmp-device-discovery-1.0.0.jar
```

### 실행 옵션
```bash
# 커뮤니티 스트링 변경
java -jar target/snmp-device-discovery-1.0.0.jar --snmp.community=your-community

# 포트 체크 비활성화 (SNMP 체크만 사용)
java -jar target/snmp-device-discovery-1.0.0.jar --discovery.port-check-enabled=false

# 스레드 수 조정
java -jar target/snmp-device-discovery-1.0.0.jar --discovery.max-threads=100
```

## API 사용법

### 장비 조회 (POST)
```bash
curl -X POST http://localhost:8080/api/devices/discover \
  -H "Content-Type: application/json" \
  -d '{
    "ipAddresses": [
      "192.168.1.1",
      "192.168.1.2",
      "192.168.1.3"
    ]
  }'
```

### 응답 예시
```json
[
  {
    "ipAddress": "192.168.1.1",
    "deviceName": "Router-01"
  },
  {
    "ipAddress": "192.168.1.2",
    "deviceName": "Switch-02"
  }
]
```

### 단일 IP 조회 (GET)
```bash
curl http://localhost:8080/api/devices/192.168.1.1
```

## 설정 설명

### 포트 체크 vs SNMP 설정 확인

**포트 체크 (port-check-enabled)**:
- 장점: 매우 빠름 (약 1초)
- 단점: 포트가 열려있어도 SNMP가 설정되지 않았을 수 있음

**SNMP 설정 확인 (snmp-check-enabled)**:
- 장점: 실제 SNMP 응답을 확인하므로 더 정확함
- 단점: 포트 체크보다 약간 느림 (약 1-2초)

**권장 설정**:
- 둘 다 활성화: 포트 체크로 빠르게 필터링 → SNMP 체크로 정확도 향상
- 포트 체크만: 매우 빠르지만 일부 오탐 가능
- SNMP 체크만: 정확하지만 상대적으로 느림

## 성능 최적화 팁

1. **스레드 수 조정**: `discovery.max-threads` 값을 네트워크 환경에 맞게 조정
2. **타임아웃 조정**: `snmp.timeout` 값을 줄이면 빠르지만 일부 장비를 놓칠 수 있음
3. **포트 체크 우선**: 포트 체크를 먼저 수행하여 SNMP 조회 대상을 줄임

## 주의사항

- SNMP 커뮤니티 스트링이 장비마다 다를 수 있습니다
- 방화벽 규칙에 따라 포트 체크가 실패할 수 있습니다
- 대량의 IP 조회시 네트워크 부하가 발생할 수 있습니다
