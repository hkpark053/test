package com.example.snmp.service;

import com.example.snmp.util.SnmpChecker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * DHCP IP 리스트를 받아서 SNMP로 장비명을 조회하는 서비스
 * SNMP 설정이 안된 장비는 빠르게 스킵하여 성능 최적화
 */
@Service
public class DhcpSnmpService {
    
    private static final Logger logger = LoggerFactory.getLogger(DhcpSnmpService.class);
    
    @Autowired
    private SnmpChecker snmpChecker;
    
    // 병렬 처리를 위한 스레드 풀
    private final ExecutorService executorService;
    
    public DhcpSnmpService() {
        // CPU 코어 수에 맞춰 스레드 풀 크기 설정
        int threadPoolSize = Math.max(4, Runtime.getRuntime().availableProcessors() * 2);
        this.executorService = Executors.newFixedThreadPool(threadPoolSize);
    }
    
    /**
     * DHCP IP 리스트를 받아서 각 IP의 장비명을 조회
     * SNMP 설정이 안된 장비는 자동으로 스킵
     * 
     * @param ipList 조회할 IP 주소 리스트
     * @return IP 주소와 장비명의 맵 (조회 실패한 IP는 포함되지 않음)
     */
    public Map<String, String> getDeviceNames(List<String> ipList) {
        return getDeviceNames(ipList, "public");
    }
    
    /**
     * DHCP IP 리스트를 받아서 각 IP의 장비명을 조회 (커뮤니티 지정 가능)
     * 
     * @param ipList 조회할 IP 주소 리스트
     * @param community SNMP 커뮤니티 문자열
     * @return IP 주소와 장비명의 맵 (조회 실패한 IP는 포함되지 않음)
     */
    public Map<String, String> getDeviceNames(List<String> ipList, String community) {
        logger.info("총 {}개의 IP에 대해 장비명 조회 시작", ipList.size());
        
        long startTime = System.currentTimeMillis();
        
        // 병렬 처리로 SNMP 설정 확인 및 장비명 조회
        List<CompletableFuture<DeviceInfo>> futures = ipList.stream()
            .map(ip -> CompletableFuture.supplyAsync(() -> {
                try {
                    // 1단계: SNMP 설정 유무 빠르게 확인 (500ms 타임아웃)
                    if (!snmpChecker.isSnmpConfigured(ip, community)) {
                        logger.debug("SNMP 미설정으로 스킵: {}", ip);
                        return new DeviceInfo(ip, null, false);
                    }
                    
                    // 2단계: SNMP 설정이 확인된 경우에만 장비명 조회
                    String deviceName = snmpChecker.getDeviceName(ip, community);
                    return new DeviceInfo(ip, deviceName, true);
                    
                } catch (Exception e) {
                    logger.warn("IP {} 처리 중 오류: {}", ip, e.getMessage());
                    return new DeviceInfo(ip, null, false);
                }
            }, executorService))
            .collect(Collectors.toList());
        
        // 모든 작업 완료 대기
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
            futures.toArray(new CompletableFuture[0])
        );
        
        // 결과 수집
        Map<String, String> result = new HashMap<>();
        int configuredCount = 0;
        int successCount = 0;
        
        try {
            allFutures.join(); // 모든 작업 완료 대기
            
            for (CompletableFuture<DeviceInfo> future : futures) {
                DeviceInfo info = future.get();
                if (info.isConfigured) {
                    configuredCount++;
                    if (info.deviceName != null) {
                        result.put(info.ip, info.deviceName);
                        successCount++;
                    }
                }
            }
        } catch (Exception e) {
            logger.error("결과 수집 중 오류 발생", e);
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        logger.info("장비명 조회 완료 - 총: {}, SNMP 설정됨: {}, 성공: {}, 소요시간: {}ms", 
            ipList.size(), configuredCount, successCount, duration);
        
        return result;
    }
    
    /**
     * 장비 정보를 담는 내부 클래스
     */
    private static class DeviceInfo {
        String ip;
        String deviceName;
        boolean isConfigured;
        
        DeviceInfo(String ip, String deviceName, boolean isConfigured) {
            this.ip = ip;
            this.deviceName = deviceName;
            this.isConfigured = isConfigured;
        }
    }
    
    /**
     * 리소스 정리
     */
    public void shutdown() {
        executorService.shutdown();
    }
}
