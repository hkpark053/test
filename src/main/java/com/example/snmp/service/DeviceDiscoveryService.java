package com.example.snmp.service;

import com.example.snmp.dto.DeviceInfo;
import com.example.snmp.util.PortChecker;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * DHCP IP 리스트에서 장비명을 조회하는 서비스
 * 포트 체크와 SNMP 설정 확인을 통해 빠른 필터링 수행
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DeviceDiscoveryService {

    private final PortChecker portChecker;
    private final SnmpService snmpService;

    @Value("${discovery.port-check-enabled:true}")
    private boolean portCheckEnabled;

    @Value("${discovery.snmp-check-enabled:true}")
    private boolean snmpCheckEnabled;

    @Value("${discovery.max-threads:50}")
    private int maxThreads;

    private ExecutorService executorService;

    @javax.annotation.PostConstruct
    public void init() {
        executorService = Executors.newFixedThreadPool(maxThreads);
    }

    @javax.annotation.PreDestroy
    public void destroy() {
        if (executorService != null) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(60, java.util.concurrent.TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * DHCP IP 리스트에서 장비명 조회
     * 
     * @param ipAddresses IP 주소 리스트
     * @return 장비 정보 리스트
     */
    public List<DeviceInfo> discoverDevices(List<String> ipAddresses) {
        log.info("장비 조회 시작: {} 개 IP", ipAddresses.size());
        long startTime = System.currentTimeMillis();

        List<CompletableFuture<DeviceInfo>> futures = ipAddresses.stream()
            .map(ip -> CompletableFuture.supplyAsync(() -> discoverDevice(ip), executorService))
            .collect(Collectors.toList());

        List<DeviceInfo> results = futures.stream()
            .map(CompletableFuture::join)
            .filter(info -> info.getDeviceName() != null) // 장비명 조회 성공한 것만 필터링
            .collect(Collectors.toList());

        long elapsedTime = System.currentTimeMillis() - startTime;
        log.info("장비 조회 완료: {} 개 성공, {} ms 소요", results.size(), elapsedTime);

        return results;
    }

    /**
     * 단일 IP에 대한 장비 조회
     * 포트 체크 -> SNMP 설정 확인 -> 장비명 조회 순서로 진행
     */
    private DeviceInfo discoverDevice(String ipAddress) {
        DeviceInfo deviceInfo = new DeviceInfo();
        deviceInfo.setIpAddress(ipAddress);

        try {
            // 1단계: 포트 체크 (빠른 필터링)
            if (portCheckEnabled) {
                if (!portChecker.isSnmpPortOpen(ipAddress)) {
                    log.debug("포트 체크 실패, 스킵 [{}]", ipAddress);
                    return deviceInfo; // 장비명이 null인 상태로 반환
                }
            }

            // 2단계: SNMP 설정 확인 (더 정확한 필터링)
            if (snmpCheckEnabled) {
                if (!snmpService.isSnmpConfigured(ipAddress)) {
                    log.debug("SNMP 설정 확인 실패, 스킵 [{}]", ipAddress);
                    return deviceInfo; // 장비명이 null인 상태로 반환
                }
            }

            // 3단계: 장비명 조회
            String deviceName = snmpService.getDeviceName(ipAddress);
            deviceInfo.setDeviceName(deviceName);

            if (deviceName != null) {
                log.debug("장비 조회 성공 [{}]: {}", ipAddress, deviceName);
            }

        } catch (Exception e) {
            log.warn("장비 조회 중 예외 발생 [{}]: {}", ipAddress, e.getMessage());
        }

        return deviceInfo;
    }

    /**
     * 비동기로 장비 조회 (결과를 실시간으로 받을 수 있는 버전)
     */
    public CompletableFuture<List<DeviceInfo>> discoverDevicesAsync(List<String> ipAddresses) {
        return CompletableFuture.supplyAsync(() -> discoverDevices(ipAddresses), executorService);
    }
}
