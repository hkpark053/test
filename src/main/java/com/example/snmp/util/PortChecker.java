package com.example.snmp.util;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * 포트 연결 가능 여부를 빠르게 확인하는 유틸리티
 * SNMP 포트(161)가 열려있는지 확인하여 빠른 필터링 수행
 */
@Slf4j
@Component
public class PortChecker {

    private static final int DEFAULT_TIMEOUT_MS = 1000; // 1초 타임아웃
    private static final int SNMP_PORT = 161;

    /**
     * SNMP 포트(161)가 열려있는지 확인
     * 
     * @param ipAddress 확인할 IP 주소
     * @param timeoutMs 타임아웃 시간 (밀리초)
     * @return 포트가 열려있으면 true, 아니면 false
     */
    public boolean isSnmpPortOpen(String ipAddress, int timeoutMs) {
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(ipAddress, SNMP_PORT), timeoutMs);
            return true;
        } catch (IOException e) {
            log.debug("SNMP 포트 체크 실패 [{}:{}]: {}", ipAddress, SNMP_PORT, e.getMessage());
            return false;
        }
    }

    /**
     * SNMP 포트(161)가 열려있는지 확인 (기본 타임아웃 사용)
     */
    public boolean isSnmpPortOpen(String ipAddress) {
        return isSnmpPortOpen(ipAddress, DEFAULT_TIMEOUT_MS);
    }

    /**
     * 비동기로 SNMP 포트가 열려있는지 확인
     */
    public CompletableFuture<Boolean> isSnmpPortOpenAsync(String ipAddress, int timeoutMs) {
        return CompletableFuture.supplyAsync(() -> isSnmpPortOpen(ipAddress, timeoutMs));
    }

    /**
     * 비동기로 SNMP 포트가 열려있는지 확인 (기본 타임아웃 사용)
     */
    public CompletableFuture<Boolean> isSnmpPortOpenAsync(String ipAddress) {
        return isSnmpPortOpenAsync(ipAddress, DEFAULT_TIMEOUT_MS);
    }
}
