package com.example.snmp.controller;

import com.example.snmp.dto.DeviceInfo;
import com.example.snmp.dto.DiscoveryRequest;
import com.example.snmp.service.DeviceDiscoveryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.List;

/**
 * 장비 조회 REST API 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/api/devices")
@RequiredArgsConstructor
@Validated
public class DeviceDiscoveryController {

    private final DeviceDiscoveryService deviceDiscoveryService;

    /**
     * DHCP IP 리스트에서 장비명 조회
     * 
     * POST /api/devices/discover
     * Body: { "ipAddresses": ["192.168.1.1", "192.168.1.2", ...] }
     */
    @PostMapping("/discover")
    public ResponseEntity<List<DeviceInfo>> discoverDevices(@Valid @RequestBody DiscoveryRequest request) {
        log.info("장비 조회 요청 수신: {} 개 IP", request.getIpAddresses().size());
        
        List<DeviceInfo> results = deviceDiscoveryService.discoverDevices(request.getIpAddresses());
        
        return ResponseEntity.ok(results);
    }

    /**
     * 단일 IP에 대한 장비명 조회 (테스트용)
     * 
     * GET /api/devices/{ipAddress}
     */
    @GetMapping("/{ipAddress}")
    public ResponseEntity<DeviceInfo> getDeviceInfo(@PathVariable String ipAddress) {
        List<String> ipList = List.of(ipAddress);
        List<DeviceInfo> results = deviceDiscoveryService.discoverDevices(ipList);
        
        if (results.isEmpty() || results.get(0).getDeviceName() == null) {
            return ResponseEntity.notFound().build();
        }
        
        return ResponseEntity.ok(results.get(0));
    }
}
