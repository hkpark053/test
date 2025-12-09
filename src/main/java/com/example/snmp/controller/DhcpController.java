package com.example.snmp.controller;

import com.example.snmp.service.DhcpSnmpService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DHCP IP 리스트를 받아서 SNMP로 장비명을 조회하는 REST API 컨트롤러
 */
@RestController
@RequestMapping("/api/dhcp")
public class DhcpController {
    
    private static final Logger logger = LoggerFactory.getLogger(DhcpController.class);
    
    @Autowired
    private DhcpSnmpService dhcpSnmpService;
    
    /**
     * IP 리스트를 받아서 각 IP의 장비명을 조회
     * 
     * POST /api/dhcp/devices
     * Body: { "ipList": ["192.168.1.1", "192.168.1.2", ...], "community": "public" (optional) }
     * 
     * @param request IP 리스트와 SNMP 커뮤니티를 담은 요청 객체
     * @return IP 주소와 장비명의 맵
     */
    @PostMapping("/devices")
    public ResponseEntity<Map<String, Object>> getDeviceNames(@RequestBody DeviceRequest request) {
        try {
            if (request.getIpList() == null || request.getIpList().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(createErrorResponse("IP 리스트가 비어있습니다."));
            }
            
            String community = request.getCommunity() != null ? request.getCommunity() : "public";
            
            logger.info("{}개의 IP에 대해 장비명 조회 요청 (커뮤니티: {})", 
                request.getIpList().size(), community);
            
            Map<String, String> deviceMap = dhcpSnmpService.getDeviceNames(
                request.getIpList(), 
                community
            );
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("devices", deviceMap);
            response.put("total", request.getIpList().size());
            response.put("found", deviceMap.size());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            logger.error("장비명 조회 중 오류 발생", e);
            return ResponseEntity.internalServerError()
                .body(createErrorResponse("장비명 조회 중 오류가 발생했습니다: " + e.getMessage()));
        }
    }
    
    /**
     * 단일 IP의 장비명 조회 (테스트용)
     * 
     * GET /api/dhcp/device?ip=192.168.1.1&community=public
     */
    @GetMapping("/device")
    public ResponseEntity<Map<String, Object>> getDeviceName(
            @RequestParam String ip,
            @RequestParam(required = false, defaultValue = "public") String community) {
        try {
            Map<String, String> deviceMap = dhcpSnmpService.getDeviceNames(
                List.of(ip), 
                community
            );
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("ip", ip);
            response.put("deviceName", deviceMap.getOrDefault(ip, null));
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            logger.error("장비명 조회 중 오류 발생", e);
            return ResponseEntity.internalServerError()
                .body(createErrorResponse("장비명 조회 중 오류가 발생했습니다: " + e.getMessage()));
        }
    }
    
    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> response = new HashMap<>();
        response.put("success", false);
        response.put("error", message);
        return response;
    }
    
    /**
     * 요청 DTO 클래스
     */
    public static class DeviceRequest {
        private List<String> ipList;
        private String community;
        
        public List<String> getIpList() {
            return ipList;
        }
        
        public void setIpList(List<String> ipList) {
            this.ipList = ipList;
        }
        
        public String getCommunity() {
            return community;
        }
        
        public void setCommunity(String community) {
            this.community = community;
        }
    }
}
