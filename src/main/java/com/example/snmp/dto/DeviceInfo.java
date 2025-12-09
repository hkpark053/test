package com.example.snmp.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 장비 정보 DTO
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class DeviceInfo {
    private String ipAddress;
    private String deviceName;
}
