package com.example.snmp.dto;

import lombok.Data;
import javax.validation.constraints.NotEmpty;
import java.util.List;

/**
 * 장비 조회 요청 DTO
 */
@Data
public class DiscoveryRequest {
    @NotEmpty(message = "IP 주소 리스트는 필수입니다")
    private List<String> ipAddresses;
}
