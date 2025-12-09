package com.example.snmp.service;

import lombok.extern.slf4j.Slf4j;
import org.snmp4j.CommunityTarget;
import org.snmp4j.PDU;
import org.snmp4j.Snmp;
import org.snmp4j.TransportMapping;
import org.snmp4j.event.ResponseEvent;
import org.snmp4j.mp.SnmpConstants;
import org.snmp4j.smi.*;
import org.snmp4j.transport.DefaultUdpTransportMapping;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * SNMP 조회 서비스
 * 장비명(system.sysName.0) 조회 기능 제공
 */
@Slf4j
@Service
public class SnmpService {

    @Value("${snmp.community:public}")
    private String community;

    @Value("${snmp.timeout:2000}")
    private int timeout;

    @Value("${snmp.retries:1}")
    private int retries;

    private Snmp snmp;
    private TransportMapping<UdpAddress> transport;

    @PostConstruct
    public void init() throws IOException {
        transport = new DefaultUdpTransportMapping();
        snmp = new Snmp(transport);
        transport.listen();
        log.info("SNMP 서비스 초기화 완료");
    }

    @PreDestroy
    public void destroy() {
        try {
            if (transport != null) {
                transport.close();
            }
            if (snmp != null) {
                snmp.close();
            }
        } catch (IOException e) {
            log.error("SNMP 서비스 종료 중 오류", e);
        }
    }

    /**
     * SNMP GET 요청으로 장비명 조회
     * 
     * @param ipAddress 대상 IP 주소
     * @return 장비명, 조회 실패시 null
     */
    public String getDeviceName(String ipAddress) {
        return getDeviceName(ipAddress, community);
    }

    /**
     * SNMP GET 요청으로 장비명 조회 (커뮤니티 지정)
     * 
     * @param ipAddress 대상 IP 주소
     * @param community SNMP 커뮤니티 스트링
     * @return 장비명, 조회 실패시 null
     */
    public String getDeviceName(String ipAddress, String community) {
        try {
            // SNMP GET 요청 생성
            PDU pdu = new PDU();
            pdu.setType(PDU.GET);
            // system.sysName.0 OID (1.3.6.1.2.1.1.5.0)
            pdu.add(new VariableBinding(new OID("1.3.6.1.2.1.1.5.0")));

            // 타겟 설정
            CommunityTarget target = new CommunityTarget();
            target.setCommunity(new OctetString(community));
            target.setAddress(new UdpAddress(ipAddress + "/161"));
            target.setVersion(SnmpConstants.version2c);
            target.setTimeout(timeout);
            target.setRetries(retries);

            // SNMP 요청 전송
            ResponseEvent response = snmp.send(pdu, target);
            PDU responsePDU = response.getResponse();

            if (responsePDU != null && responsePDU.getErrorStatus() == PDU.noError) {
                VariableBinding vb = responsePDU.get(0);
                if (vb != null && vb.getVariable() != null) {
                    String deviceName = vb.getVariable().toString();
                    log.debug("장비명 조회 성공 [{}]: {}", ipAddress, deviceName);
                    return deviceName;
                }
            } else {
                log.debug("SNMP 조회 실패 [{}]: {}", ipAddress, 
                    responsePDU != null ? responsePDU.getErrorStatusText() : "응답 없음");
            }
        } catch (Exception e) {
            log.debug("SNMP 조회 중 예외 발생 [{}]: {}", ipAddress, e.getMessage());
        }
        return null;
    }

    /**
     * SNMP가 설정되어 있는지 간단히 확인 (빠른 체크)
     * system.sysDescr.0 OID로 간단한 GET 요청을 보내서 응답이 오는지 확인
     * 
     * @param ipAddress 대상 IP 주소
     * @return SNMP 설정되어 있으면 true, 아니면 false
     */
    public boolean isSnmpConfigured(String ipAddress) {
        return isSnmpConfigured(ipAddress, community);
    }

    /**
     * SNMP가 설정되어 있는지 간단히 확인 (커뮤니티 지정)
     * 
     * @param ipAddress 대상 IP 주소
     * @param community SNMP 커뮤니티 스트링
     * @return SNMP 설정되어 있으면 true, 아니면 false
     */
    public boolean isSnmpConfigured(String ipAddress, String community) {
        try {
            // 간단한 GET 요청 (system.sysDescr.0)
            PDU pdu = new PDU();
            pdu.setType(PDU.GET);
            pdu.add(new VariableBinding(new OID("1.3.6.1.2.1.1.1.0")));

            CommunityTarget target = new CommunityTarget();
            target.setCommunity(new OctetString(community));
            target.setAddress(new UdpAddress(ipAddress + "/161"));
            target.setVersion(SnmpConstants.version2c);
            target.setTimeout(1000); // 빠른 체크를 위해 짧은 타임아웃
            target.setRetries(0); // 재시도 없음

            ResponseEvent response = snmp.send(pdu, target);
            PDU responsePDU = response.getResponse();

            if (responsePDU != null && responsePDU.getErrorStatus() == PDU.noError) {
                return true;
            }
        } catch (Exception e) {
            // 예외 발생시 SNMP 미설정으로 간주
        }
        return false;
    }

    /**
     * 비동기로 장비명 조회
     */
    public CompletableFuture<String> getDeviceNameAsync(String ipAddress) {
        return CompletableFuture.supplyAsync(() -> getDeviceName(ipAddress));
    }
}
