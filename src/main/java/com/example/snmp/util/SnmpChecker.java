package com.example.snmp.util;

import org.snmp4j.*;
import org.snmp4j.mp.SnmpConstants;
import org.snmp4j.smi.*;
import org.snmp4j.transport.DefaultUdpTransportMapping;
import org.snmp4j.util.DefaultPDUFactory;
import org.snmp4j.util.TreeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.InetAddress;
import java.util.concurrent.TimeUnit;

/**
 * SNMP 설정 유무를 빠르게 확인하고, SNMP Walk를 수행하는 유틸리티 클래스
 * 
 * 핵심 기능:
 * 1. SNMP 설정 유무 확인 (타임아웃 500ms로 빠른 스킵)
 * 2. SNMP Walk로 장비명 조회
 */
@Component
public class SnmpChecker {
    
    private static final Logger logger = LoggerFactory.getLogger(SnmpChecker.class);
    
    // SNMP 기본 설정
    private static final int SNMP_PORT = 161;
    private static final int CHECK_TIMEOUT_MS = 500; // SNMP 설정 확인용 짧은 타임아웃
    private static final int WALK_TIMEOUT_MS = 3000; // SNMP Walk용 타임아웃
    private static final String DEFAULT_COMMUNITY = "public";
    
    // 장비명을 조회할 OID (sysName)
    private static final String SYS_NAME_OID = "1.3.6.1.2.1.1.5.0";
    
    /**
     * SNMP 설정이 되어있는지 빠르게 확인
     * UDP 소켓으로 161 포트에 간단한 SNMP GET 요청을 보내서 응답 여부를 확인
     * 
     * @param ipAddress 확인할 IP 주소
     * @return SNMP 설정이 되어있으면 true, 아니면 false
     */
    public boolean isSnmpConfigured(String ipAddress) {
        return isSnmpConfigured(ipAddress, DEFAULT_COMMUNITY);
    }
    
    /**
     * SNMP 설정이 되어있는지 빠르게 확인 (커뮤니티 지정 가능)
     * 
     * @param ipAddress 확인할 IP 주소
     * @param community SNMP 커뮤니티 문자열
     * @return SNMP 설정이 되어있으면 true, 아니면 false
     */
    public boolean isSnmpConfigured(String ipAddress, String community) {
        Snmp snmp = null;
        try {
            // UDP 전송 매핑 생성
            TransportMapping<?> transport = new DefaultUdpTransportMapping();
            snmp = new Snmp(transport);
            transport.listen();
            
            // 타겟 주소 설정
            Address targetAddress = new UdpAddress(InetAddress.getByName(ipAddress), SNMP_PORT);
            CommunityTarget target = new CommunityTarget();
            target.setCommunity(new OctetString(community));
            target.setAddress(targetAddress);
            target.setRetries(0); // 재시도 없음 (빠른 실패)
            target.setTimeout(CHECK_TIMEOUT_MS); // 짧은 타임아웃
            target.setVersion(SnmpConstants.version2c);
            
            // 간단한 GET 요청 (sysDescr OID 사용)
            PDU pdu = new PDU();
            pdu.setType(PDU.GET);
            pdu.add(new VariableBinding(new OID("1.3.6.1.2.1.1.1.0"))); // sysDescr
            
            // 비동기 요청 전송
            ResponseEvent response = snmp.send(pdu, target);
            
            if (response != null && response.getResponse() != null) {
                PDU responsePDU = response.getResponse();
                if (responsePDU.getErrorStatus() == PDU.noError) {
                    logger.debug("SNMP 설정 확인 성공: {}", ipAddress);
                    return true;
                }
            }
            
            return false;
            
        } catch (Exception e) {
            logger.debug("SNMP 설정 확인 실패 (예상 가능): {} - {}", ipAddress, e.getMessage());
            return false;
        } finally {
            if (snmp != null) {
                try {
                    snmp.close();
                } catch (IOException e) {
                    logger.warn("SNMP 연결 종료 중 오류: {}", e.getMessage());
                }
            }
        }
    }
    
    /**
     * SNMP Walk를 사용하여 장비명 조회
     * 
     * @param ipAddress 조회할 IP 주소
     * @return 장비명, 조회 실패시 null
     */
    public String getDeviceName(String ipAddress) {
        return getDeviceName(ipAddress, DEFAULT_COMMUNITY);
    }
    
    /**
     * SNMP Walk를 사용하여 장비명 조회 (커뮤니티 지정 가능)
     * 
     * @param ipAddress 조회할 IP 주소
     * @param community SNMP 커뮤니티 문자열
     * @return 장비명, 조회 실패시 null
     */
    public String getDeviceName(String ipAddress, String community) {
        Snmp snmp = null;
        try {
            // UDP 전송 매핑 생성
            TransportMapping<?> transport = new DefaultUdpTransportMapping();
            snmp = new Snmp(transport);
            transport.listen();
            
            // 타겟 주소 설정
            Address targetAddress = new UdpAddress(InetAddress.getByName(ipAddress), SNMP_PORT);
            CommunityTarget target = new CommunityTarget();
            target.setCommunity(new OctetString(community));
            target.setAddress(targetAddress);
            target.setRetries(1);
            target.setTimeout(WALK_TIMEOUT_MS);
            target.setVersion(SnmpConstants.version2c);
            
            // TreeUtils를 사용하여 SNMP Walk 수행
            TreeUtils treeUtils = new TreeUtils(snmp, new DefaultPDUFactory());
            
            // sysName OID로 장비명 조회
            OID[] oids = new OID[] { new OID(SYS_NAME_OID) };
            
            List<? extends VariableBinding> result = treeUtils.getSubtree(target, oids[0]);
            
            if (result != null && !result.isEmpty()) {
                VariableBinding vb = result.get(0);
                if (vb != null && vb.getVariable() != null) {
                    String deviceName = vb.getVariable().toString();
                    logger.debug("장비명 조회 성공: {} -> {}", ipAddress, deviceName);
                    return deviceName;
                }
            }
            
            // Walk가 실패하면 간단한 GET으로 시도
            PDU pdu = new PDU();
            pdu.setType(PDU.GET);
            pdu.add(new VariableBinding(new OID(SYS_NAME_OID)));
            
            ResponseEvent response = snmp.send(pdu, target);
            
            if (response != null && response.getResponse() != null) {
                PDU responsePDU = response.getResponse();
                if (responsePDU.getErrorStatus() == PDU.noError) {
                    VariableBinding vb = responsePDU.get(0);
                    if (vb != null && vb.getVariable() != null) {
                        String deviceName = vb.getVariable().toString();
                        logger.debug("장비명 조회 성공 (GET): {} -> {}", ipAddress, deviceName);
                        return deviceName;
                    }
                }
            }
            
            return null;
            
        } catch (Exception e) {
            logger.warn("장비명 조회 실패: {} - {}", ipAddress, e.getMessage());
            return null;
        } finally {
            if (snmp != null) {
                try {
                    snmp.close();
                } catch (IOException e) {
                    logger.warn("SNMP 연결 종료 중 오류: {}", e.getMessage());
                }
            }
        }
    }
}
