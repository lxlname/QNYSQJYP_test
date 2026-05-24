package com.example.demo.config;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.BinaryWebSocketHandler;

import java.net.URI;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class VoiceGatewayHandler extends BinaryWebSocketHandler {

    // 维护前端 Session 和 Python ASR 客户端的映射
    private final Map<String, WebSocketClient> clientMap = new ConcurrentHashMap<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        //当前端连上 Java 时，Java 作为客户端去连接 Python AI
        URI aiServerUri = new URI("ws://127.0.0.1:8000/asr");
        WebSocketClient aiClient = new WebSocketClient(aiServerUri) {
            @Override
            public void onOpen(ServerHandshake handshakedata) { }

            @Override
            public void onMessage(String message) {
                // 收到 Python 识别的文字，转推给前端
                try {
                    if (session.isOpen()) {
                        session.sendMessage(new TextMessage(message));
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            @Override
            public void onClose(int code, String reason, boolean remote) { }
            @Override
            public void onError(Exception ex) { }
        };
        
        aiClient.connectBlocking(); // 同步建立连接
        clientMap.put(session.getId(), aiClient);
    }

    @Override
    protected void handleBinaryMessage(WebSocketSession session, BinaryMessage message) {
        // 收到前端发来的音频流，原封不动转给 Python
        WebSocketClient aiClient = clientMap.get(session.getId());
        if (aiClient != null && aiClient.isOpen()) {
            aiClient.send(message.getPayload());
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        // 断开连接时清理资源
        WebSocketClient aiClient = clientMap.remove(session.getId());
        if (aiClient != null) {
            aiClient.close();
        }
    }
}