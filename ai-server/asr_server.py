import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from funasr import AutoModel

app = FastAPI()

# 初始化 FunASR 模型 (首次启动会自动从 ModelScope 下载模型文件，请保持网络畅通)
print("⏳ 正在加载 Paraformer 流式语音识别模型，这可能需要一点时间...")
model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
print("✅ 模型加载完成！AI 引擎已就绪，等待音频流注入...")

@app.websocket("/asr")
async def asr_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📞 新的 WebSocket 客户端已连接")
    
    # 初始化流式识别的缓存状态 (chunk_size 控制语音切片的延迟)
    chunk_size = [0, 10, 5] 
    param_dict = {"cache": dict()}
    
    try:
        while True:
            # 1. 持续接收客户端发来的二进制 PCM 音频切片
            data = await websocket.receive_bytes()
            
            # 2. 送入模型进行流式推理
            rec_result = model.generate(
                input=data, 
                cache=param_dict["cache"], 
                is_final=False, 
                chunk_size=chunk_size
            )
            
            # 3. 解析结果并实时推回
            if rec_result and len(rec_result) > 0 and 'text' in rec_result[0]:
                text = rec_result[0]['text']
                if text.strip():
                    # 将文字实时发送回客户端
                    await websocket.send_text(text)
                    print(f"识别输出: {text}")
                    
    except WebSocketDisconnect:
        print("🔌 客户端正常断开连接")
    except Exception as e:
        print(f"❌ 识别过程中发生异常: {e}")
    finally:
        # 清理当前会话的上下文缓存
        param_dict["cache"].clear()
        print("🧹 资源清理完毕")