import asyncio
import os
import pyautogui
import pyperclip

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from funasr import AutoModel
from openai import OpenAI

app = FastAPI()

# --- 核心配置 ---
CHUNK_SIZE = [0, 10, 5]

# 多场景配置字典 (包含热词和专属 Prompt)
SCENES = {
    "academic": {
        "hotword": "FSCR-Net UIQA 注意力机制 对比学习",
        "prompt": "你是一个学术语音输入法。请对以下无标点文本进行语义纠错并添加正确标点。特别注意学术词汇（如 FSCR-Net, UIQA等）拼写必须正确。直接输出最终文本，不带任何解释：\n\n{text}"
    },
    "coding": {
        "hotword": "for循环 while 递归 报错 异常处理 Bug",
        "prompt": "你是一个程序员语音输入法。请对以下无标点文本进行纠错并添加标点。特别注意将中文编程术语转换为正确的英文单词或符号（如将‘负循环’改为‘for循环’，‘布尔’改为‘boolean’）。直接输出最终文本，不带任何解释：\n\n{text}"
    },
    "daily": {
        "hotword": "好的 收到 没问题 哈哈 晚安",
        "prompt": "你是一个高情商的日常聊天输入法。请对以下无标点文本进行纠错，并加上恰当的标点符号和语气词（如果适合的话）。使其读起来自然、得体。直接输出最终文本，不带任何解释：\n\n{text}"
    }
}

client = OpenAI(
    api_key="sk-d0927a42f9be43218576ac3241d9f5f6", 
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)

print("⏳ 正在加载 Paraformer 流式语音识别模型...")
model = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
)
print("✅ 模型加载完成！AI 引擎已就绪...")

def _extract_text(rec_result) -> str:
    if rec_result and len(rec_result) > 0 and "text" in rec_result[0]:
        return rec_result[0]["text"].strip()
    return ""

# LLM 流式输出与键盘映射逻辑
async def _stream_llm_and_type(websocket: WebSocket, full_text: str, scene: str) -> None:
    if not full_text.strip():
        return

    prompt = SCENES.get(scene, SCENES["daily"])["prompt"]
    
    try:
        print(f"🤖 正在呼叫大模型 ({scene} 模式) 进行流式精修...")
        # 1. 发送清屏信号给前端
        await websocket.send_text("[LLM_START]")
        
        # 2. 开启流式请求
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt.format(text=full_text)}],
            stream=True # 核心：开启打字机模式
        )
        
        final_corrected_text = ""
        # 3. 逐字接收并推给前端
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                char = chunk.choices[0].delta.content
                final_corrected_text += char
                await websocket.send_text(f"[LLM_CHUNK]{char}")
                await asyncio.sleep(0.01) # 微小延迟，让前端动画更平滑
        
        # 4. OS 级键盘输入 (终极魔法)
        if final_corrected_text:
            print(f"⌨️ 正在模拟键盘输入: {final_corrected_text}")
            pyperclip.copy(final_corrected_text) # 将文字拷入剪贴板以防中文乱码
            # 模拟按下 Ctrl + V (如果是 Mac 用户，请把 'ctrl' 改为 'command')
            pyautogui.hotkey('ctrl', 'v') 
            
    except Exception as e:
        print(f"⚠️ LLM 纠错失败: {e}")
        await websocket.send_text(f"[LLM_CHUNK]纠错失败，原文本: {full_text}")

@app.websocket("/asr")
async def asr_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📞 新的 WebSocket 客户端已连接")

    try:
        # 🌍 核心修复：外层无限循环，保持长连接不断，支持连续多次说话！
        while True: 
            param_dict = {"cache": dict()}
            transcript_parts: list[str] = []
            current_scene = "academic" 

            # 🎶 内层循环：处理用户单次说话的音频流
            while True:
                data = await websocket.receive_bytes()

                # 收到前端松开按钮的结束暗号，跳出内层循环
                if len(data) <= 20 and data.startswith(b"EOF_"):
                    current_scene = data.decode("utf-8").split("_")[1]
                    print(f"📨 收到结束暗号，当前场景: {current_scene}")
                    break

                current_hotword = SCENES.get(current_scene, SCENES["academic"])["hotword"]
                rec_result = model.generate(
                    input=data,
                    cache=param_dict["cache"],
                    is_final=False,
                    chunk_size=CHUNK_SIZE,
                    hotword=current_hotword,
                )

                text = _extract_text(rec_result)
                if text:
                    transcript_parts.append(text)
                    await websocket.send_text(text)

            # 🛑 用户说完了，进行收尾和 LLM 流式输出
            current_hotword = SCENES.get(current_scene, SCENES["academic"])["hotword"]
            rec_result = model.generate(
                input=b"",
                cache=param_dict["cache"],
                is_final=True,
                chunk_size=CHUNK_SIZE,
                hotword=current_hotword,
            )
            final_text = _extract_text(rec_result)
            if final_text:
                transcript_parts.append(final_text)

            full_text = "".join(transcript_parts)
            if full_text.strip():
                # 触发大模型流式打字机效果
                await _stream_llm_and_type(websocket, full_text, current_scene)
            
            param_dict["cache"].clear()
            print("🧹 本次识别与输出完毕，引擎待机，等待您下一次说话...\n")

    except WebSocketDisconnect:
        print("🔌 客户端网页已关闭，连接断开")
    except Exception as e:
        print(f"❌ 发生异常: {e}")