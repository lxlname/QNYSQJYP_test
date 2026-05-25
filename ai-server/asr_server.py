import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from funasr import AutoModel
from openai import OpenAI

app = FastAPI()

HOTWORD = "FSCR-Net UIQA 注意力机制"
CHUNK_SIZE = [0, 10, 5]

LLM_PROMPT = (
    "你是一个专业的学术语音输入法引擎。请对以下语音识别生成的无标点文本进行语义纠错并添加正确的标点符号。"
    "请特别注意纠正同音字错误，确保学术词汇（如 FSCR-Net, UIQA, 注意力机制等）拼写绝对正确。"
    "请直接输出最终的完美文本，不要包含任何多余的解释或问候：\n\n"
    "{text}"
)

client = OpenAI(
    api_key="sk-d0927a42f9be43218576ac3241d9f5f6",
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)

print("⏳ 正在加载 Paraformer 流式语音识别模型，这可能需要一点时间...")
model = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
)
print("✅ 模型加载完成！AI 引擎已就绪，等待音频流注入...")


def _extract_text(rec_result) -> str:
    if rec_result and len(rec_result) > 0 and "text" in rec_result[0]:
        return rec_result[0]["text"].strip()
    return ""


def _refine_with_llm(text: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": LLM_PROMPT.format(text=text)}],
    )
    return response.choices[0].message.content.strip()


async def _send_llm_refined_result(websocket: WebSocket, full_text: str) -> None:
    if not full_text.strip():
        return

    try:
        print("🤖 正在呼叫大模型进行语义精修...")
        corrected = await asyncio.to_thread(_refine_with_llm, full_text)
    except Exception as e:
        print(f"⚠️ LLM 纠错失败，回退为原始文本: {e}")
        corrected = full_text

    message = f"[大模型精修版]: {corrected}"
    await websocket.send_text(message)
    print(f"精修输出: {message}")


@app.websocket("/asr")
async def asr_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📞 新的 WebSocket 客户端已连接")

    param_dict = {"cache": dict()}
    transcript_parts: list[str] = []

    try:
        while True:
            data = await websocket.receive_bytes()

            # 结束暗号：前端停止录音时发送的极短控制包（len <= 10）
            if len(data) <= 10:
                print("📨 收到结束暗号，准备进入大模型精修")
                break

            rec_result = model.generate(
                input=data,
                cache=param_dict["cache"],
                is_final=False,
                chunk_size=CHUNK_SIZE,
                hotword=HOTWORD,
            )

            text = _extract_text(rec_result)
            if text:
                transcript_parts.append(text)
                await websocket.send_text(text)
                print(f"识别输出: {text}")

    except WebSocketDisconnect:
        print("🔌 客户端正常断开连接")
    except Exception as e:
        print(f"❌ 识别过程中发生异常: {e}")
    finally:
        try:
            rec_result = model.generate(
                input=b"",
                cache=param_dict["cache"],
                is_final=True,
                chunk_size=CHUNK_SIZE,
                hotword=HOTWORD,
            )
            final_text = _extract_text(rec_result)
            if final_text:
                transcript_parts.append(final_text)

            full_text = "".join(transcript_parts)
            if full_text.strip():
                await _send_llm_refined_result(websocket, full_text)
        except Exception as e:
            print(f"⚠️ 最终精修发送失败（连接可能已关闭）: {e}")
        finally:
            param_dict["cache"].clear()
            print("🧹 资源清理完毕")
