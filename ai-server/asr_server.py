import asyncio
import os
import json
import time
import pyautogui
pyautogui.FAILSAFE = False  
import pyperclip
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from funasr import AutoModel
from openai import OpenAI

app = FastAPI()
CHUNK_SIZE = [0, 10, 5]
USER_DYNAMIC_HOTWORDS = ""

# ==========================================
# 🌟 领域知识库 (取代原先写死的 SCENES)
# ==========================================
LEXICONS = {
    "academic": "水下图像质量评价 FSCR-Net UIQA 注意力机制 卷积神经网络 评价模型",
    "coding": "for循环 while 递归 异常处理 Bug 接口 数据库 部署 编译器",
    "daily": "你好 吃饭 电影 散步 谢谢 没问题 聊天"
}

def get_smart_hotwords(text: str, current_scene: str) -> str:
    """语义路由引擎：如果用户说了特定领域词，自动提升该领域热词权重"""
    text = text.lower()
    # 匹配优先级：学术 > 编程 > 默认日常
    if any(k in text for k in ["评价", "模型", "论文", "网络", "机制"]):
        return LEXICONS["academic"]
    if any(k in text for k in ["代码", "循环", "函数", "数据库", "bug"]):
        return LEXICONS["coding"]
    return LEXICONS.get(current_scene, LEXICONS["daily"])

# ==========================================
# 🌟 企业级内存向量缓存系统 (Vector Semantic Cache)
# ==========================================
class VectorCache:
    def __init__(self):
        self.records = {} # 存储 query -> corrected_text

    def add(self, query: str, corrected: str):
        self.records[query] = corrected

    def search(self, query: str, threshold=0.75):
        if not self.records:
            return None, 0.0
        
        corpus = list(self.records.keys())
        # 使用字符级 N-gram 向量化，完美兼容中文错别字和冗余语气词
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])
        
        # 计算余弦相似度
        sims = cosine_similarity(query_vec, tfidf_matrix)[0]
        best_idx = sims.argmax()
        
        if sims[best_idx] >= threshold:
            return self.records[corpus[best_idx]], sims[best_idx]
        return None, 0.0

SEMANTIC_CACHE = VectorCache()

# ==========================================
# 🌟 大模型 Prompt 架构 (分离 System 与 User)
# ==========================================
SYSTEM_PROMPT = """你是一个专业的底层语音转写引擎。
你的唯一任务是对用户的语音识别文本进行纠错并添加标点。
绝对规则：
1. 只输出最终修正好的文本，绝不能包含任何解释、注脚或“好的”、“如下”等废话。
2. 下方提供的【RAG专属词库】仅在你认为发音高度相似时才作为纠错参考，不要生搬硬套。"""

client = OpenAI(
    api_key="sk-d0927a42f9be43218576ac3241d9f5f6",
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)

print("⏳ 正在加载 Paraformer 流式语音识别模型...")
model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
print("✅ 模型加载完成！智能路由 AI 引擎已就绪...")

def _extract_text(rec_result) -> str:
    if rec_result and len(rec_result) > 0 and "text" in rec_result[0]:
        return rec_result[0]["text"].strip()
    return ""

async def _stream_llm_and_type(websocket: WebSocket, full_text: str, scene: str, audio_duration: float, start_time: float) -> None:
    if not full_text.strip():
        return

    print(f"\n🗣️ 收到原始语音识别结果: [{full_text}]")

    # 1. 尝试触发向量语义缓存
    cached_text, sim_score = SEMANTIC_CACHE.search(full_text, threshold=0.80)
    if cached_text:
        print(f"⚡ 向量缓存命中! 相似度 {sim_score*100:.1f}%\n  直接输出: {cached_text}")
        await websocket.send_text("[LLM_START]")
        await websocket.send_text(f"[LLM_CHUNK]{cached_text}")
        _send_metrics(websocket, audio_duration, start_time, len(cached_text), hit_cache=True)
        _type_text(cached_text)
        return

    # 2. 缓存未命中，构造完美的 LLM 请求
    print("🤖 缓存未命中，正在呼叫大模型进行精修排版...")
    await websocket.send_text("[LLM_START]")
    
    # 动态选用最优提示词
    smart_base = get_smart_hotwords(full_text, scene)
    rag_context = USER_DYNAMIC_HOTWORDS if USER_DYNAMIC_HOTWORDS else smart_base
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n【RAG专属词库】：{rag_context}"},
        {"role": "user", "content": full_text}
    ]

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            messages=messages,
            stream=True
        )
        
        final_corrected_text = ""
        print("✍️ 大模型流式输出中: ", end="", flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                char = chunk.choices[0].delta.content
                final_corrected_text += char
                print(char, end="", flush=True) 
                await websocket.send_text(f"[LLM_CHUNK]{char}")
                await asyncio.sleep(0.01)
        print() 
        
        # 将原始结果与大模型修正结果存入向量缓存
        SEMANTIC_CACHE.add(full_text, final_corrected_text)
        
        _send_metrics(websocket, audio_duration, start_time, len(final_corrected_text), hit_cache=False)
        _type_text(final_corrected_text)
            
    except Exception as e:
        # 只在后台打印真实报错，绝对不污染前端用户的视线
        print(f"\n❌ 系统执行异常: {e}")
        # await websocket.send_text(f"[LLM_CHUNK] (系统开小差了，请重试)")

def _type_text(text: str):
    if text:
        pyperclip.copy(text)
        pyautogui.hotkey('ctrl', 'v')

def _send_metrics(websocket, audio_duration, start_time, text_len, hit_cache):
    process_time = time.time() - start_time
    rtf = process_time / audio_duration if audio_duration > 0 else 0
    tokens = text_len * 1.5 if not hit_cache else 0
    metrics = {
        "type": "metrics",
        "rtf": round(rtf, 3),
        "tokens": int(tokens),
        "cacheHit": "向量命中 (⚡0ms)" if hit_cache else "未命中 (API调用)"
    }
    asyncio.create_task(websocket.send_text(f"[METRICS]{json.dumps(metrics)}"))

@app.websocket("/asr")
async def asr_endpoint(websocket: WebSocket):
    global USER_DYNAMIC_HOTWORDS
    await websocket.accept()

    print("\n🟢 [服务器] -> 客户端握手成功，WebSocket 通道已建立！")
    
    try:
        while True:
            param_dict = {"cache": dict()}
            transcript_parts: list[str] = []
            current_scene = "academic"
            total_audio_bytes = 0
            start_time = time.time()

            while True:
                data = await websocket.receive_bytes()

                if data.startswith(b"RAG_TEXT:"):
                    content = data.decode("utf-8")[9:]
                    keywords = jieba.analyse.extract_tags(content, topK=10)
                    USER_DYNAMIC_HOTWORDS = " ".join(keywords)
                    print(f"📚 RAG 热词提取成功: {USER_DYNAMIC_HOTWORDS}")
                    await websocket.send_text(f"[RAG_DONE]已提取专属词库: {USER_DYNAMIC_HOTWORDS}")
                    continue

                if len(data) <= 20 and data.startswith(b"EOF_"):
                    current_scene = data.decode("utf-8").split("_")[1]
                    break

                total_audio_bytes += len(data)
                
                # 🌟 核心修改：动态获取当前识别到的文字串，进行路由
                current_transcript = "".join(transcript_parts)
                base_hotword = get_smart_hotwords(current_transcript, current_scene)
                combined_hotword = f"{base_hotword} {USER_DYNAMIC_HOTWORDS}".strip()

                rec_result = model.generate(
                    input=data,
                    cache=param_dict["cache"],
                    is_final=False,
                    chunk_size=CHUNK_SIZE,
                    hotword=combined_hotword,
                )
                text = _extract_text(rec_result)
                if text:
                    transcript_parts.append(text)
                    await websocket.send_text(text)

            audio_duration = total_audio_bytes / 32000.0

            # 最终收尾识别时，再次进行语义路由
            current_transcript = "".join(transcript_parts)
            base_hotword = get_smart_hotwords(current_transcript, current_scene)
            combined_hotword = f"{base_hotword} {USER_DYNAMIC_HOTWORDS}".strip()
            
            rec_result = model.generate(
                input=b"",
                cache=param_dict["cache"],
                is_final=True,
                chunk_size=CHUNK_SIZE,
                hotword=combined_hotword.strip(),
            )
            final_text = _extract_text(rec_result)
            if final_text:
                transcript_parts.append(final_text)

            full_text = "".join(transcript_parts)
            if full_text.strip():
                await _stream_llm_and_type(websocket, full_text, current_scene, audio_duration, start_time)
            
            param_dict["cache"].clear()

    except WebSocketDisconnect:
        print("🔌 客户端已断开")