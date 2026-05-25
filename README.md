bilibili视频地址：https://www.bilibili.com/video/BV1sCGd6eETT/?vd_source=c2d5118f96e671b06ffa90b85693baee
组员分工：李晓龙负责代码编写，刘发桢负责视频录制



# 🎙️ 智能语音输入法
## 📖 项目简介
本项目是一个具备自适应语义路由与本地向量缓存的商业级流式语音输入法。有别于传统的“纯发声-转写”Demo，本项目致力于解决实际工程落地中的高并发 API 开销、多领域（学术/编程）专业词汇识别权重干扰，以及底层 Web Audio 资源泄漏等痛点，实现了真正的 O(1) 级相似语义秒回与零延迟交互。

---

## 原创声明与第三方依赖明细 (Compliance Declaration)

根据评审规范，特此声明本项目的原创核心逻辑与引用的第三方开源框架：

### 核心原创功能部分 (100% 自主研发)
1. **智能语义路由器 (Semantic Router)：** 摒弃静态热词池，通过后端实时识别文本的零散特征，动态挂载对应领域（学术/编程/日常）的专属词汇表，防止跨领域权重污染。
2. **本地轻量级向量缓存 (Vector Semantic Cache)：** 不依赖外部笨重的向量数据库，基于 `scikit-learn` 手写 `TF-IDF + Cosine Similarity` 缓存层。实现 O(1) 级别的相似句式（允许语气词、少字）精准拦截与秒回。
3. **高可用录音生命周期治理：** 前端深度定制 Web Audio API 逻辑，配合 VAD 静音帧拦截（减少无效网络传输），并在结束录音时物理销毁 Audio 节点，彻底杜绝僵尸内存泄漏。
4. **全双工非阻塞网关转发层：** Java 端负责高性能的 WebSocket 流量透传与 Session 映射，实现前后端计算资源的彻底解耦。

### 第三方库与框架依赖清单
本项目站在巨人的肩膀上，以下基础能力由优秀的开源组件提供支持：
* **前端 (纯原生无框架)：** 仅使用 HTML5 Web Audio API 与原生 WebSocket。
* **网关层 (Java Backend)：** * `Spring Boot (3.x)`: 提供基础应用容器。
  * `Java-WebSocket`: 用于构建与 Python 引擎的双向长连接通信。
* **AI 引擎层 (Python Server)：**
  * `FastAPI` / `Uvicorn`: 提供高性能异步 WebSocket 服务端。
  * `FunASR (iic/speech_paraformer)`: 阿里达摩院开源的基础流式语音识别声学模型。
  * `OpenAI SDK`: 用于对接大语言模型（DeepSeek）的 API 接口，进行最终的文本标点与错别字精修。
  * `scikit-learn`: 用于构建本地字符级 N-gram 向量矩阵计算。
  * `jieba`: 用于 RAG 专属私有语料库上传时的 TF-IDF 热词抽取。
  * `pyautogui` / `pyperclip`: 提供系统级的剪贴板写入与自动化按键映射（模拟真实的输入法输出动作）。

---

## 快速部署与运行指引

本项目包含 Java 网关与 Python AI 引擎双微服务，请按以下步骤启动，确保主分支代码随时保持可运行状态（符合评审规范要求）。

### 1. 环境准备
* JDK 17+ & Maven
* Python 3.10+
* 麦克风硬件支持正常

### 2. 启动 Python AI 引擎
进入 `ai-server` 目录，安装依赖并启动服务：
```bash
cd ai-server
pip install -r requirements.txt
uvicorn asr_server:app --host 127.0.0.1 --port 8000
