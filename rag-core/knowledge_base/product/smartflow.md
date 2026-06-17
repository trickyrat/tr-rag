# SmartFlow 智能客服平台概述

SmartFlow 是一款基于大语言模型的智能客服平台，支持多渠道接入（网页、微信、APP），提供自动问答、工单流转、情感分析等功能。平台核心模块包括：对话引擎、知识库管理、数据分析仪表板。

## 部署要求

### 硬件最低配置
- CPU: 4核
- 内存: 16GB
- 磁盘: 100GB SSD
- GPU: 可选，推荐 NVIDIA T4 或更高（用于本地 embedding 模型）

### 软件依赖
- Ubuntu 20.04 / CentOS 8
- Docker 20.10+
- Python 3.9+
- ChromaDB 0.4.0+

## 快速安装（Docker）

```bash
# 拉取镜像
docker pull smartflow/smartflow:latest

# 运行容器
docker run -d \
  --name smartflow \
  -p 8080:8080 \
  -v ./data:/app/data \
  -e EMBEDDING_MODEL=BAAI/bge-large-zh \
  smartflow/smartflow:latest

访问 http://localhost:8080 即可开始使用。
```

## API 接口：发送消息

**端点**：`POST /api/v1/chat`

**请求体**：
```json
{
  "session_id": "uuid",
  "message": "用户问题",
  "channel": "web"
}
```

**响应**：

```json
{
  "answer": "回答内容",
  "confidence": 0.95,
  "suggested_actions": ["转人工", "查看订单"]
}
```

**错误码**：

- 400: 参数错误
- 429: 请求频率过高


## 如何配置 ChromaDB 持久化路径？

编辑 `config.yaml` 文件：

```yaml
vector_store:
  type: chromadb
  persist_directory: /var/lib/smartflow/chroma
  collection_name: knowledge_base
```

修改后需重启服务：docker restart smartflow。

## 故障排查：容器启动失败

**现象**：`docker run` 后容器立即退出，日志显示 `chromadb 连接失败`。

**原因**：ChromaDB 数据目录权限不足。

**解决方法**：
```bash
sudo chown -R 1000:1000 /var/lib/smartflow/chroma
docker restart smartflow
```

## 知识库更新流程

1. 在管理后台点击「知识库管理」
2. 上传 Markdown/PDF 文件
3. 系统自动分块并建立向量索引
4. 索引完成后，新知识即可被检索到
5. 如需回滚，可恢复上一版本备份

## 性能优化建议

- 使用 GPU 加速 embedding 可提升检索速度 3-5 倍
- 对于百万级知识库，建议将 ChromaDB 部署为独立服务
- 分块大小建议 512 tokens，重叠 10%
- 启用 Redis 缓存热点查询结果

## RRF 融合参数说明

`rrf_constant` 控制向量检索与 BM25 的融合平滑度。值越小，排名靠前的结果权重越高。推荐范围 30~100。

- 小常数（30）：更激进，强调第一名
- 大常数（100）：更平滑，给后面结果更多机会

## 常见问题：为什么召回准确率低于 90%？

可能原因及解决方案：
1. 分块不合理：使用 MarkdownHeaderTextSplitter 按标题切分
2. Embedding 模型弱：换成 BGE-large 或 GTE-Qwen2
3. 缺少重排序：增加 Cross-Encoder reranker
4. 融合参数不当：调低 rrf_constant 或改用加权归一化

## Webhook 事件订阅

SmartFlow 支持通过 Webhook 将用户对话事件推送到你的服务器。

**配置方式**：在管理后台「系统设置」→「Webhook」填入 URL。

**支持的事件类型**：
- `message.received`: 用户发送消息
- `agent.joined`: 人工客服接入
- `feedback.submitted`: 用户提交反馈

## 嵌入模型对比测试

| 模型              | 中文支持 | 召回 Hit@1 | 显存占用 |
| ----------------- | -------- | ---------- | -------- |
| all-MiniLM-L6-v2  | 差       | 0.62       | 1GB      |
| BAAI/bge-base-zh  | 好       | 0.78       | 2GB      |
| BAAI/bge-large-zh | 很好     | 0.85       | 4GB      |
| GTE-Qwen2-1.5B    | 优秀     | 0.91       | 6GB      |

