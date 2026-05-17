"""RAG问答链：检索 + DeepSeek生成"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

SYSTEM_PROMPT = """你是一个专业的供应链技术文档助手，服务于制造业采购团队。

请仅根据提供的上下文回答问题。
- 如果上下文中没有答案，直接说"文档中未找到相关信息"
- 用方括号引用来源文件，例如：[TE_spec.pdf]
- 技术参数用列表格式呈现
- 不要编造数字、型号、认证编号"""


def ask(question: str, chat_history: list = None) -> dict:
    chunks = retrieve(question)

    if not chunks:
        return {"answer": "知识库中暂无文档，请先上传文件。", "sources": []}

    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = c["metadata"].get("source", "?")
        page = c["metadata"].get("page")
        page_str = f" 第{page+1}页" if page is not None else ""
        context_parts.append(f"[Doc {i} | {source}{page_str}]\n{c['content']}")
    context = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\n上下文：\n{context}"}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        messages=messages,
        temperature=0.2,
    )

    sources = [{
        "file": c["metadata"].get("source", "unknown"),
        "page": c["metadata"].get("page"),
        "similarity": c["similarity"],
        "snippet": c["content"][:200],
    } for c in chunks]

    return {"answer": response.choices[0].message.content, "sources": sources}