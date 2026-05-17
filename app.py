"""供应链RAG系统 · Streamlit界面"""

import streamlit as st
import tempfile, os
from pathlib import Path
from chain import ask
from ingest import ingest

st.set_page_config(page_title="供应链知识库助手", page_icon="🔩", layout="wide")
st.title("🔩 供应链知识库助手")
st.caption("基于RAG · 支持产品手册、规格文档、合规文件问答")

with st.sidebar:
    st.header("📁 上传文件")
    uploaded = st.file_uploader("支持 PDF / TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded and st.button("📥 摄入到知识库", type="primary"):
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            with st.spinner(f"摄入 {f.name}..."):
                ingest(tmp_path)
            os.unlink(tmp_path)
            st.success(f"✅ {f.name} 摄入完成")
    st.divider()
    st.caption("💡 先上传TE Connectivity规格PDF，再开始问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 查看来源"):
                for s in msg["sources"]:
                    page = f" · 第{s['page']+1}页" if s.get("page") is not None else ""
                    st.markdown(f"**{s['file']}**{page} · 相似度 `{s['similarity']}`")
                    st.caption(s["snippet"])

if prompt := st.chat_input("问一个关于产品规格的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索知识库..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            result = ask(prompt, chat_history=history)
        st.write(result["answer"])
        if result["sources"]:
            with st.expander("📄 查看来源"):
                for s in result["sources"]:
                    page = f" · 第{s['page']+1}页" if s.get("page") is not None else ""
                    st.markdown(f"**{s['file']}**{page} · 相似度 `{s['similarity']}`")
                    st.caption(s["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })