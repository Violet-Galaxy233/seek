from __future__ import annotations

import gradio as gr

from app.errors import SeekError


def search_gallery(runtime, description: str) -> tuple[list[tuple[str, str]], str]:
    normalized = " ".join((description or "").split())
    if not normalized:
        return [], "请输入封面描述。"
    try:
        result = runtime.get_service().search(description)
    except (SeekError, ValueError) as error:
        return [], f"⚠️ {error}"
    gallery = [
        (
            candidate.cover.image_path or candidate.cover.image_uri,
            f"{candidate.album.title} · {candidate.album.artist} · 相似度 {candidate.score:.3f}",
        )
        for candidate in result.candidates
    ]
    return gallery, f"找到 {len(gallery)} 张候选封面。相似度是图文向量余弦分数，不是概率。"


def build_ui(runtime) -> gr.Blocks:
    with gr.Blocks(title="Seek · 日语专辑封面搜索") as demo:
        gr.Markdown(
            "# Seek\n"
            "用中文描述你记得的日语专辑封面。搜索只使用本机数据库、封面和模型。"
        )
        status = gr.Markdown()
        description = gr.Textbox(
            label="封面描述",
            placeholder="例如：很多杂乱的小皮鞋，背景好像是红色",
            lines=2,
        )
        search_button = gr.Button("搜索相似专辑", variant="primary")
        gallery = gr.Gallery(
            label="相似专辑",
            columns=4,
            object_fit="cover",
            height="auto",
            interactive=False,
        )

        def runtime_message() -> str:
            current = runtime.initialize()
            icon = "✅" if current.state == "ready" else "⚠️"
            return f"{icon} {current.message}"

        def handle_search(text: str) -> tuple[list[tuple[str, str]], str]:
            return search_gallery(runtime, text)

        demo.load(runtime_message, outputs=status)
        search_button.click(handle_search, inputs=description, outputs=[gallery, status])
        description.submit(handle_search, inputs=description, outputs=[gallery, status])
    return demo
