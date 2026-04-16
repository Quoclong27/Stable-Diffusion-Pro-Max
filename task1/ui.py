"""Task 1 — Gradio UI for Multi-Exposure Fusion."""

import gradio as gr
from .inference import run


def create_task1_tab():
    """Create the UI tab for Task 1 (Multi-Exposure Fusion)."""
    gr.Markdown(
        "**Multi-Exposure Fusion (UFRetinex-MEF-ToneNet)** — "
        "Tải 2+ ảnh với các mức độ sáng khác nhau để ghép thành 1 ảnh cân bằng ánh sáng."
    )

    with gr.Row():
        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            input_files = gr.File(
                label="Chọn ảnh (2+ ảnh có cùng kích thước)",
                file_count="multiple",
                file_types=["image"],
            )
            gr.Markdown(
                "<small style='color:#888'>Kéo thả hoặc nhấn để chọn file ảnh. "
                "Các ảnh phải có cùng kích thước (width × height).</small>"
            )

            with gr.Row():
                apply_phase2_chk = gr.Checkbox(
                    label="Tăng cường màu sắc (Phase 2 — ToneNet)",
                    value=True,
                    info="Nếu bỏ chọn: chỉ ghép ảnh, không tăng cường tông màu",
                )
                align_chk = gr.Checkbox(
                    label="Căn chỉnh SIFT (alignment)",
                    value=False,
                    info="Khớp các ảnh sử dụng đặc trưng SIFT nếu chúng bị lệch",
                )

            run_button = gr.Button("🚀 Ghép & Xử lý", variant="primary")

        # ── Right column ─────────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Kết quả",
                interactive=False,
                sources=[],
            )
            output_info = gr.Textbox(
                label="Thông tin",
                interactive=False,
                lines=2,
            )

    # ── Event handlers ───────────────────────────────────────────────────────
    run_button.click(
        fn=run,
        inputs=[input_files, apply_phase2_chk, align_chk],
        outputs=[output_image, output_info],
    )
