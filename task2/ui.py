"""Task 2 — Gradio UI for Inpainting & Editing."""

import gradio as gr
from .inference import run_inference


def create_task2_tab():
    """Create the UI tab for Task 2 (SDXL Inpainting)."""
    gr.Markdown(
        "**Sửa & Điền ảnh (Inpainting)** — "
        "Vẽ mask trên vùng muốn sửa/điền, sau đó chọn chế độ Delete hoặc Add."
    )

    with gr.Row():
        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            image_editor = gr.ImageEditor(
                label="Vẽ mask (lớp tròn/vuông)",
                type="pil",
                sources=["upload", "clipboard"],
                layers=True,
            )
            gr.Markdown(
                "<small style='color:#888'>Kéo/vẽ lên vùng cần sửa. "
                "Nền trắng = bỏ qua, lớp vẽ = mask.</small>"
            )

            prompt_input = gr.Textbox(
                label="Prompt yêu cầu (cho chế độ Add)",
                lines=2,
                placeholder="Ví dụ: a modern lamp, warm lighting, ...",
            )

            with gr.Row():
                task_type = gr.Radio(
                    choices=["Delete", "Add"],
                    value="Delete",
                    label="Chế độ",
                    info="Delete=xóa, mặc định prompt; Add=thêm nội dung mới",
                )
                run_button = gr.Button("🚀 Xử lý", variant="primary", scale=1)

            with gr.Accordion("Tham số nâng cao", open=False):
                steps_slider = gr.Slider(
                    10, 50, step=1, value=30,
                    label="Số bước inference",
                )
                strength_slider = gr.Slider(
                    0.1, 1.0, step=0.05, value=1.0,
                    label="Strength (ảnh hưởng của mask)",
                )
                guidance_scale_slider = gr.Slider(
                    1.0, 20.0, step=0.5, value=12.0,
                    label="Guidance Scale (CFG)",
                )
                cn_scale_slider = gr.Slider(
                    0.0, 1.0, step=0.1, value=0.3,
                    label="ControlNet Scale (hướng khỏe độ)",
                )

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
        fn=run_inference,
        inputs=[
            image_editor,
            prompt_input,
            steps_slider,
            task_type,
            strength_slider,
            guidance_scale_slider,
            cn_scale_slider,
        ],
        outputs=[output_image, output_info],
    )
