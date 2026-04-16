"""Task 3 — Gradio UI."""

import time

import gradio as gr

from .inference import (
    preview,
    infer,
    _resolve_res,
    SDXL_BUCKETS,
    _RESIZE_OPTS,
    _ALIGN_OPTS,
)
from .model import preload_to_cpu


def create_task3_tab():
    gr.Markdown(
        "**Mở rộng ảnh (Outpainting)** — Ảnh gốc sẽ được đặt lên canvas SDXL, "
        "vùng màu đỏ trong preview là nơi model tự vẽ thêm."
    )

    with gr.Row():
        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", label="Ảnh đầu vào",
                sources=["upload", "clipboard"],
            )
            gr.Markdown(
                "<small style='color:#888'>Kéo thả ảnh, nhấn 📎 để chọn file "
                "hoặc 📋 để dán từ clipboard</small>"
            )

            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt mô tả nội dung mở rộng (tuỳ chọn)", scale=2
                )
                run_button = gr.Button("Generate", variant="primary", scale=1)

            target_res = gr.Radio(
                choices=list(SDXL_BUCKETS.keys()),
                value="1:1  — 1024×1024",
                label="Kích thước đầu ra",
            )
            with gr.Row(visible=False) as custom_res_row:
                custom_w = gr.Number(value=1024, label="Width (px)",  precision=0, minimum=256, maximum=2048)
                custom_h = gr.Number(value=1024, label="Height (px)", precision=0, minimum=256, maximum=2048)

            with gr.Row():
                alignment = gr.Dropdown(
                    choices=_ALIGN_OPTS,
                    value="Middle",
                    label="Vị trí ảnh gốc trên canvas",
                    info="Middle=giữa, Left/Right/Top/Bottom=sát cạnh",
                    scale=1,
                )
                resize_option = gr.Radio(
                    choices=_RESIZE_OPTS,
                    value="Full",
                    label="Kích thước ảnh gốc",
                    scale=2,
                )

            custom_resize_pct = gr.Slider(
                10, 100, step=1, value=100,
                label="Custom resize (%)",
                visible=False,
            )

            with gr.Accordion("Cài đặt nâng cao", open=False):
                overlap_percentage = gr.Slider(
                    1, 50, step=1, value=10,
                    label="Mask overlap (%)",
                )
                with gr.Row():
                    overlap_top   = gr.Checkbox(label="Overlap Trên", value=True)
                    overlap_right = gr.Checkbox(label="Overlap Phải", value=True)
                with gr.Row():
                    overlap_left   = gr.Checkbox(label="Overlap Trái", value=True)
                    overlap_bottom = gr.Checkbox(label="Overlap Dưới", value=True)
                num_steps = gr.Slider(10, 50, step=5, value=20, label="Số bước inference")
                sharpen = gr.Slider(0.0, 2.0, step=0.1, value=1.0, label="Độ sắc nét")
                lora_scale = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                    label="Use Base Model"
                )

        # ── Right column ─────────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_tabs = gr.Tabs(selected="tab_preview")
            with output_tabs:
                with gr.Tab("Preview vùng mở rộng", id="tab_preview"):
                    preview_image = gr.Image(
                        interactive=False, container=False,
                        sources=[], show_label=False,
                    )
                with gr.Tab("Kết quả", id="tab_result"):
                    result_image = gr.Image(
                        interactive=False, container=False,
                        sources=[], show_label=False,
                    )
            info_box = gr.Textbox(label="Thông tin xử lý", interactive=False)

    with gr.Accordion("📋 Lịch sử xử lý", open=False):
        history_gallery = gr.Gallery(
            columns=4, object_fit="contain",
            interactive=False, allow_preview=False,
            show_label=False, label="",
        )
        clear_hist_btn = gr.Button("Xóa lịch sử", size="sm", variant="secondary")

    # ── Shared param lists ────────────────────────────────────────────────────
    _preview_inputs = [
        input_image, target_res, custom_w, custom_h,
        alignment, resize_option, custom_resize_pct,
        overlap_percentage, overlap_left, overlap_right,
        overlap_top, overlap_bottom,
    ]
    _all_inputs = _preview_inputs + [prompt_input, num_steps, sharpen, lora_scale]

    _INPUT_KEYS = [
        "input_image", "target_res", "custom_w", "custom_h",
        "alignment", "resize_option", "custom_resize_pct",
        "overlap_percentage", "overlap_left", "overlap_right",
        "overlap_top", "overlap_bottom",
        "prompt", "num_steps", "sharpen", "lora_scale",
    ]

    # ── Show/hide custom resolution row ──────────────────────────────────────
    target_res.change(
        fn=lambda v: gr.update(visible=(v == "Customize")),
        inputs=target_res,
        outputs=custom_res_row,
        queue=False,
    )

    # ── Show/hide custom resize slider ────────────────────────────────────────
    resize_option.change(
        fn=lambda opt: gr.update(visible=(opt == "Custom")),
        inputs=resize_option,
        outputs=custom_resize_pct,
        queue=False,
    )

    # ── Auto-preview on any canvas param change ───────────────────────────────
    for _comp in _preview_inputs:
        _comp.change(
            fn=preview,
            inputs=_preview_inputs,
            outputs=preview_image,
            queue=False,
        )

    input_image.change(
        fn=lambda: gr.update(selected="tab_preview"),
        outputs=output_tabs,
        queue=False,
    )

    # ── History ──────────────────────────────────────────────────────────────
    history_state = gr.State([])

    def _save_history(history, result_img, info, *all_inputs):
        if result_img is None:
            return history
        entry = {"output": result_img, "info": info or ""}
        for k, v in zip(_INPUT_KEYS, all_inputs):
            entry[k] = v
        return [entry] + history[:19]

    def _gallery_items(history):
        return [(e["output"], (e["info"] or "—")[:60]) for e in history]

    def _restore_history(history, evt: gr.SelectData):
        n_out = 1 + len(_all_inputs)
        if not history or evt.index >= len(history):
            return tuple(gr.update() for _ in range(n_out))
        e = history[evt.index]
        return (e["info"],) + tuple(e[k] for k in _INPUT_KEYS)

    clear_hist_btn.click(fn=lambda: ([], []), outputs=[history_state, history_gallery])
    history_gallery.select(
        fn=_restore_history,
        inputs=history_state,
        outputs=[info_box] + _all_inputs,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(*args):
        t0   = time.time()
        last = None
        for cnet_img, out_img in infer(*args):
            last = out_img
            yield last, ""
        dt   = round(time.time() - t0, 1)
        bw, bh = _resolve_res(args[1], args[2], args[3])
        steps  = args[-1]
        yield last, f"Hoàn thành trong {dt}s · {bw}×{bh}px · {steps} steps"

    for trigger in (run_button.click, prompt_input.submit):
        trigger(
            fn=lambda: (gr.update(value=None), gr.update(value="")),
            outputs=[result_image, info_box],
        ).then(
            fn=lambda: gr.update(selected="tab_result"),
            outputs=output_tabs,
        ).then(
            fn=_run,
            inputs=_all_inputs,
            outputs=[result_image, info_box],
        ).then(
            fn=_save_history,
            inputs=[history_state, result_image, info_box] + _all_inputs,
            outputs=history_state,
        ).then(
            fn=_gallery_items,
            inputs=history_state,
            outputs=history_gallery,
        )
