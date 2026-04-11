"""WebDemo — Gradio entry point."""

import gradio as gr

from task1 import create_task1_tab, load_model as load_task1_model
from task2 import create_task2_tab
from task3 import create_task3_tab, load_model as load_task3_model

# ── Load models at startup ───────────────────────────────────────────────────
print("[Startup] Loading Task 1 model…")
load_task1_model()
print("[Startup] Loading Task 3 pipeline…")
load_task3_model()
print("[Startup] Ready.")

# ── Build UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Image Processing Demo") as demo:
    gr.Markdown("# Image Processing Demo")

    with gr.Tabs():
        with gr.Tab("Task 1 — Multi-Exposure Fusion"):
            create_task1_tab()

        with gr.Tab("Task 2 — Inpainting & Editing"):
            create_task2_tab()

        with gr.Tab("Task 3 — Outpainting"):
            create_task3_tab()

demo.queue(max_size=4).launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    theme=gr.themes.Soft(),
    enable_monitoring=False,
)
