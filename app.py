"""WebDemo — Gradio entry point."""

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)

import gradio as gr

from model_manager import manager
from task1 import create_task1_tab, preload_to_cpu as preload_task1_cpu
from task2 import create_task2_tab, preload_to_cpu as preload_task2_cpu
from task3 import create_task3_tab, preload_to_cpu as preload_task3_cpu

# ── Preload ALL models to CPU RAM at startup ──────────────────────────────────
print("[Startup] Preloading Task 1 weights to CPU RAM (background)…")
preload_task1_cpu()
print("[Startup] Preloading Task 2 pipeline to CPU RAM (background)…")
preload_task2_cpu()
print("[Startup] Preloading Task 3 pipeline to CPU RAM (background)…")
preload_task3_cpu()

import time as _time
_time.sleep(1)  # give background threads a moment to start

print("[Startup] Activating Task 1 on GPU (default tab)…")
manager.activate("task1")
print("[Startup] Ready.")

# ── Tab indices ───────────────────────────────────────────────────────────────
_TASK_NAMES = ["task1", "task2", "task3"]

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Image Processing Demo") as demo:
    gr.Markdown("# Image Processing Demo")
    _loading_banner = gr.Markdown(visible=False)

    with gr.Tabs() as tabs:
        with gr.Tab("Task 1 — Multi-Exposure Fusion", id="tab1"):
            create_task1_tab()

        with gr.Tab("Task 2 — Inpainting & Editing", id="tab2"):
            create_task2_tab()

        with gr.Tab("Task 3 — Outpainting", id="tab3"):
            create_task3_tab()

    # ── Tab-switch handler ────────────────────────────────────────────────────
    _TAB_MAP = {
        "tab1": "task1", 0: "task1",
        "tab2": "task2", 1: "task2",
        "tab3": "task3", 2: "task3",
        "Task 1 — Multi-Exposure Fusion": "task1",
        "Task 2 — Inpainting & Editing": "task2",
        "Task 3 — Outpainting": "task3",
    }

    def _on_tab_select(evt: gr.SelectData):
        task = _TAB_MAP.get(evt.index) or _TAB_MAP.get(evt.value)
        print(f"[TabSwitch] evt.index={evt.index!r}  evt.value={evt.value!r}  → {task}")
        if task:
            manager.activate(task)
        return gr.update(visible=False)

    tabs.select(
        fn=_on_tab_select,
        inputs=None,
        outputs=_loading_banner,
    )

demo.queue(max_size=4).launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    theme=gr.themes.Soft(),
    enable_monitoring=False,
)
