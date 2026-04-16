"""ModelManager — quản lý tập trung việc chuyển model GPU ↔ CPU khi đổi tab."""

import gc
import threading
import time
import torch


class ModelManager:
    """Quản lý các nhóm models theo task.

    Mỗi task đăng ký một dict { tên: nn.Module } sau khi load xong RAM.
    Khi activate(task_name):
      1. Đẩy task cũ về CPU
      2. Đưa task mới lên GPU
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._tasks: dict[str, dict[str, torch.nn.Module]] = {}
        self.active_task: str | None = None
        self._lock = threading.RLock()  # bảo vệ các trạng thái chung
        self._inference_count = 0
        self._inference_cond = threading.Condition(self._lock)

    def register(self, task_name: str, models: dict[str, torch.nn.Module]):
        """Đăng ký nhóm models của một task (gọi sau khi _load_to_ram xong).

        Args:
            task_name: tên task, ví dụ "task1"
            models: dict name→nn.Module, TẤT CẢ đang ở CPU
        """
        self._tasks[task_name] = models
        print(f"[ModelManager] Registered '{task_name}' with {list(models.keys())}")

    def is_registered(self, task_name: str) -> bool:
        return task_name in self._tasks

    def is_active(self, task_name: str) -> bool:
        return self.active_task == task_name

    def _move_module(self, module, device, dtype=None):
        device = torch.device(device)
        try:
            if dtype is not None:
                module.to(device, dtype=dtype)
            else:
                module.to(device)
        except NotImplementedError as exc:
            if "meta tensor" in str(exc).lower():
                if hasattr(module, 'to_empty'):
                    module.to_empty(device)
                else:
                    raise
            else:
                raise
        except TypeError:
            # Fallback cho custom model không hỗ trợ dtype (Task1Model)
            module.to(device)

    def activate(self, task_name: str, timeout: float = 180.0):
        """Chuyển task_name lên GPU, đẩy task cũ về CPU.

        Nếu task chưa được register (đang load background), đợi tối đa timeout giây.
        Thread-safe: dùng RLock để ngăn concurrent activate() racing nhau.
        """
        if self.active_task == task_name:
            return  # đã trên GPU rồi

        # Đợi task được register (background thread đang load) — NGOÀI LOCK để không block register()
        t0 = time.time()
        _last_log = 0.0
        while task_name not in self._tasks:
            elapsed = time.time() - t0
            if elapsed > timeout:
                raise RuntimeError(
                    f"[ModelManager] Timeout waiting for '{task_name}' to finish loading."
                )
            if elapsed - _last_log >= 5.0:
                print(f"[ModelManager] Waiting for '{task_name}' to finish loading… ({elapsed:.0f}s)")
                _last_log = elapsed
            time.sleep(0.5)

        with self._lock:
            while self._inference_count > 0 and self.active_task != task_name:
                self._inference_cond.wait()

            if self.active_task == task_name:
                return

            if self.active_task is not None and self.active_task in self._tasks:
                print(f"[ModelManager] Moving '{self.active_task}' → CPU RAM…")
                for name, model in self._tasks[self.active_task].items():
                    self._move_module(model, "cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"[ModelManager] Moving '{task_name}' → {self.device}…")
            for name, model in self._tasks[task_name].items():
                self._move_module(model, self.device)

            self.active_task = task_name
            print(f"[ModelManager] '{task_name}' ready on {self.device}.")

    def start_inference(self, task_name: str):
        """Đánh dấu bắt đầu inference của task.

        Nếu có inference khác đang chạy trên task khác, sẽ chờ đến khi hoàn thành.
        """
        with self._lock:
            while self._inference_count > 0 and self.active_task != task_name:
                self._inference_cond.wait()
            self._inference_count += 1

    def end_inference(self):
        """Kết thúc inference và thông báo để activate có thể chạy.
        """
        with self._lock:
            if self._inference_count <= 0:
                raise RuntimeError("[ModelManager] end_inference called without active inference")
            self._inference_count -= 1
            if self._inference_count == 0:
                self._inference_cond.notify_all()

    def deactivate_all(self):
        """Đẩy tất cả về CPU (dùng khi shutdown)."""
        for task_name, models in self._tasks.items():
            for model in models.values():
                self._move_module(model, "cpu")
        self.active_task = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Singleton — dùng chung toàn app
_DEVICE = "cuda:0"
manager = ModelManager(device=_DEVICE)
