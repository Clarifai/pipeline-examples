import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

import psutil
import requests
from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable

# Global reference for signal handler cleanup
_server_instance = None


def _signal_handler(signum, frame):
  """Handle SIGTERM and SIGINT to clean up child processes."""
  global _server_instance
  sig_name = signal.Signals(signum).name
  logger.info(f"Received {sig_name}, shutting down server...")
  if _server_instance is not None:
    _server_instance.close()
  sys.exit(0)


def _setup_signal_handlers(server_instance):
  """Register signal handlers for graceful shutdown."""
  global _server_instance
  _server_instance = server_instance
  signal.signal(signal.SIGTERM, _signal_handler)
  signal.signal(signal.SIGINT, _signal_handler)
  atexit.register(server_instance.close)


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
  """Kill the process and all its child processes."""
  if parent_pid is None:
    parent_pid = os.getpid()
    include_parent = False

  try:
    itself = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return

  children = itself.children(recursive=True)
  for child in children:
    if child.pid == skip_pid:
      continue
    try:
      child.kill()
    except psutil.NoSuchProcess:
      pass

  if include_parent:
    try:
      itself.kill()

      # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
      # so we send an additional signal to kill them.
      itself.send_signal(signal.SIGQUIT)
    except psutil.NoSuchProcess:
      pass


class OpenAI_APIServer:

  def __init__(self, **kwargs):
    self.server_started_event = threading.Event()
    self.process = None
    self.backend = None
    self.server_thread = None

  def __del__(self):
    self.close()

  def close(self):
    if self.process:
      try:
        kill_process_tree(self.process.pid)
      except Exception:
        try:
          self.process.terminate()
        except Exception:
          pass
      self.process = None
    if self.server_thread:
      self.server_thread.join(timeout=5)
      self.server_thread = None

  @classmethod
  def from_sglang_backend(
      cls,
      checkpoints,
      tp_size: int = 1,
      quantization: str = None,
      context_length: str = None,
      port: int = 23333,
      host: str = "0.0.0.0",
      chat_template: str = None,
      mem_fraction_static: float = 0.8,
      max_running_requests: int = None,
      max_total_tokens: int = None,
      max_prefill_tokens: int = None,
      additional_list_args: Optional[List[str]] = None,
  ):
    """Start SGlang OpenAI compatible server.

    Args:
        checkpoints (str): model id or path.
        tp_size (int, optional): The number of GPUs the model weights get sharded over. Defaults to 1.
        quantization (str, optional): Quantization format {"awq","fp8","gptq","marlin","gptq_marlin","awq_marlin","bitsandbytes","gguf","modelopt","w8a8_int8"}. Defaults to None.
        context_length (str, optional): The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).
        port (int, optional): Port number. Defaults to 23333.
        host (str, optional): Host name. Defaults to "0.0.0.0".
        chat_template (str, optional): The builtin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server. Defaults to None.
        mem_fraction_static (float, optional): The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. Defaults to 0.8.
        max_running_requests (int, optional): The maximum number of running requests. Defaults to None.
        max_total_tokens (int, optional): The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. Defaults to None.
        max_prefill_tokens (int, optional): The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. Defaults to None.
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at https://github.com/sgl-project/sglang. Defaults to None.

    Returns:
        OpenAI_APIServer: The server instance.
    """
    cmds = [
        PYTHON_EXEC, '-m', 'sglang.launch_server',
        '--model-path', checkpoints,
        '--tp-size', str(tp_size),
        '--mem-fraction-static', str(mem_fraction_static),
        '--port', str(port),
        '--host', host,
    ]
    if chat_template:
      cmds += ["--chat-template", chat_template]
    if quantization:
      cmds += ['--quantization', quantization]
    if context_length:
      cmds += ['--context-length', context_length]
    if max_running_requests:
      cmds += ['--max-running-requests', str(max_running_requests)]
    if max_total_tokens:
      cmds += ['--max-total-tokens', str(max_total_tokens)]
    if max_prefill_tokens:
      cmds += ['--max-prefill-tokens', str(max_prefill_tokens)]
    if additional_list_args:
      cmds += additional_list_args

    logger.info("CMDS to run `sglang` server: %s", " ".join(cmds))
    _self = cls()

    _self.host = host
    _self.port = port
    _self.backend = "sglang"

    logger.info("Checkpoints already exist" if os.path.exists(checkpoints) else f"Downloading checkpoints from {checkpoints}")

    env = os.environ.copy()
    env["VLLM_USAGE_SOURCE"] = "production-docker-image"

    server_url = f"http://{host}:{port}/health"
    timeout = 1800  # 30 minutes
    logger.info(f"Waiting for {server_url} (timeout={timeout}s)...")

    # Start the server process
    _self.process = subprocess.Popen(
        cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # Background thread to consume stdout to prevent pipe deadlock
    def log_process_output(proc):
      for line in proc.stdout:
        logger.info(f"SGLang: {line.strip()}")

    t = threading.Thread(target=log_process_output, args=(_self.process,), daemon=True)
    t.start()

    # Wait for server to be ready with timeout
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < timeout:
      # Check if process has crashed
      if _self.process.poll() is not None:
        raise RuntimeError(
            f"SGLang server process exited unexpectedly with code {_self.process.returncode}. "
            f"Check logs above for error details."
        )

      # Try to connect to health endpoint
      try:
        res = requests.get(server_url, timeout=1)
        if res.status_code == 200:
          server_ready = True
          break
      except requests.exceptions.RequestException:
        # Server not ready yet, wait and retry
        time.sleep(5)
        continue

    if not server_ready:
      _self.process.terminate()
      raise TimeoutError(
          f"SGLang server did not start within {timeout} seconds. "
          f"Check logs above for issues during initialization."
      )

    logger.info("SGLang server is ready!")

    _setup_signal_handlers(_self)

    return _self
