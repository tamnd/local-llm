# Provisioning scripts

These bring up the RTX 4090 box from a clean Windows 11 install to a serving
local-LLM stack. Run them in order, elevated, as the `gopher` user. Each is
idempotent: re-running a completed step prints what it found and exits clean.

```powershell
.\scripts\01-driver.ps1        # verify NVIDIA driver 566.36 (manual install if wrong)
.\scripts\02-openssh.ps1       # OpenSSH server, key-only over the tailnet
.\scripts\03-ollama.ps1        # Ollama 0.30.6 -> OllamaServe task, loopback 11434
.\scripts\04-llamacpp.ps1      # llama.cpp b9553 CUDA build -> C:\llama.cpp
.\scripts\05-python-tabby.ps1  # Python 3.11 + venv + ExLlamaV3 + TabbyAPI, TabbyServe task on 5000
.\scripts\06-gateway.ps1       # build bin\llmgw.exe -> GatewayServe task, data 8888 / admin 8889
.\scripts\07-models.ps1        # pull every model in catalog\models.manifest.yaml
.\scripts\health-check.ps1     # pass/fail across driver, backends, gateway, end-to-end
```

`common.ps1` is dot-sourced by every script for `Log`, `Fail`, `Test-HttpOk`, and
`Register-Service` (the one place the scheduled-task shape is defined).
`hf-download.py` runs inside the Tabby venv to fetch EXL2 quants from Hugging Face.

Ports: Ollama 11434, TabbyAPI 5000, gateway data plane 8888, gateway admin plane
8889 (loopback). Only the gateway data plane is tailnet-facing; every backend is
loopback-only and reached through the gateway.

Before 06 runs, fill in real auth tokens in `configs\llmgw.yaml`. The script
refuses to start a tailnet-facing gateway while the `REPLACE_ME` placeholders are
still in place.
