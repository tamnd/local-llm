#!/usr/bin/env bash
# run-bench-quality.sh: run the quality benchmark suite via Windows Task
# Scheduler so the job survives SSH disconnect on the GamingPC.
#
# Run this from WSL2. It writes the install + run commands to a helper
# script, registers a one-shot Task Scheduler job, fires it, and exits.
# Check progress with: tail -f /root/bench-quality.log
#
# Usage:
#   ./scripts/run-bench-quality.sh [model,model,...] [extra bench-quality.py args]
#
# Examples:
#   ./scripts/run-bench-quality.sh
#   ./scripts/run-bench-quality.sh qwen3.6:27b,qwen3.6:35b --skip humaneval
#   OLLAMA_URL=http://127.0.0.1:11434 ./scripts/run-bench-quality.sh

set -euo pipefail

MODELS="${1:-qwen3.6:27b,qwen3.6:35b}"
EXTRA_ARGS="${*:2}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
LOG="/root/bench-quality.log"
HELPER="/root/run_bench_quality_bg.sh"
SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/bench-quality.py"

# Write the helper script that Task Scheduler will run inside WSL2.
python3 -c "
content = '''#!/bin/bash
exec > ${LOG} 2>&1
set -x
echo start: \$(date)
OLLAMA_URL=${OLLAMA_URL} /opt/vllm-venv/bin/python ${SCRIPT_SRC} \\\\
    --models ${MODELS} ${EXTRA_ARGS}
echo done: \$(date)
'''
open('${HELPER}', 'w').write(content)
import os; os.chmod('${HELPER}', 0o755)
print('helper written to ${HELPER}')
"

# Register and launch via Windows Task Scheduler (survives SSH disconnect).
powershell.exe -Command "
\$action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument '-e ${HELPER}'
\$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
\$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName 'BenchQuality' -Action \$action -Trigger \$trigger -Settings \$settings -Force | Out-Null
Start-ScheduledTask -TaskName 'BenchQuality'
Write-Host 'BenchQuality task started'
"

echo ""
echo "Benchmark running in background. Follow progress:"
echo "  tail -f ${LOG}"
echo ""
echo "Results will be saved to /tmp/bench-quality.json"
