ARG BASE_IMAGE
FROM ghcr.io/astral-sh/uv:0.7.13 AS uvbin

FROM ${BASE_IMAGE}

COPY --from=uvbin /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV OPENHANDS_VENV=/opt/openhands-venv

RUN uv python install 3.12 \
 && uv venv --seed --python 3.12 ${OPENHANDS_VENV} \
 && ${OPENHANDS_VENV}/bin/python -m pip install --upgrade pip setuptools wheel \
 && ${OPENHANDS_VENV}/bin/python -m pip install "openhands==1.13.1" \
 && chmod -R a+rX ${OPENHANDS_VENV} ${UV_PYTHON_INSTALL_DIR}

RUN /opt/openhands-venv/bin/python -c "from pathlib import Path; \
p=Path('/opt/openhands-venv/lib/python3.12/site-packages/openhands_cli/stores/agent_store.py'); \
s=p.read_text(); \
s=s.replace('load_user_skills=True,','load_user_skills=False,').replace('load_public_skills=True,','load_public_skills=False,'); \
p.write_text(s)"

# Force reasoning_effort='none' for thinking-capable models (Qwen3 family).
# Ollama 0.24+ honors reasoning_effort='none' on the OpenAI v1 endpoint to bypass <think> blocks.
# Without this, OpenHands defaults to reasoning_effort='high' and the model emits internal
# thinking that comes back as message.thinking, leaving message.content empty -> agent loop stalls.
RUN /opt/openhands-venv/bin/python -c "from pathlib import Path; \
p=Path('/opt/openhands-venv/lib/python3.12/site-packages/openhands_cli/stores/agent_store.py'); \
s=p.read_text(); \
s=s.replace('usage_id=\"agent\",\n        )', 'usage_id=\"agent\",\n            reasoning_effort=\"none\",\n        )'); \
s=s.replace('usage_id=\"condenser\",\n        )', 'usage_id=\"condenser\",\n            reasoning_effort=\"none\",\n        )'); \
p.write_text(s); \
assert 'reasoning_effort=\"none\"' in s, 'reasoning_effort patch did not match — agent_store.py layout changed'"

# Allow OPENHANDS_MAX_ITERATIONS env var to override the hard-coded 500-iteration cap.
# The cap burns iterations on AgentErrorEvents (tool-validation failures) as well as
# productive steps, so the default 500 is easy to exhaust on longer-context runs.
RUN /opt/openhands-venv/bin/python -c "from pathlib import Path; \
p=Path('/opt/openhands-venv/lib/python3.12/site-packages/openhands_cli/setup.py'); \
s=p.read_text(); \
s=s.replace('        hook_config=hook_config,\n    )', '        hook_config=hook_config,\n        max_iteration_per_run=int(__import__(\"os\").environ.get(\"OPENHANDS_MAX_ITERATIONS\", \"500\")),\n    )'); \
p.write_text(s); \
assert 'OPENHANDS_MAX_ITERATIONS' in s, 'max_iterations patch did not match — setup.py layout changed'"

# The default 300s HTTP timeout is too short for 35B long-context generations on
# the current Ollama/Vast setup. Raise it to 1800s so long raw/ADP prompts can
# complete without burning retries on avoidable APITimeoutError failures.
RUN /opt/openhands-venv/bin/python -c "from pathlib import Path; \
p=Path('/opt/openhands-venv/lib/python3.12/site-packages/openhands/sdk/llm/llm.py'); \
s=p.read_text(); \
old='default=300,'; \
new='default=1800,'; \
assert old in s, 'llm timeout patch did not match — llm.py layout changed'; \
p.write_text(s.replace(old, new, 1))"

# Default missing security_risk to UNKNOWN so longer prompts cannot deadlock the
# agent on tool-schema omissions.
RUN /opt/openhands-venv/bin/python -c "from pathlib import Path; \
p=Path('/opt/openhands-venv/lib/python3.12/site-packages/openhands/sdk/agent/agent.py'); \
s=p.read_text(); \
old='if requires_sr and raw is None:\n            raise ValueError(\n                f\"Failed to provide security_risk field in tool \\'{tool_name}\\'\"\n            )'; \
new='if requires_sr and raw is None:\n            return risk.SecurityRisk.UNKNOWN'; \
assert old in s, 'security_risk patch did not match — agent.py layout changed'; \
p.write_text(s.replace(old, new))"

RUN printf '%s\n' \
  '#!/bin/sh' \
  'if [ -n "${CONTEXTBENCH_WORKSPACE_ROOT:-}" ]; then' \
  '  git config --global --add safe.directory "$CONTEXTBENCH_WORKSPACE_ROOT" >/dev/null 2>&1 || true' \
  'fi' \
  'exec /opt/openhands-venv/bin/openhands "$@"' \
  > /usr/local/bin/openhands && chmod +x /usr/local/bin/openhands

RUN command -v openhands && openhands --version
