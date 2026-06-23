---
title: "Connecting a client"
description: "Point any OpenAI-compatible client at the gateway over the tailnet with a base URL, a token, and a model name."
weight: 20
---

`llmgw` speaks the OpenAI API, so anything that already talks to OpenAI can talk
to it. You change three things: the base URL, the API key, and the model name.
There is nothing else to install.

The base URL is the box's tailnet address with the data plane port and the `/v1`
prefix:

```text
http://100.71.238.128:8888/v1
```

If you have MagicDNS on, the box's MagicDNS name works the same way, which is
easier to remember and survives an address change. The API key is the token from
the `auth` block of the gateway config. Set it once in your environment so it
does not end up pasted into snippets:

```bash
export LLMGW_TOKEN="your-token-from-the-auth-block"
```

## OpenAI Python SDK

Override `base_url` and `api_key` on the client and use whatever model id or
alias the gateway exposes:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://100.71.238.128:8888/v1",
    api_key="your-token-from-the-auth-block",
)

resp = client.chat.completions.create(
    model="qwen3-30b",
    messages=[{"role": "user", "content": "Say hi in one line."}],
)
print(resp.choices[0].message.content)
```

Streaming works the same way, pass `stream=True` and iterate the chunks. The
gateway returns server-sent events just like the upstream API.

## curl

```bash
curl -s http://100.71.238.128:8888/v1/chat/completions \
  -H "Authorization: Bearer $LLMGW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "Say hi in one line."}]
  }'
```

## Editor and IDE assistants

Most coding assistants that support a custom OpenAI endpoint ask for three
fields. Fill them in like this:

```text
Base URL:  http://100.71.238.128:8888/v1
API key:   your-token-from-the-auth-block
Model:     qwen3-30b
```

Use a model id or an alias from the config. If the assistant lets you pick a fast
model for completions and a stronger one for chat, point them at two different
ids and let the gateway swap between them.

## A note on TLS

The box is reachable only over Tailscale, and the tailnet is the security
boundary. Traffic between your client and the box is already carried inside the
encrypted tailnet, so there is no TLS to terminate at the gateway and nothing is
exposed publicly. Plain `http://` to the tailnet address is the intended setup,
not a shortcut. Do not put the data plane on a public interface to add HTTPS, the
right move is to keep it tailnet-only.
