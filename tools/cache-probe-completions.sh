#!/usr/bin/env bash
# Does /v1/chat/completions propagate cache_control from messages too?
# Tries multi-breakpoint: cache_control on system + cache_control on a
# later message. If the later breakpoint fires, call 2 caches the full
# extension, not just the system block.
set -u

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY before running}"
BASE="${NVIDIA_BASE_URL:-https://inference-api.nvidia.com}"
MODEL="${CACHE_PROBE_MODEL:-aws/anthropic/bedrock-claude-opus-4-6}"
PY="${PY:-$(command -v python3 || echo "$(pwd)/.venv/bin/python3")}"

PARA1=$(printf 'Inference Hub is an OpenAI-compatible LLM gateway built on LiteLLM. It fronts many model backends and translates between API formats. This is stable boilerplate repeated to exceed the model minimum cacheable prefix length. %.0s' {1..80})
PARA2=$(printf 'The Anthropic Messages API supports up to four cache_control breakpoints per request. The longest matching cached prefix reads at the cache rate, with the remainder paid at full price. %.0s' {1..80})

post() {
  local label="$1" path="$2" body="$3"
  echo "== $label ($path)"
  for n in 1 2; do
    resp=$(curl -sS "$BASE$path" \
      -H "Authorization: Bearer $NVIDIA_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$body")
    usage=$(echo "$resp" | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print(json.dumps(r.get("usage", r.get("error", "NO_USAGE"))))' 2>/dev/null || echo "$resp" | head -c 300)
    echo "  call$n: $usage"
  done
}

# Baseline: system block only
post "completions system only" "/v1/chat/completions" "$("$PY" -c "
import json
PARA1='''$PARA1'''
print(json.dumps({
  'model': '$MODEL',
  'max_tokens': 32,
  'messages': [
    {'role':'system','content':[
      {'type':'text','text':'[c-base]\n'+PARA1,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"

# Multi: system + user content blocks both with cache_control
post "completions system + user" "/v1/chat/completions" "$("$PY" -c "
import json
PARA1='''$PARA1'''
PARA2='''$PARA2'''
print(json.dumps({
  'model': '$MODEL',
  'max_tokens': 32,
  'messages': [
    {'role':'system','content':[
      {'type':'text','text':'[c-su]\n'+PARA1,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':[
      {'type':'text','text':PARA2,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"

# Multi: system + assistant content blocks both with cache_control
post "completions system + assistant" "/v1/chat/completions" "$("$PY" -c "
import json
PARA1='''$PARA1'''
PARA2='''$PARA2'''
print(json.dumps({
  'model': '$MODEL',
  'max_tokens': 32,
  'messages': [
    {'role':'system','content':[
      {'type':'text','text':'[c-sa]\n'+PARA1,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'hello'},
    {'role':'assistant','content':[
      {'type':'text','text':PARA2,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"
