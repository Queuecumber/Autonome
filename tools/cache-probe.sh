#!/usr/bin/env bash
# Test cache_control behavior across /v1/responses positions vs /v1/chat/completions.
# Each path runs twice; we want call 2 to show cache_read > 0 if caching worked.
set -u

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY before running}"
BASE="${NVIDIA_BASE_URL:-https://inference-api.nvidia.com}"
MODEL="${CACHE_PROBE_MODEL:-aws/anthropic/bedrock-claude-opus-4-6}"
PY="${PY:-$(command -v python3 || echo "$(pwd)/.venv/bin/python3")}"

# ~7k tokens of stable boilerplate — clears the 4096 min for Opus 4.6
PARA=$(printf 'Inference Hub is an OpenAI-compatible LLM gateway built on LiteLLM. It fronts many model backends and translates between API formats. This is stable boilerplate repeated to exceed the model minimum cacheable prefix length. %.0s' {1..80})

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

# A: /v1/chat/completions, system as content blocks + cache_control (known-good per PDF)
post "chat/completions BLOCKS+cache_control" "/v1/chat/completions" "$("$PY" -c "
import json, sys
PARA='''$PARA'''
print(json.dumps({
  'model': '$MODEL',
  'max_tokens': 32,
  'messages': [
    {'role':'system','content':[
      {'type':'text','text':'[chat]\n'+PARA,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"

# B: /v1/responses, instructions as plain string (no cache_control possible) — control
post "responses INSTRUCTIONS plain-string" "/v1/responses" "$("$PY" -c "
import json
PARA='''$PARA'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'instructions': '[resp-plain]\n'+PARA,
  'input': 'Reply with a brief greeting.'
}))
")"

# C: /v1/responses, instructions as content-blocks list with cache_control
post "responses INSTRUCTIONS blocks+cache_control" "/v1/responses" "$("$PY" -c "
import json
PARA='''$PARA'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'instructions': [
    {'type':'text','text':'[resp-instr-blocks]\n'+PARA,'cache_control':{'type':'ephemeral'}}
  ],
  'input': 'Reply with a brief greeting.'
}))
")"

# D: /v1/responses, cache_control on a developer-role message inside input (current code shape)
post "responses INPUT developer-msg blocks+cache_control" "/v1/responses" "$("$PY" -c "
import json
PARA='''$PARA'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'input': [
    {'role':'developer','content':[
      {'type':'input_text','text':'[resp-input-dev]\n'+PARA,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"

# E: /v1/responses, cache_control on a system-role message inside input
post "responses INPUT system-msg blocks+cache_control" "/v1/responses" "$("$PY" -c "
import json
PARA='''$PARA'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'input': [
    {'role':'system','content':[
      {'type':'input_text','text':'[resp-input-sys]\n'+PARA,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"
