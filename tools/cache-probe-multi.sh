#!/usr/bin/env bash
# Test whether a second cache_control breakpoint inside `input` fires
# when there's also one on top-level `instructions`. Each path runs twice;
# call 2 cached_tokens tells us how much got cached.
set -u

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY before running}"
BASE="${NVIDIA_BASE_URL:-https://inference-api.nvidia.com}"
MODEL="${CACHE_PROBE_MODEL:-aws/anthropic/bedrock-claude-opus-4-6}"
PY="${PY:-$(command -v python3 || echo "$(pwd)/.venv/bin/python3")}"

# Two stable prefixes, each ~7k tokens.
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

# Baseline: both breakpoints on instructions content blocks (known-good extension test).
post "responses TWO breakpoints, both in instructions" "/v1/responses" "$("$PY" -c "
import json
PARA1='''$PARA1'''
PARA2='''$PARA2'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'instructions': [
    {'type':'text','text':PARA1,'cache_control':{'type':'ephemeral'}},
    {'type':'text','text':PARA2,'cache_control':{'type':'ephemeral'}}
  ],
  'input': 'Reply with a brief greeting.'
}))
")"

# Test the multi-position case: one on instructions, one on a developer-role message in input.
post "responses instructions+input dev breakpoints" "/v1/responses" "$("$PY" -c "
import json
PARA1='''$PARA1'''
PARA2='''$PARA2'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'instructions': [
    {'type':'text','text':PARA1,'cache_control':{'type':'ephemeral'}}
  ],
  'input': [
    {'role':'developer','content':[
      {'type':'input_text','text':PARA2,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"

# Same but with input user-role message instead of developer.
post "responses instructions+input user breakpoints" "/v1/responses" "$("$PY" -c "
import json
PARA1='''$PARA1'''
PARA2='''$PARA2'''
print(json.dumps({
  'model': '$MODEL',
  'max_output_tokens': 32,
  'instructions': [
    {'type':'text','text':PARA1,'cache_control':{'type':'ephemeral'}}
  ],
  'input': [
    {'role':'user','content':[
      {'type':'input_text','text':PARA2,'cache_control':{'type':'ephemeral'}}
    ]},
    {'role':'user','content':'Reply with a brief greeting.'}
  ]
}))
")"
