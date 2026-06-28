#!/usr/bin/env bash
# Does moving the cache_control marker forward across requests invalidate
# the prior cache entry? Test by:
#   1. Call A: messages [system, m1, m2] with cache_control on m2
#   2. Call B: messages [system, m1, m2, m3, m4] with cache_control on m4
# If B reads the prefix up through m2, Anthropic treats cache_control as
# metadata and sliding markers extend cache. If B only reads system,
# sliding markers invalidate cache and we need a rolling-markers strategy.
set -u

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY before running}"
BASE="${NVIDIA_BASE_URL:-https://inference-api.nvidia.com}"
MODEL="${CACHE_PROBE_MODEL:-aws/anthropic/bedrock-claude-opus-4-6}"

# Each message above the cache point must total >=4096 tokens for opus.
# Use ~5k tokens of stable text in m1 + system.
SYSTEM=$(printf 'You are a helper. Stable system content. %.0s' {1..200})
M1=$(printf 'Static historical message one with stable content for caching. %.0s' {1..200})
M2=$(printf 'Static historical message two with stable content for caching. %.0s' {1..200})

show() {
  curl -sS "$BASE/v1/chat/completions" \
    -H "Authorization: Bearer $NVIDIA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$1" 2>&1 | grep -oE '"prompt_tokens":[0-9]+|"cached_tokens":[0-9]+|"cache_creation_tokens":[0-9]+|"cache_read_input_tokens":[0-9]+'
}

echo "== Call A: marker on m2 (last) =="
BODY_A=$(cat <<EOF
{
  "model": "$MODEL",
  "max_tokens": 16,
  "messages": [
    {"role":"system","content":[{"type":"text","text":"$SYSTEM","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"user","content":"$M1"},
    {"role":"user","content":[{"type":"text","text":"$M2","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"user","content":"reply briefly"}
  ]
}
EOF
)
show "$BODY_A"

echo ""
echo "== Call B: marker moves past m2 (new content m3, marker on m3) =="
M3=$(printf 'New message three added in second call with stable content. %.0s' {1..100})
BODY_B=$(cat <<EOF
{
  "model": "$MODEL",
  "max_tokens": 16,
  "messages": [
    {"role":"system","content":[{"type":"text","text":"$SYSTEM","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"user","content":"$M1"},
    {"role":"user","content":"$M2"},
    {"role":"user","content":[{"type":"text","text":"$M3","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
    {"role":"user","content":"reply briefly"}
  ]
}
EOF
)
show "$BODY_B"
