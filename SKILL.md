---
name: slack-archive-search
description: Search archived multi-workspace Slack messages and threads through slackquery MCP.
---

# Slack Archive Search

- `list_slack_scopes`: discover stable workspace/channel IDs and archive coverage.
- `search_slack`: retrieve messages; use `mode="hybrid"` by default.
- `get_slack_message`: exact lookup with bounded same-channel context.
- `get_slack_thread`: expand one promising result chronologically.

Use the narrowest known workspace, channel, author, and time filters before
increasing limits. Use lexical mode for exact IDs, errors, filenames, URLs, code,
and quoted phrases. Use semantic mode for paraphrased how/why questions. Scores
rank candidates; they are not probabilities. Expand only promising threads.

Cite workspace, channel, author, timestamp, and `document_id`; include the
permalink when available. Cursors are opaque and bound to one artifact, query,
mode, and filter set. No result does not prove absence: filters, spelling,
archive dates, deletion, and incomplete coverage can hide information.

Harness-specific configuration is in `docs/clients/README.md`.

Operator note: semantic and hybrid search use the active local embedding
transport selected by `EMBEDDING_BACKEND`; PyTorch and Ollama remain one vector
generation only when exact model digest, 768 native dimensions, prefixes,
first-512 truncation, and L2 normalization all match. Check transport health with
`slackquery embedding-status`; do not re-embed solely when switching compatible
backends.
