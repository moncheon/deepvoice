---
title: Deepvoice Run 0 Private Worker
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Private bounded deepvoice experiment worker
startup_duration_timeout: 1h
---

# Deepvoice Run 0 private worker

This Docker Space exposes health/status and a token-protected, enumerated audio-generation/training job trigger. It is intended for a newly created **private** Space only. It does not expose generated audio or a public evasion API.

Required Space secrets/settings are documented in `DEPLOY.md` in the source workspace. `/data` is ephemeral; successful model outputs must be uploaded to an existing private model repository and verified before pausing the Space.
