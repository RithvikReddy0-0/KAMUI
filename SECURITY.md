# Security Policy

## Supported versions

KAMUI is a research and educational project. Security fixes are applied to the
latest release on the `main` branch only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/RithvikReddy0-0/KAMUI/security/advisories/new)**
(Security tab → "Report a vulnerability").

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal example is ideal),
- affected version or commit.

You can expect an acknowledgement within a few days. As a small project there
is no formal SLA, but confirmed issues will be addressed as a priority and
disclosed once a fix is available.

## Scope notes

- **Checkpoints are executable.** `kamui.training.load_checkpoint` /
  `load_model_only` call `torch.load(..., weights_only=False)`, which can
  execute arbitrary code during unpickling. **Only load checkpoints you
  produced or fully trust** — treat a `.pt` file like a Python script.
- KAMUI is intended for local research use; it is not hardened for running
  untrusted models or serving in production.
