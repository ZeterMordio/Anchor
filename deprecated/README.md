# Deprecated / historical artifacts

This directory keeps code and notes that are no longer the main path for improving negotiation performance, but remain useful for provenance or comparison.

| File | Status | Why archived |
|---|---|---|
| `engineering_notebook.md` | Historical notes only | Its useful content was consolidated into `JOURNAL.md`; keeping the original avoids losing early context without leaving it beside active docs. |
| `train_negotiation_dual_role.py` | Historical experiment script | The SPIRAL/RLVR dual-role approach trained one shared policy as buyer and seller with RAE. Current project direction is buyer-only SDPO+GRPO against a frozen regulated seller, because that matches the negotiation RLVR target more directly and avoids cross-role objective interference. |

Do not launch scripts from here as production runs without first revalidating APIs, prompts, defaults, and memory assumptions against the current active scripts.
