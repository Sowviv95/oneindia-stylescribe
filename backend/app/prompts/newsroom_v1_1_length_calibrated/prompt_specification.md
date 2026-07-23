# Oneindia Newsroom V1.1 Length-Calibrated Prompt Specification

Version: `oneindia_newsroom_v1.1_length_calibrated`

This prompt is an optional Sprint 2 calibration variant. It preserves the
generic newsroom rules from `oneindia_newsroom_v1.0` and adds explicit length
control.

The prompt must not be used as the default v1.0 path. It is selected explicitly
through the benchmark runner with:

`--generation-mode newsroom_v1 --newsroom-prompt-version oneindia_newsroom_v1.1_length_calibrated`

The calibration goal is to reach the requested target range only when the
grounded brief and plan contain enough source-supported facts. It must never add
unsupported context, repetition or generic filler to satisfy length.
