---
type: concept
title: Project overview
description: Scope and research question for SarcasTone.
tags: [scope, research-question]
---

# Project overview

**Goal:** compare text-only, speech-only and multimodal sarcasm detection.

**Research question:** how much does adding speech prosody help when tone contradicts the literal words?

This bundle covers only the text branch. Speech and fusion are owned by other teammates.

**Timeline:** 2–3 weeks.

**Training constraint:** do not train encoders from scratch. Use frozen pre-trained encoders with small trained heads.

See [dataset](../data/dataset.md), [models](../models/roberta.md) and [evaluation conventions](../evaluation/conventions.md).
