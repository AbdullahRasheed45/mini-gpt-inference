# mini-gpt-inference

Project B of a three-project ladder targeting ML/research engineer roles at
frontier labs: (A) distributed pretraining, (B) inference engine, (C) GRPO
post-training with a vLLM rollout server.

Project A ([mini-gpt-ddp](https://github.com/AbdullahRasheed45/mini-gpt-ddp))
built and trained the base model this project serves: an 8L/8H/512d,
50.9M-param GPT trained from scratch on TinyStories, checkpoint on the
Hugging Face Hub. This project is not started yet -- planning happens next.

## Goal (from Project A's roadmap, starting point for planning)

Inference lab: KV cache for the Project A model (measure the naive
`generate()` loop vs. a cached one), batched inference, then speculative
decoding with a draft model on one GPU and the target model on another.

## Status

Scaffolding only. Architecture, milestones, and scope are being planned in
a separate session.
