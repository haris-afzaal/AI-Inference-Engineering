# Inference Engine Decision Matrix
**Prepared By:** AI Systems Engineer (Track 1)
**Date:** November 2026

| Engine | Core Technology | Best Use Case | Weakness / Complexity |
| :--- | :--- | :--- | :--- |
| **vLLM** | PagedAttention (Block-based memory) | General purpose LLM serving, high throughput chatbots. | Duplicates KV cache for shared system prompts. Single-GPU focus. |
| **NVIDIA Triton** | Universal Orchestration | Running Audio + Vision + LLMs on the same server, strict routing. | Heavy setup, requires exact `config.pbtxt` for every model. |
| **SGLang** | RadixAttention (Prefix Caching) | RAG apps, Multi-turn Agents, long shared system prompts. | Newer ecosystem, fewer community extensions than vLLM. |

## Golden Rule of Inference:
- If building a simple chatbot -> **vLLM**
- If building an enterprise system with multiple model types -> **Triton**
- If building a RAG application with massive shared context -> **SGLang**
