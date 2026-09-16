# Task 3: Retrieval-Augmented Generation (RAG) with Unsloth Dynamic 4-bit Quantization

## Overview
This repository contains a complete, memory-optimized Retrieval-Augmented Generation (RAG) pipeline built using Unsloth's dynamic 4-bit quantized language model (`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`). The implementation is designed to run efficiently within resource-constrained environments like Google Colab's Free T4 GPU runtime without encountering Out-Of-Memory (OOM) errors.

## Key Features & Workflow
* **Dynamic 4-bit Quantization:** Utilizes Unsloth's 4-bit quantized Llama 3.2 3B model, reducing VRAM footprint by up to 80% while retaining precision in critical parameters.
* **VRAM Monitoring:** Real-time CUDA memory tracking ensures optimal hardware utilization during model execution (~2.8 GB VRAM allocated).
* **Document Chunking & Vector Store:** Implements LangChain's `RecursiveCharacterTextSplitter` and `sentence-transformers/all-MiniLM-L6-v2` embeddings indexed inside a FAISS vector database.
* **Grounded Generation:** Formats retrieved document chunks into structured Llama 3 chat templates to ensure accurate, hallucination-free responses grounded purely in context.
* **Anti-Hallucination Guardrails:** Tested with out-of-bounds queries to ensure the model safely declines answering when relevant facts are missing from the context.
* **Dynamic PDF Loader:** Includes modular logic to ingest and process custom PDF documents for dynamic dataset handling.

## Tech Stack
* **LLM Engine:** Unsloth (FastLanguageModel)
* **Base Model:** `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Framework:** LangChain, PyTorch, Hugging Face Transformers
