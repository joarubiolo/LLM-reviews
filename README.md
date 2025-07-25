# Aprendiendo LLMs

## Nivel 1: Fundamentos de Procesamiento de Lenguaje Natural (NLP)
  Descripción:
  Antes de sumergirte en LLMs, es crucial entender los fundamentos de NLP, incluyendo representaciones de texto, modelos clásicos y redes neuronales básicas.
  
  Temas clave:
  - Tokenización y preprocesamiento de texto (BPE, WordPiece).
  
  - Embeddings de palabras (Word2Vec, GloVe, FastText).
  
  - Modelos secuenciales (RNNs, LSTMs, GRUs).
  
  - Transformers (arquitectura base para LLMs) (Attention, Self-Attention, Encoder-Decoder).
  
  Pasos prácticos:
  - Implementar un modelo de clasificación de texto con LSTMs (PyTorch/TensorFlow).
  
  - Experimentar con embeddings usando Word2Vec/GloVe en un dataset como IMDB.
  
  - Implementar un Transformer básico desde cero (ej: traductor español-inglés).
  
## Nivel 2: Modelos de Lenguaje (LLMs)
  Descripción:
  Los LLMs (GPT, BERT, T5, etc.) son modelos basados en Transformers entrenados en grandes corpus de texto para tareas generativas o de comprensión.
  
  Temas clave:
  - Arquitecturas clave:
  
  - GPT (Generative Pre-trained Transformer, decodificador).
  
  - BERT (Bidirectional Encoder Representations, codificador).
  
  - T5 (Text-to-Text Transfer Transformer).
  
  - Fine-tuning vs. Prompting.
  
  - Modelos Open-Source (LLaMA, Mistral, Falcon).
  
  Pasos prácticos:
  - Cargar un modelo pre-entrenado (GPT-2 o BERT) con Hugging Face.
  
  - Hacer fine-tuning de BERT para una tarea específica (ej: QA con SQuAD).
  
  - Experimentar con prompting en GPT-3.5/4 o modelos open-source (LLaMA).
  
## Nivel 3: Técnicas Avanzadas (RAG, Fine-tuning, Quantization)

### 1. RAG (Retrieval-Augmented Generation)
  Descripción: Combina recuperación de información (búsqueda en bases de datos/vectoriales) con generación de texto para mejorar respuestas.
  
  Pasos prácticos:
  - Implementar un sistema RAG básico con:
  
  - Retriever: FAISS / Pinecone / Weaviate (búsqueda semántica).
  
  - Generator: LLM (GPT-3.5, LLaMA-2).
  
  - Usar LangChain o LlamaIndex para integrar componentes.
  
  - Desplegar un chatbot con RAG sobre documentos personalizados (PDFs, Wikipedia).

### 2. Fine-tuning Eficiente
  Descripción: Adaptar LLMs a dominios específicos con técnicas eficientes en recursos.
  
  - LoRA (Low-Rank Adaptation).
  
  - QLoRA (Quantized LoRA).
  
  - Prompt Tuning.
  
  Pasos prácticos:
  - Aplicar LoRA para fine-tuning de LLaMA-2 en un dataset custom.
  
  - Usar PEFT (Parameter-Efficient Fine-Tuning) de Hugging Face.
  
### 3. Quantization & Optimización
  Descripción: Reducir el tamaño de LLMs para ejecutarlos en hardware limitado.
  
  - GGUF/GGML (quantization para CPUs).
  
  - bitsandbytes (8-bit/4-bit quantization).
  
  Pasos prácticos:
  - Cuantizar un modelo LLaMA-2 con llama.cpp.
  
  - Ejecutar un modelo cuantizado en una GPU pequeña (ej: T4).
  
## Nivel 4: Aplicaciones y Producción
  Temas clave:
  - Despliegue de LLMs (FastAPI, Triton Inference Server).
  
  - Evaluación de LLMs (BLEU, ROUGE, human eval).
  
  - Ética y mitigación de riesgos (sesgos, alucinaciones).
  
  Pasos prácticos:
  - Desplegar un LLM como API con FastAPI + Hugging Face.
  
  - Evaluar respuestas generadas con métricas cualitativas.
  
  - Mitigar alucinaciones con RAG + verificaciones.
