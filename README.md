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




# Tokenización y Preprocesamiento de Texto (BPE, WordPiece)
  La tokenización es un paso fundamental en NLP que consiste en dividir el texto en unidades más pequeñas (tokens) para que los modelos puedan procesarlo.
  Los métodos modernos como BPE (Byte-Pair Encoding) y WordPiece son ampliamente usados en modelos como GPT, BERT y otros LLMs.
  
  1. ¿Qué es la Tokenización?
  Objetivo: Convertir texto en tokens (palabras, subpalabras o caracteres) para representarlo numéricamente.
  
  Desafíos:
  
  - Idiomas sin espacios (ej: chino).
  
  - Palabras raras o técnicas (ej: "anticonstitucionalmente").
  
  - Manejo de signos de puntuación, emojis, etc.
  
  Tipos de Tokenización
  - Método	Ejemplo ("unhappiness")	Tokens
  - Word-Level	Basado en palabras	["un", "happiness"]
  - Character-Level	Por caracteres	["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]
  - Subword-Level (BPE/WordPiece)	Balance entre palabras y caracteres	["un", "happiness"] o ["un", "happ", "iness"]
  
  2. Tokenización por Subpalabras: BPE y WordPiece
  Byte-Pair Encoding (BPE)
  - Usado en: GPT, RoBERTa, LLaMA.
  
  Funcionamiento:
  
  - Parte del vocabulario inicial como caracteres individuales.
  
  - Itera combinando los pares más frecuentes hasta alcanzar un tamaño de vocabulario fijo.
  
  Ejemplo:
  Texto: "low lower newest"
  
  Paso 1: l o w l o w e r n e w e s t
  
  Paso 2: Fusiona "lo" (par más frecuente) → lo w lo w e r n e w e s t
  
  Paso 3: Fusiona "low" → low low e r n e w e s t
  
  WordPiece
  - Usado en: BERT, DistilBERT.
  
  - Similar a BPE, pero elige fusiones basadas en probabilidad (no frecuencia).
  
  - Usa el criterio de maximizar la verosimilitud del lenguaje.
  
  Ejemplo:
  Si "happ" y "iness" aparecen juntas con frecuencia, se fusionan en "happiness".
  
  3. Preprocesamiento de Texto
  Antes de tokenizar, se aplican técnicas como:
  
  - Normalización:
  
  - Convertir a minúsculas.
  
  - Eliminar acentos ("café" → "cafe").
  
  - Expandir contracciones ("don't" → "do not").
  
  Filtrado:
  
  - Remover stopwords ("the", "and").
  
  - Eliminar URLs, hashtags, caracteres especiales.
  
  - Stemming/Lemmatización (opcional para LLMs): "running" → "run".
  
  4. Pasos Prácticos para Implementar Tokenización
  Opción 1: Usando Hugging Face Tokenizers
  *python*
  ```
  from transformers import AutoTokenizer
  
  # Cargar tokenizer (ej: BPE en GPT-2)
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  
  text = "¿Cómo tokenizar esto?"
  tokens = tokenizer.tokenize(text)  # ["¿", "C", "ómo", "token", "izar", "esto", "?"]
  ids = tokenizer.encode(text)      # [16968, 193, 4321, 1152, 534, 1026, 35]
  ```
  Opción 2: Implementar BPE desde cero
  *python*
  ```
  from tokenizers import Tokenizer, models, trainers
  
  tokenizer = Tokenizer(models.BPE())
  trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]"])
  
  # Entrenar con un corpus de texto
  corpus = ["lista de textos para entrenar..."]
  tokenizer.train_from_iterator(corpus, trainer)
  
  # Tokenizar
  output = tokenizer.encode("Texto de ejemplo")
  print(output.tokens)  # ["Texto", "de", "ejemplo"]
  ```
  Opción 3: Comparar BPE vs. WordPiece
  *python*
  ```
  from transformers import BertTokenizer, GPT2Tokenizer
  
  # WordPiece (BERT)
  bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  print(bert_tokenizer.tokenize("unhappiness"))  # ["un", "##happiness"]
  
  # BPE (GPT-2)
  gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  print(gpt2_tokenizer.tokenize("unhappiness"))  # ["un", "happiness"]
  ```

  5. Ejercicios Recomendados
  Tokenizar un dataset (ej: tweets) y comparar resultados con BPE/WordPiece.
  
  Entrenar un tokenizer BPE desde cero en un dominio específico (ej: código Python).
  
  Preprocesar texto para un LLM: normalizar, limpiar y tokenizar un corpus de Wikipedia.
  
  6. Recursos Adicionales
  Paper original de BPE
  
  Documentación de Hugging Face Tokenizers
  
  Tutorial de tokenización con código
  
  
