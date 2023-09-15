---
title: "Large Language Models and their Applications"
date: 2023-09-15T10:29:21+08:00
draft: false
ShowToc: true
category: [ai]
tags: ["llms", "ai"]
description: "Understanding the Large Language Models and their applications"
summary: "This post is an introduction of LLMs, what makes them such a hot topic in the world of AI today."
---

> :memo: **NOTE** This post is still in active drafting stage. I am working on adding diagrams, figures and code snippets which will be added soon.

# 1. Large Language Models

## Why LLMs?

1. **Introduction to LLMs (Large Language Models)**
    - LLMs are transforming industries involving human-computer interaction and language-related tasks.
    - Examples of LLM impact include businesses like Chegg and startups like you.com using them to enhance services.
    - Even existing tools like GitHub Copilot are becoming more powerful due to LLMs.
    - LLMs have evolved significantly in quality, accessibility, and data availability.
    
2. **Understanding LLMs and Language Modeling**
    - LLMs are statistical models predicting words in text based on large datasets.
    - They can learn about the world, including concepts like the color of avocados.
    - LLMs enable tasks such as generating text, classifying data, and more, based on language knowledge.
    - LLMs can automate tasks, democratize AI, open up rich use cases, and reduce development costs.
3. **Choosing the Right LLM for Your Application**
    - Consider factors like model quality, serving cost, serving latency, and customizability when selecting an LLM.
    - Customizability is important for continuous improvement, debugging, and control over the model's behavior.

## **Introduction to NLP**

1. **Introduction to Natural Language Processing (NLP):**
    - NLP is the study of natural language and how computational programs can solve language-related problems.
    - NLP is part of our daily lives, powering autocomplete, spell-check, and other language-related features.
    - LLMs are increasingly used in NLP to address various challenges.
2. **Key Concepts in NLP:**
    - Tokens are the fundamental building blocks of text in NLP.
    - Sequences are collections of tokens, representing either entire sentences or smaller fragments.
    - Vocabulary refers to the set of tokens available for modeling.
    - NLP tasks can be categorized as sequence-to-sequence, sequence-to-non-sequence, or open-ended problems.
3. **Scope of NLP:**
    - NLP extends beyond text and includes tasks like speech recognition, image captioning, and more.
    - The course focuses primarily on text-based NLP challenges, recognizing the complexities even within this domain.
    - Challenges in NLP include language ambiguity, contextual variability, and multiple valid responses.
4. **Model Size and Training Data:**
    - The course acknowledges the growth in model sizes and training data, which contribute to improved accuracy and performance.
    - Details about model sizes and training data are explored in later modules.
5. **Upcoming Language Models:**
    - The course sets the stage for discussing various types of language models in the following sections.

### **Language Models**

1. **Understanding Language Models:**
    - A language model is a computational model that takes a sequence of tokens and predicts the probability distribution over the vocabulary to find the most likely word.
    - Language models can be generative, predicting the next word in a sequence, or classification-based, predicting masked words in sentences.
2. **Generative Language Models:**
    - Generative language models aim to predict the likely next word in a given sequence.
    - The model internally calculates probabilities for all tokens in the vocabulary to find the best-fitting word to follow the sequence.
3. **Large Language Models (LLMs):**
    - LLMs are a category of language models that have grown significantly in size, from millions to billions of parameters.
    - Transformers, an architecture introduced in 2017, played a pivotal role in the rise of LLMs.
    - Prior language models had fewer parameters but required substantial computational effort.

### Tokenization

1. **Tokenization in NLP:**
    - Tokenization is the process of breaking text into individual units or tokens for computational analysis.
    - Tokens can be words, characters, or subwords, and the choice of tokenization impacts model performance and flexibility.
2. **Word Tokenization:**
    - Word tokenization involves creating a vocabulary of words from the training data and assigning each word a unique index.
    - It can lead to out-of-vocabulary errors if uncommon words are encountered during inference.
    - Misspellings and new words are challenging to handle with word-based tokenization.
    - Large vocabularies are required, which can be memory-intensive.
3. **Character Tokenization:**
    - Character tokenization represents each character as a token, resulting in a small vocabulary size.
    - It can accommodate new words and misspellings but loses the concept of words.
    - Character-based sequences can become very long, affecting computational efficiency.
4. **Subword Tokenization:**
    - Subword tokenization splits words into meaningful subword units, like prefixes and suffixes.
    - It offers a middle ground between word and character tokenization, balancing vocabulary size and flexibility.
    - Popular subword tokenization methods include Byte Pair Encoding (BPE), SentencePiece, and WordPiece.
    

**Summary:** Subword tokenization is a common choice in modern NLP because it strikes a balance between vocabulary size, adaptability, and maintaining word meaning.

### **Word Embeddings**

1. **Word Embeddings in NLP:**
    - Word embeddings aim to capture the context and meaning of words or tokens in a numerical format.
    - Similar words often occur in similar contexts, and embeddings help represent this similarity.
    - The context may include relationships with other words or intrinsic word meanings.
2. **Frequency-Based Vectorization:**
    - One way to represent words numerically is by counting their frequency in documents.
    - Each word is assigned an index, and a vector is built by counting word occurrences in a document.
    - This method can result in high sparsity and does not capture word meanings effectively.
3. **Word Embeddings through Context:**
    - Word embeddings capture word meanings by analyzing the context in which words appear.
    - Algorithms like Word2Vec consider words and their surrounding words, building vectors that represent contextual relationships.
    - These vectors typically have dimensions ranging from hundreds to thousands.
4. **Visualization of Word Embeddings:**
    - High-dimensional word embeddings can be projected onto 2D space for visualization.
    - Words with similar meanings tend to cluster together, indicating that their vectors have similarities.
    - While individual dimensions may not have specific meanings, the overall vector encodes word context and relationships.

**Summary:** Word embeddings are essential in NLP to represent words in a way that preserves their context and meaning, facilitating tasks such as text similarity, sentiment analysis, and more.

## Summary

- Natural Language Processing (NLP) is a field that focuses on understanding and processing natural language, encompassing various applications like text analysis, speech recognition, and more.
- NLP tasks include translation, summarization, classification, and more, often relying on language models to solve these problems.
- Large language models (LLMs) are based on the transformer architecture and have millions or billions of parameters, enabling them to handle complex NLP tasks.
- Tokens are the fundamental building blocks of language models, representing words or subword units in text.
- Tokenization converts text into numerical indices, and word embedding vectors help capture context and meaning for each token.

## References

1. Natural Language Processing
    - [Stanford Online Course on NLP](https://online.stanford.edu/courses/xcs224n-natural-language-processing-deep-learning)
    - [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
2. Language Modeling
    - [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
    - [Bag of Words](https://www.kaggle.com/code/vipulgandhi/bag-of-words-model-for-beginners)
    - [LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    - [Language Modeling](https://web.stanford.edu/~jurafsky/slp3/)
3. Word Embeddings
    - [Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)
    - [Tensorflow Page on Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
4. Tokenization
    - [Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
    - [SentencePiece](https://github.com/google/sentencepiece)
    - [WordPiece](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html)


# 2. Applications with LLM

## Hugging Face: Github of Large Language Models

- Hugging Face is introduced as a significant resource in the world of Large Language Models (LLMs) and NLP. It is often referred to as the "GitHub of Large Language Models" and encompasses both a company and an open-source community.
- Hugging Face's Hub serves as a repository for various resources, including models, datasets, demos, and code, which are available for download under different licenses.
- The Libraries provided by Hugging Face include:
    - **Datasets Library:** Allows users to download data from the Hub, making data access easy.
    - **Transformers Library:** Provides tools to work with core components of NLP, including pipelines, tokenizers, and models. It facilitates the download of pre-trained models from the Hub.
    - **Evaluate:** A library for model evaluation.
- Hugging Face's Transformers library is highly popular due to its simple APIs and underlying deep learning capabilities using libraries like PyTorch, TensorFlow, and Jax.
- The Transformers library makes it straightforward to create NLP pipelines. For example, it allows users to create a summarization pipeline to generate summaries for text articles using pre-trained LLMs.
- The pipeline involves several components, such as tokenization, encoding, decoding, and more. Tokenization converts text to numerical inputs for the model, while decoding converts the model's output back into text.
- The **AutoTokenizer** class automatically selects the appropriate tokenizer for a given model based on user input. It enables users to configure aspects like maximum length, padding, and truncation.
- Auto classes, such as **AutoModelForSeq2SeqLM**, assist in selecting the right model class for specific tasks. These classes handle tasks like transforming variable-length input sequences into variable-length output sequences.
- Inference parameters, such as *num_beams*, *min_length*, and *max_length*, allow users to fine-tune the model's behavior during text generation.
- The Hugging Face Hub hosts datasets, making them easily accessible through the load_dataset API, which also supports version specification. The Hub allows users to filter datasets by various attributes and discover related models.

**Summary:** Overall, Hugging Face simplifies working with LLMs and offers a wide range of resources for NLP tasks.

## Model Selection

- Selecting the right model for a specific NLP task involves considering various factors and requirements. Details within broader tasks like summarization, translation, etc., need to be addressed. For instance, summarization can be either extractive or abstractive.
- The abundance of available models, such as the 176,000+ on Hugging Face Hub, makes it essential to filter and search efficiently. Several easy choices for filtering include task, license, language, and model size, with parameters like max_length, padding, and truncation to adapt models for specific requirements.
- Filtering by task, license, language, and model size are straightforward ways to narrow down model options. License requirements and hardware constraints can drive these choices.
- Sorting models by popularity and updates is beneficial for selecting reliable models, as popular models are often tried and tested by the community. Regular updates ensure compatibility with the latest libraries.
- Model variants, fine-tuned models, and choosing the right size (e.g., small models for quick prototyping) can aid in selecting the right model for the task.
- It's valuable to explore examples and datasets related to the task, as they often provide practical usage insights and guidance, especially when models lack detailed documentation.
- Understanding whether a model is a generalist or fine-tuned for specific tasks, along with knowledge of the datasets used for pre-training and fine-tuning, can influence model selection.
- Evaluating models using defined key performance indicators (KPIs) and metrics on your specific data and user needs is crucial.
- Some of famous LLMs and model families: Many of these models are part of larger families and can vary in size and specialization. Model architecture, pre-training datasets, and fine-tuning contribute to model differences.
    1. **GPT (Generative Pre-trained Transformer):**
        - **GPT-3.5:** Part of the GPT-3 model family, known for its impressive language generation capabilities.
        - **GPT-4:** A continuation of the GPT series, expected to be a larger and more powerful language model.
    2. **Pythia:**
        - Pythia is a family of models, often used as base models for further fine-tuning.
        - **Dolly:** A fine-tuned version of one of the Pythia models, specifically designed for instruction following tasks.
- Recognizing well-known models and families is essential for efficient model selection.

**Summary:** Overall, selecting the right model requires a combination of understanding the task's nuances, applying practical filtering techniques, and assessing the suitability of different models for specific requirements.

## **NLP Tasks**

Here we discuss various common Natural Language Processing (NLP) tasks. 

1. **Sentiment Analysis:**
    - Determining the sentiment (positive, negative, or neutral) of a given text.
    - Useful for applications like analyzing Twitter commentary for stock market trends.
    - LLMs can provide confidence scores along with sentiment labels.
2. **Translation:**
    - Translating text from one language to another.
    - Some LLMs are fine-tuned for specific language pairs, while others are more general.
    - Models like t5_translator can perform translations based on user instructions.
3. **Zero-Shot Classification:**
    - Categorizing text into predefined labels without retraining the model.
    - Useful for tasks like categorizing news articles into topics like sports, politics, etc.
    - LLMs leverage their language understanding to classify text based on provided labels.
4. **Few-Shot Learning:**
    - A flexible technique where the model learns from a few examples provided by the user.
    - Allows LLMs to perform various tasks without explicit fine-tuning.
    - Requires crafting instructions and examples to guide the model's behavior.
    

**Summary:** LLMs can be highly versatile and adaptable to different tasks, even when specific fine-tuned models are not available. 

## **Prompts**

Here we discuss the concept of prompts and their importance in interacting with Large Language Models (LLMs). 

1. **Instruction-Following LLMs vs. Foundation Models:**
    - Instruction-following LLMs are tuned to follow specific instructions or prompts.
    - Foundation models are pre-trained on general text generation tasks like predicting the next token.
2. **Types of Prompts:**
    - Prompts are inputs or queries used to elicit responses from LLMs.
    - They can be natural language sentences, questions, code, combinations, emojis, or any form of text.
    - Prompts can include outputs from other LLM queries, allowing for complex and dynamic interactions.
3. **Examples of Prompts:**
    - In the context of summarization, a prompt might consist of a prefix like "summarize:" followed by the input text.
    - For few-shot learning, prompts include instructions, examples to teach the model, and the actual query.
    - Prompts can become complex, such as structured output extraction examples, which provide high-level instructions, output format specifications, and more.
4. **Power of Prompt Engineering:**
    - Prompt engineering is the process of crafting prompts to achieve specific desired behaviors from LLMs.
    - Well-designed prompts can leverage the capabilities of LLMs effectively and enable them to perform a wide range of tasks.
    

### **Prompt Engineering**

Here we discuss the concept of prompt engineering and its importance in guiding Large Language Models (LLMs) to produce desired outputs.

1. **Model-Specific Prompt Engineering:**
    - Prompt engineering is specific to the LLM being used, as different models may require different prompts.
    - Iterative development is crucial, involving testing different prompt variations to find what works best for a given model and task.
2. **Clarity and Specificity in Prompts:**
    - Effective prompts are clear and specific, consisting of an instruction, context, input or question, and output type or format.
    - Clear task descriptions, including specific keywords or detailed instructions, enhance prompt effectiveness.
3. **Testing and Data-Driven Approach:**
    - Prompt engineering involves testing various prompt variations on different samples to determine which prompts yield better results on average for a specific set of inputs.
4. **Enhancing Model Behavior with Prompts:**
    - Prompts can be used to instruct the model not to generate false or nonsensical information (hallucinations).
    - Models can be guided to avoid making assumptions or probing for sensitive information.
    - Chain of thought reasoning prompts encourage the model to think through problems step by step, often resulting in improved responses.
5. **Prompt Formatting:**
    - Proper delimiters should be used to distinguish between the instruction, context, and user input in a prompt.
    - Prompts can request structured output and provide correct examples.
    - Careful prompt formatting can help prevent prompt injection and other security concerns.
6. **Security and Prompt Hacking:**
    - Prompt hacking involves exploiting vulnerabilities in LLMs by manipulating inputs to override instructions, extract sensitive information, or bypass rules.
    - Techniques to mitigate prompt hacking include post-processing, filtering, repeating instructions, enclosing user input with random strings or tags, and selecting different models or restricting prompt length.
7. **Resources and Tools:**
    - Several guides and tools are available to assist in writing effective prompts, including both OpenAI-specific and general resources.

**Summary:** Prompt engineering plays a crucial role in maximizing the utility of LLMs and ensuring they produce desired and safe outputs for various applications. It requires careful consideration of the model's behavior and iterative refinement of prompts to achieve optimal results.

## References

1. **NLP tasks**
    - [Hugging Face tasks page](https://huggingface.co/tasks)
    - [Hugging Face NLP course chapter 7: Main NLP Tasks](https://huggingface.co/course/chapter7/1?fw=pt)
    - Background reading on specific tasks
        - Summarization: [Hugging Face summarization task page](https://huggingface.co/tasks/summarization) and [course section](https://huggingface.co/learn/nlp-course/chapter7/5)
        - Sentiment Analysis: [Blog on “Getting Started with Sentiment Analysis using Python”](https://huggingface.co/blog/sentiment-analysis-python)
        - Translation: [Hugging Face translation task page](https://huggingface.co/docs/transformers/tasks/translation) and [course section](https://huggingface.co/learn/nlp-course/chapter7/4)
        - Zero-shot classification: [Hugging Face zero-shot classification task page](https://huggingface.co/tasks/zero-shot-classification)
        - Few-shot learning: [Blog on “Few-shot learning in practice: GPT-Neo and the 🤗 Accelerated Inference API”](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api)
2. **[Hugging Face Hub](https://huggingface.co/docs/hub/index)**
    - [Models](https://huggingface.co/models)
    - [Datasets](https://huggingface.co/datasets)
    - [Spaces](https://huggingface.co/spaces)
3. **Hugging Face libraries**
    - [Transformers](https://huggingface.co/docs/transformers/index)
        - Blog post on inference configuration: [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
    - [Datasets](https://huggingface.co/docs/datasets)
    - [Evaluate](https://huggingface.co/docs/evaluate/index)
4. **Models**
    - Base model versions of models used in the demo notebook
        - [T5](https://huggingface.co/docs/transformers/model_doc/t5)
        - [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
        - [Marian NMT framework](https://huggingface.co/docs/transformers/model_doc/marian) (with 1440 language translation models!)
        - [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta) (Also see [DeBERTa-v2](https://huggingface.co/docs/transformers/model_doc/deberta-v2))
        - [GPT-Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo) (Also see [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox))
    - [Table of LLMs](https://crfm.stanford.edu/ecosystem-graphs/index.html)
5. **Prompt engineering**
    - [Best practices for OpenAI-specific models](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
    - [DAIR.AI guide](https://www.promptingguide.ai/)
    - [ChatGPT Prompt Engineering Course](https://learn.deeplearning.ai/chatgpt-prompt-eng) by OpenAI and DeepLearning.AI
    - [🧠 Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) for fun examples with ChatGPT


# 3. Embeddings, Vector Databases, and Search

## Introduction

"Embeddings, Vector Databases, and Search" plays a pivotal role in enhancing the capabilities of Large Language Models (LLMs) for knowledge-based question answering. 

1. **Knowledge-Based Question Answering (QA):**
    - Knowledge-based QA involves using LLMs to access and utilize information from a knowledge base (documents, data sets, etc.) to answer questions and perform tasks specific to a particular domain or area.
    - It is a common and valuable application, especially for organizations with custom knowledge or data sets.
2. **Advantages of Knowledge-Based QA:**
    - Knowledge-based QA enhances productivity by leveraging internal or domain-specific knowledge.
    - It has applications in internal processes, customer support, and knowledge updates.
    - By referencing documents and sources, it provides a more reliable and verifiable source of information.
3. **Role of Embeddings, Vector Databases, and Search:**
    - Embeddings are mathematical representations of documents and questions that enable efficient comparison and retrieval of related information.
    - Vector databases and search technologies facilitate the search for relevant documents based on input and tasks.
    - These tools help LLMs locate and utilize the appropriate knowledge for a given task, improving the accuracy of responses.
4. **Training Vector Databases:**
    - Vector databases can be trained to perform effective searches, including relevant documents and sources.
    - Classical keyword-based search techniques can also be used and optimized for specific applications.
5. **Workflow for Knowledge-Based QA:**
    - The workflow involves using an LLM to understand and process information from a knowledge base.
    - Embeddings and vector databases are employed to search for relevant knowledge based on input or questions.
    - The retrieved knowledge, along with the original question, is fed into the LLM to generate accurate responses.
6. **Importance of Custom Domains:**
    - Knowledge-based QA is especially valuable in custom domains where organizations have unique data and knowledge that can be utilized effectively.
7. **Fast-Evolving Field:**
    - The field of knowledge-based QA and related technologies is continuously evolving, with ongoing developments and improvements.

### Key Concepts

Here we focus on understanding how to leverage embeddings, vector databases, and search techniques to build effective question-answering (Q&A) systems. We will gain insights into various vector search strategies, evaluation methods for search results, and when to utilize vector databases, libraries, or plugins. 

1. **Knowledge Acquisition:**
    - Language models can acquire knowledge through two primary methods: training/fine-tuning and passing context or knowledge as inputs.
    - Training and fine-tuning involve updating model weights to learn knowledge during the training process. ( Suited to teach a model specialized tasks)
    - Passing context to the model is a newer approach and is closely tied to prompt engineering, where context is provided to the model through prompts. Passing context as model inputs improves factual recall.
    - Downsides due to context limitations**:**
        - Context length limitations exist for models like OpenAI's GPT-3.5, which allows up to 4000 tokens (approximately 5 pages of text) per input.
        - Longer contexts require more API calls, leading to increased costs and processing time.
        - Increasing context length alone may not help models retain information effectively.
2. **Role of Embeddings and Vector Databases:**
    - Embeddings are essential for enabling similarity searches in question-answering systems.
    - Vector databases have gained popularity and are not limited to text data; they are also used for unstructured data like images and audio.
    - Embeddings help convert data into vectors, which are then stored in vector databases for efficient retrieval.
3. **Vector Database Use Cases:**
    - Vector databases find applications beyond text data, including recommendation engines, anomaly detection, and security threat identification.
    - Spotify, for example, uses vector databases for podcast episode recommendations based on user queries.
4. **Question-Answering System Workflow:**
    - Knowledge-based Q&A systems consist of two main components: search and retrieval.
    - The knowledge base contains documents converted into embedding vectors and stored in a vector index (vector database or library).
    - User queries are converted into embedding vectors by a language model.
    - The search component involves searching the vector index to find relevant documents.
    - Retrieved documents are passed as context to the language model in a prompt.
    - The language model generates responses that incorporate the retrieved context.
    - This workflow is known as a search and retrieval-augmented generation workflow.
        
        ![Search and Retrieval Augmented Generation Workflow. Note: create my own diagram before publishing](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.01.28_PM.png)
        
        Search and Retrieval Augmented Generation Workflow. Note: create my own diagram before publishing
        
    

**Summary:**  Embeddings, vector databases, and search techniques are important concepts and techniques enabling LLMs to effectively access and utilize knowledge from various sources, making knowledge-based question answering a practical and powerful application in various domains. Here  the critical components and strategies for implementing knowledge-based question-answering systems, emphasizing the role of embeddings, vector databases, and search techniques in optimizing information retrieval and generation processes are discussed.

## **How does Vector Search work?**

Here we explore the concept of vector search, which is fundamental to building effective question-answering systems. 

1. **Exact and Approximate Search:**
    - Vector search involves two main strategies: exact search and approximate search.
    - Exact search aims to find the nearest neighbors precisely, while approximate search sacrifices some accuracy for speed.
    - Examples:
        - Exact Search: K-nearest neighbours (KNN)
        - Approximate nearest neighbours (ANN)
            - Tree-based: ANNOY by Spotify
            - Proximity graphs: HNSW
            - Clustering: FAISS by Facebook
            - Hashing: LSH
            - Vector Compression: ScaNN by Google
2. **Indexing Algorithms:**
    - Various indexing algorithms play a crucial role in vector search.
    - These algorithms convert data into vectors and generate a data structure called a vector index.
    - Indexing methods include tree-based approaches, clustering, and hashing.
3. **Distance and Similarity Metrics:**
    - Similarity between vectors is determined using distance or similarity metrics.
    - Common distance metrics include L1 (Manhattan) and L2 (Euclidean) distances, where higher values indicate less similarity.
    - Cosine similarity measures the angle between vectors, with higher values indicating more similarity.
    - L2 distance and cosine similarity produce functionally equivalent ranking distances for normalized embeddings.
        
        ![Screenshot 2023-09-09 at 9.06.18 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.06.18_PM.png)
        
4. **Vector Compression with Product Quantization (PQ):**
    - To reduce memory usage, dense embedding vectors can be compressed using PQ.
    - PQ quantizes subvectors independently and maps them to centroids, reducing the storage space required.
        
        ![Screenshot 2023-09-09 at 9.07.27 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.07.27_PM.png)
        
5. **FAISS (Facebook AI Similarity Search):**
    - FAISS is a clustering algorithm that computes L2 Euclidean distances between query vectors and stored points.
    - It optimizes search by using Voronoi cells to narrow down the search space based on centroids.
    - Well-suited for dense vectors but not sparse ones.
        
        ![Screenshot 2023-09-09 at 9.08.30 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.08.30_PM.png)
        
6. **HNSW (Hierarchical Navigable Small Worlds):**
    - HNSW uses Euclidean distance as a metric and is based on a proximity graph approach.
    - It employs a linked list or skip list structure to find nearest neighbors efficiently.
    - Hierarchy is introduced to reduce the number of layers and enhance search performance.
        
        ![Screenshot 2023-09-09 at 9.09.17 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.09.17_PM.png)
        
7. **Vector Search Impact:**
    - Vector search greatly expands the possibilities of use cases by enabling similarity-based searches.
    - It allows for more flexible querying compared to exact matching rules and SQL filter statements.

Summary: This section provides insights into the techniques and algorithms used in vector search, highlighting their importance in optimizing knowledge-based question-answering systems.

### **Filtering**

Here we delve into filtering strategies used in vector databases, emphasizing their challenges and nuances. 

1. **Filtering Categories:**
    - Filtering in vector databases falls into three main categories: post-query, in-query, and pre-query.
    - Some vector databases may implement their proprietary filtering algorithms, rooted in one of these categories.
2. **Post-Query Filtering:**
    - Post-query filtering involves applying filters after identifying the top-K nearest neighbors in a search.
    - It leverages the speed of approximate nearest neighbor (ANN) search but may result in unpredictable or empty results if no data meets the filter criteria.
        
        ![Screenshot 2023-09-09 at 9.13.51 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.13.51_PM.png)
        
3. **In-Query Filtering:**
    - In-query filtering combines ANN and filtering during the search process.
    - It computes both vector similarity and metadata information simultaneously, demanding higher system memory due to loading both vector and scalar data.
    - Performance may degrade as more filters are applied, potentially causing out-of-memory issues.
        
        ![Screenshot 2023-09-09 at 9.14.06 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.14.06_PM.png)
        
4. **Pre-Query Filtering:**
    - Pre-query filtering restricts the scope of similarity search based on applied filters.
    - It does not leverage the speed of ANN, requiring brute-force filtering of all data within the specified scope.
    - Typically less performant compared to post-query and in-query methods due to the absence of ANN speed.
        
        ![Screenshot 2023-09-09 at 9.15.11 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.15.11_PM.png)
        
5. **Considerations for Filter Implementation:**
    - Deciding when and how to implement filters depends on factors such as system memory, query complexity, and performance requirements.
    - Effective filtering strategies can enhance the precision and relevance of search results in vector databases.

**Summary:** This section highlights the challenges and trade-offs associated with different filtering strategies in vector databases, providing insights into their practical application.

## Vector Stores

Here we focus on vector stores and their practical aspects, including vector databases, vector libraries, and plugins.

1. **Vector Stores Overview:**
    - Vector stores encompass vector databases, libraries, and plugins, serving as specialized solutions for storing unstructured data as vectors.
    - Vector databases offer advanced search capabilities as a primary differentiator, providing search as a service.
        
        ![Screenshot 2023-09-09 at 9.22.15 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.22.15_PM.png)
        
2. **Vector Libraries:**
    - Vector libraries create vector indexes for efficient vector search.
    - They are suitable for small and static datasets and typically lack database properties such as CRUD support, data replication, and on-disk storage.
    - Changes to data may require rebuilding the vector index.
3. **Vector Plugins:**
    - Existing relational databases or search systems may offer vector search plugins.
    - These plugins may provide fewer metrics or ANN options but can be expected to evolve and expand in functionality.
        
        ![Screenshot 2023-09-09 at 9.24.07 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.24.07_PM.png)
        
4. **Choosing Between Vector Database and Library:**
    - Consider factors like the volume of data, query speed requirements, and the need for full-fledged database properties.
    - Vector databases are more suitable for large, dynamic datasets and offer comprehensive database features.
    - Vector libraries can suffice for smaller, static datasets and simpler use cases.
5. **Performance Considerations:**
    - The decision to use a vector database should be based on data volume, query speed requirements, and database properties needs.
    - For largely static data, offline computation of embeddings followed by storage in a vector database can be cost-effective.
    - Vector databases come with additional costs and require learning, integration, and maintenance.
        
        ![Screenshot 2023-09-09 at 9.25.20 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.25.20_PM.png)
        
6. **Vector Database Comparisons:**
    - Starter comparisons of popular vector database choices are provided, with the understanding that the landscape may evolve over time.
        
        ![Screenshot 2023-09-09 at 9.25.32 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.25.32_PM.png)
        

**Summary:** This section highlights the considerations and trade-offs involved in selecting and using vector stores, aiding in the practical implementation of vector-based solutions.

## **Best Practices**

Here we discuss several best practices for using vector stores and implementing search-retrieval systems are discussed. 

1. **Consider the Need for a Vector Store:**
    - The decision to use a vector store, such as a vector database, library, or plugin, depends on whether context augmentation is needed for your specific use case.
    - Use cases like summarization, text classification (e.g., sentiment analysis), and translation may not require context augmentation and can function effectively without vector stores.
        
        ![Screenshot 2023-09-09 at 9.29.19 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.29.19_PM.png)
        
2. **Improving Retrieval Performance:**
    
    ![Screenshot 2023-09-09 at 9.30.04 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.30.04_PM.png)
    
    - Select your embedding model wisely, ensuring it is trained on similar data as yours for optimal results.
        
        ![Screenshot 2023-09-09 at 9.30.21 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.30.21_PM.png)
        
    - Ensure that your embedding space encompasses all relevant data, including user queries, for accurate retrieval.
        
        ![Screenshot 2023-09-09 at 9.31.05 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.31.05_PM.png)
        
    - Use consistent models for both document indexing and query embedding to maintain a shared embedding space.
3. **Document Storage Strategy:**
    - Consider whether to store documents as a whole or in chunks, as chunking strategies can significantly impact retrieval performance.
    - Chunking strategies depend on factors like document length, user query behavior, and the alignment of query embeddings with document chunks.
        
        ![Screenshot 2023-09-09 at 9.33.11 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.33.11_PM.png)
        
    - Experiment with different chunking approaches to optimize your system.
        
        ![Screenshot 2023-09-09 at 9.31.46 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.31.46_PM.png)
        
4. **Guard Rails for Performance and Safety:**
    - Include explicit instructions in prompts to guide the model's behavior and prevent it from generating unreliable or fictitious responses.
    - Implement failover logic to handle scenarios where vector search does not return results meeting a specified threshold of similarity.
    - Add a toxicity classification model to prevent offensive or inappropriate inputs from affecting the system's responses.
    - Configure your vector database to set timeouts for queries that take too long, indicating the absence of similar vectors.
        
        ![Screenshot 2023-09-09 at 9.33.50 PM.png](Embeddings,%20Vector%20Databases,%20and%20Search%20f7aadf9550e241bbac2bab5173888953/Screenshot_2023-09-09_at_9.33.50_PM.png)
        

These best practices provide guidance for effectively utilizing vector stores and ensuring robust retrieval performance in search-retrieval systems.

## References

1. **Research papers on increasing context length limitation**
    - [Pope et al 2022](https://arxiv.org/abs/2211.05102)
    - [Fu et al 2023](https://arxiv.org/abs/2212.14052)
2. **Industry examples on using vector databases**
    - FarFetch
        - [FarFetch: Powering AI With Vector Databases: A Benchmark - Part I](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-i/)
        - [FarFetch: Powering AI with Vector Databases: A Benchmark - Part 2](https://www.farfetchtechblog.com/en/blog/post/powering-ai-with-vector-databases-a-benchmark-part-ii/)
        - [FarFetch: Multimodal Search and Browsing in the FARFETCH Product Catalogue - A primer for conversational search](https://www.farfetchtechblog.com/en/blog/post/multimodal-search-and-browsing-in-the-farfetch-product-catalogue-a-primer-for-conversational-search/)
    - [Spotify: Introducing Natural Language Search for Podcast Episodes](https://engineering.atspotify.com/2022/03/introducing-natural-language-search-for-podcast-episodes/)
    - [Vector Database Use Cases compiled by Qdrant](https://qdrant.tech/use-cases/)
3. **Vector indexing strategies**
    - Hierarchical Navigable Small Worlds (HNSW)
        - [Malkov and Yashunin 2018](https://arxiv.org/abs/1603.09320)
    - Facebook AI Similarity Search (FAISS)
        - [Meta AI Blog](https://ai.facebook.com/tools/faiss/)
    - Product quantization
        - [PQ for Similarity Search by Peggy Chang](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd)
4. **Cosine similarity and L2 Euclidean distance**
    - [Cosine and L2 are functionally the same when applied on normalized embeddings](https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance)
5. **Filtering methods**
    - [Filtering: The Missing WHERE Clause in Vector Search by Pinecone](https://www.pinecone.io/learn/vector-search-filtering/)
6. **Chunking strategies**
    - [Chunking Strategies for LLM applications by Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
    - [Semantic Search with Multi-Vector Indexing by Vespa](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/)
7. **Other general reading**
    - [Vector Library vs Vector Database by Weaviate](https://weaviate.io/blog/vector-library-vs-vector-database)
    - [Not All Vector Databases Are Made Equal by Dmitry Kan](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696)
    - [Open Source Vector Database Comparison by Zilliz](https://zilliz.com/comparison)
    - [Do you actually need a vector database? by Ethan Rosenthal](https://www.ethanrosenthal.com/2023/04/10/nn-vs-ann/)


# 4. Multi-stage Reasoning

## **Introduction**

In this section on multi-stage reasoning, the focus is on building more sophisticated applications with Large Language Models (LLMs) by chaining prompts and leveraging external tools. 

1. **Enhancing Prompting Techniques:**
    - Exploring methods to create prompts that guide the model to reason and produce better answers.
    - Introducing the concept of prompt templates as a tool to influence the model's responses.
    - Discussing techniques like Chain of Thought that leverage the model's statistical capabilities to achieve desired outputs.
2. **Chaining Prompts for Complex Tasks:**
    - Emphasizing the use of chaining prompts to break down complex tasks into smaller, manageable subtasks.
    - Highlighting the advantages of dividing a problem into multiple stages, each handled by an LLM specialized in a specific subtask.
    - Introducing open-source frameworks like LangChain that facilitate the modularization of LLM applications for improved maintainability and quality.
3. **LLM Agents and External Tools:**
    - Exploring the concept of LLM Agents, which enable LLMs to use external tools and APIs to perform tasks beyond their inherent capabilities.
    - Discussing how LLMs can generate text that serves as API calls to external tools, expanding the range of tasks they can accomplish.
4. **Autonomous LLM Systems:**
    - Touching on autonomous LLM systems that can discover and utilize external tools and methods to perform tasks efficiently.
    - Highlighting the potential for LLMs to learn and improve their performance autonomously through data-driven feedback.
5. **Demonstrate Search and Predict (DSP) Project:**
    - Mentioning the DSP project at Stanford, which focuses on building reliable LLM pipelines capable of continuous improvement using data.
    - Emphasizing the importance of multi-stage reasoning in LLM applications and its potential to integrate with various software systems.

### Key Concepts

Here we  focus is on combining Large Language Models (LLMs) with vector databases to enhance applications.

1. **Integration of LLMs and Vector Databases:**
    - Highlighting the need to leverage both LLMs and vector databases to create more powerful applications.
    - Introducing tools and techniques for seamlessly integrating LLMs and databases.
2. **Learning Objectives:**
    - Outlining the module's learning objectives, which include understanding the flow of LLM pipelines, using tools like LangChain for building pipelines with LLMs from various providers, and creating complex logical flow patterns using agents that utilize LLMs and other tools.
3. **Limitations of LLMs:**
    - Acknowledging the strengths of LLMs in solving traditional NLP tasks like summarization, translation, and zero-shot classification.
    - Recognizing that real-world applications often involve more complex workflows beyond simple input and output responses, requiring the integration of LLMs with other code components.
        
        ![Screenshot 2023-09-09 at 10.27.54 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.27.54_PM.png)
        
4. **Modularization and Flexibility:**
    - Emphasizing the importance of modularizing LLM-based workflows to ensure flexibility and maintainability.
    - Highlighting the goal of building tools and systems that allow for the easy replacement of one LLM with another without breaking the entire application.
        
        ![Screenshot 2023-09-09 at 10.28.09 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.28.09_PM.png)
        
5. **Example Use Case: Summarization and Sentiment Analysis:**
    - Illustrating a practical example involving the summarization and sentiment analysis of articles.
    - Demonstrating the challenges of using a single LLM for both tasks, including the need for large models and the constraint of input sequence length.
    - Introducing the concept of breaking down the workflow into smaller, reusable components.
        
        ![Screenshot 2023-09-09 at 10.28.35 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.28.35_PM.png)
        
        ![Screenshot 2023-09-09 at 10.29.38 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.29.38_PM.png)
        

**Summary:** This module explores the exciting capabilities of multi-stage reasoning with LLMs, enabling the development of complex applications and the utilization of external tools to perform a wide range of tasks. We provide a comprehensive understanding of how to effectively combine LLMs and vector databases to create versatile and modular applications that can handle complex tasks with ease.

## **Prompt Engineering**

In this section, the focus is on prompt engineering and creating well-structured prompts for large language models. 

1. **Importance of Well-Written Prompts:**
    - Emphasizing that a well-structured prompt can significantly impact the quality of responses from large language models.
    - Highlighting the need for systematic approaches to prompt creation that can be shared and modularized across teams.
2. **Example Use Case: Summarization:**
    - Explaining the use case involving the summarization of articles one by one.
    - Introducing the idea of creating prompt templates step by step to achieve the desired results.
3. **Prompt Template Creation:**
    - Demonstrating the process of building a summary prompt template for summarizing articles.
    - Describing the elements of a prompt template, including the task description, variable definition (e.g., using curly braces), and specifying the expected output.
4. **Prompt Template Usage:**
    - Showing how to create a prompt template instance by defining input variables.
    - Illustrating the use of a specific article as input to the summary prompt template.
        
        ![Screenshot 2023-09-09 at 10.32.51 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.32.51_PM.png)
        
5. **Chaining Prompts:**
    - Explaining the concept of chaining one large language model to another, where the output of one model becomes the input to another.
    - Recognizing that creating prompt templates was the first part of solving a two-stage problem, and now the focus is on chaining large language models together.
        
        ![Screenshot 2023-09-09 at 10.33.14 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.33.14_PM.png)
        

**Summary:** The section provides insights into the systematic construction of prompts using templates and their use in chaining large language models to perform multi-stage reasoning tasks.

## **LLM Chains**

In this section on LLM Chains, several important points are covered:

1. **Introduction to LLM Chains:**
    - LLM Chains involve linking large language models not only with other LLMs but also with various tools and libraries.
    - The concept gained popularity with the release of the LangChain library in late 2022, enabling the creation of versatile applications and workflows.
        
        ![Screenshot 2023-09-09 at 10.35.06 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.35.06_PM.png)
        
2. **Chaining Large Language Models:**
    - The section uses a previous example of summarizing articles and applies a similar approach to sentiment analysis.
    - Emphasis is placed on chaining the output of one large language model as the input to another, forming a structured workflow.
        
        ![Screenshot 2023-09-09 at 10.35.18 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.35.18_PM.png)
        
3. **Components of LLM Chains:**
    - LLM Chains consist of a workflow chain that connects all the components together.
    - Within the workflow chain, there are smaller chains, such as the summary chain and the sentiment chain, each focusing on specific tasks.
    - This structured chaining allows for the modularization and organization of tasks within a workflow.
        
        ![Screenshot 2023-09-09 at 10.36.03 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.36.03_PM.png)
        
4. **Expanding Beyond LLM Chains:**
    - LLM Chains can extend beyond linking LLMs; they can connect with mathematical suites, programming tools, search libraries, and more.
    - The process involves taking natural language input, generating code, passing it to interpreters or APIs, receiving results, and combining them with the input to produce coherent responses.
        
        ![Screenshot 2023-09-09 at 10.36.39 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.36.39_PM.png)
        
5. **Versatility of LLMs:**
    - LLMs can be used as central reasoning tools, granting access to a wide range of programmatic resources, including search engines, email clients, and external APIs.
    - Structured prompts can guide LLMs to interact with different tools, making them adaptable to various tasks.
        
        ![Screenshot 2023-09-09 at 10.38.28 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.38.28_PM.png)
        
6. **Leveraging LLM Agents:**
    - LLM Agents are introduced as entities that can autonomously decide which tools or methods to use to fulfill a task.
    - The section hints at exploring the decision-making capabilities of LLM Agents in the subsequent section.

The section provides insights into the capabilities of LLM Chains and highlights the potential for building complex workflows and applications by connecting LLMs and programmatic tools in a structured manner.

## [Agents](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@233ea1e550a341b48ee7fe43b4881a57)

In the section on LLM Agents, several key points are highlighted:

1. **Introduction to LLM Agents:**
    - LLM Agents use large language models as centralized reasoning units, coupled with various tools and components to solve complex tasks automatically.
    - These agents are built upon the reasoning loops that large language models excel at.
2. **Reasoning Loops:**
    - LLMs can provide a step-by-step plan or thought process for a given task.
    - LLM Agents employ a thought-action-observation loop, where the LLM performs actions, observes results, and decides whether to continue or return based on predefined criteria.
        
        ![Screenshot 2023-09-09 at 10.40.48 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.40.48_PM.png)
        
3. **Components of LLM Agents:**
    - To create an LLM Agent, you need a specific task to be solved, an LLM capable of Chain of Thought reasoning, and a set of tools that can interface with the LLM.
    - Tool descriptions help the LLM determine which tool to use and how to interact with it.
        
        ![Screenshot 2023-09-09 at 10.41.41 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.41.41_PM.png)
        
4. **Development of LLM Agents:**
    - LLM Agents and plugins are emerging in the open source community, with LangChain and Hugging Face's Transformers Agents as examples.
    - Companies like Google are also integrating LLM Agents into their products.
        
        ![Screenshot 2023-09-09 at 10.42.12 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.42.12_PM.png)
        
        ![Screenshot 2023-09-09 at 10.42.43 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.42.43_PM.png)
        
5. **Automated LLMs:**
    - AutoGPT, using GPT-4, demonstrates the ability to create copies of itself and delegate tasks to these copies, enabling the solution of complex tasks with minimal prompting.
        
        ![Screenshot 2023-09-09 at 10.42.55 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.42.55_PM.png)
        
6. **Diverse Landscape of Multi-Stage Reasoning Tools:**
    - The landscape of multi-stage reasoning tools includes both proprietary and open-source solutions.
    - There are guided tools like LangChain and Hugging Face Transformers Agents and unguided ones like HuggingGPT, BabyAGI, and AutoGPT.
    - The community continuously develops and updates these tools, expanding their capabilities.
        
        ![Screenshot 2023-09-09 at 10.43.18 PM.png](Multi-stage%20Reasoning%20c69c5771bab3476089b5201d0fd8b9f9/Screenshot_2023-09-09_at_10.43.18_PM.png)
        
7. **Exciting Future Prospects:**
    - The section concludes by expressing enthusiasm for the growing landscape of LLM Agents and the possibilities they offer in automating complex tasks.

**Summary:**  LLM Agents represent a powerful paradigm for automating tasks by combining the reasoning capabilities of LLMs with various tools and components, and this field is rapidly evolving with the involvement of both proprietary and open-source communities.

# References

1. **LLM Chains**
    - [LangChain](https://docs.langchain.com/)
    - [OpenAI ChatGPT Plugins](https://platform.openai.com/docs/plugins/introduction)
2. **LLM Agents**
    - [Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents)
    - [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
    - [Baby AGI](https://github.com/yoheinakajima/babyagi)
    - [Dust.tt](https://dust.tt/)
3. **Multi-stage Reasoning in LLMs**
    - [CoT Paradigms](https://matt-rickard.com/chain-of-thought-in-llms)
    - [ReAct Paper](https://react-lm.github.io/)
    - [Demonstrate-Search-Predict Framework](https://github.com/stanfordnlp/dsp)

# 5. Fine-tuning and Evaluating LLMs

## Introduction

In Module Four, the focus is on enhancing the quality of Large Language Model (LLM) applications, particularly through fine-tuning and evaluating LLMs. The following are the key points covered in this module:

1. **Methods to Improve LLM Quality:**
    - This module addresses the ways to enhance LLM applications when you have data and feedback.
    - Various approaches are discussed, ranging from few-shot learning, where you provide a few task-specific examples to the model, to fine-tuning, where you update the LLM's parameters for a specific application.
2. **Three General Approaches:**
    - The module outlines three general approaches for improving LLM quality:
        - Few-shot learning with an existing frozen model.
        - Leveraging LLMs as a service, which allows you to use pre-trained models for specific tasks.
        - Do-it-yourself fine-tuning, typically used with open-source models, giving you full control to tailor the LLM for specific tasks.
3. **Open Source Models with Specialized Capabilities:**
    - Mention is made of open-source models with powerful features like instruction following and high-quality prompting. The Dolly model from Databricks and similar models are highlighted for their capabilities in this space.
4. **Evaluating LLMs Systematically:**
    - Evaluating LLMs is challenging due to their text generation nature.
    - The module discusses techniques for systematically evaluating LLMs, which involve assessing their text outputs in terms of quality and relevance.
5. **Alignment and Content Moderation:**
    - Addressing the importance of aligning LLMs with desired ethical and content guidelines.
    - Ensuring LLMs are not offensive and can perform content moderation tasks.

This module provides insights into fine-tuning LLMs, evaluating their performance, and maintaining alignment with ethical and content standards, emphasizing the quest for the best possible LLM performance in various applications.

Module 4 provides insights into fine-tuning Large Language Models (LLMs) and focuses on optimizing LLM applications. Here are the key points covered in this module:

1. **Introduction to Module 4:**
    - Module 4 marks the midway point in the course, where you've already learned about the capabilities and applications of LLMs.
2. **Customizing LLMs for Specific Applications:**
    - The module addresses scenarios where existing LLMs may not be perfectly suited for certain applications, emphasizing the need for customization.
    - It discusses how to adapt and fine-tune different types of LLMs to build specialized applications.
3. **Objectives of the Module:**
    - By the end of the module, learners should understand when and how to fine-tune various LLMs.
    - The module includes practical examples using DeepSpeed and Hugging Face models for fine-tuning and explains the importance of evaluating the customized LLMs.
4. **Typical LLM Releases:**
    - The module highlights the typical release structure of open-source LLMs, which often come in different sizes (base, smaller, larger) and sequence lengths.
    - Various specialized versions, such as chat-based models and instruction-based models, are also released alongside foundation models.
        
        ![Screenshot 2023-09-09 at 10.49.52 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.49.52_PM.png)
        
5. **Balancing Factors in LLM Selection:**
    - Developers must strike a balance between accuracy, speed, and task-specific performance when choosing an LLM.
    - Larger models generally offer better accuracy due to their extensive training, but they can be slower in inference. Smaller models are faster but may lack task-specific performance.
        
        ![Screenshot 2023-09-09 at 10.50.37 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.50.37_PM.png)
        
6. **Use Case: Building a News Article Summarization Application:**
    - The module presents a specific use case in which learners will build an application that takes news articles, summarizes them, and turns them into riddles for users to solve.

This module serves as a foundation for understanding the fine-tuning process of LLMs, enabling developers to tailor these models for specialized applications. It also emphasizes the need to consider factors like accuracy, speed, and task-specific performance when selecting an appropriate LLM.

## **Applying Foundation LLMs**

In this section, we'll explore the process of building a news application that summarizes daily news articles and presents them as riddles for users to solve. We'll consider various approaches using different available Large Language Models (LLMs) and tools:

1. **Application Objectives:**
    - The goal is to create a news application that automatically summarizes daily news articles and transforms them into riddles for user engagement.
        
        ![Screenshot 2023-09-09 at 10.53.47 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.53.47_PM.png)
        
2. **Available Resources:**
    - We have access to an Application Programming Interface (API) that connects to a news outlet, allowing us to fetch daily articles.
    - We also have a limited set of pre-made examples of news-based riddles.
        
        ![Screenshot 2023-09-09 at 10.54.02 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.54.02_PM.png)
        
3. **LLM Pipeline Options:**
    - **Few-Shot Learning with Open Source LLM:** We can utilize open-source LLMs for few-shot learning to make them capable of summarizing news articles and generating riddles. Few-shot learning involves providing a few examples of the task to the model.
    - **Instruction-Following LLM:** This type of LLM specializes in following specific instructions. We can use an instruction-following LLM for zero-shot learning, where it can understand and execute instructions for summarization and riddle generation.
    - **LLM as a Service (Paid Option):** We can explore using an LLM as a service, which might provide pre-trained models and APIs for tasks like summarization and generation. This option may offer convenience but could involve costs.
    - **Build Our Own Path:** Alternatively, we can choose to build a custom LLM tailored to our specific requirements. This path gives us complete control over the model's training and fine-tuning processes.
4. **Implementation Plan:**
    - Regardless of the chosen approach, the ultimate objective is to create an application interface that can process news articles, generate summaries, and convert them into engaging riddles for users.
    - Each approach will have its unique steps and considerations, such as training, fine-tuning, and integrating the LLM into the application.

The section sets the stage for exploring these options in more detail, showcasing how different LLMs and tools can be leveraged to achieve the desired functionality for the news application.

## [Fine-Tuning: Few-shot learning](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@851a9b0f70f14faaba0e223ee14154f4)

In this section, we explore the first approach to building a news application that involves Few-Shot Learning with an existing Large Language Model (LLM). Here are the key points:

1. **Approach Overview:**
    - The goal is to use an open-source LLM to perform few-shot learning, allowing the model to summarize news articles and generate riddles based on pre-made examples.
    - Available resources include a news API for fetching articles and a limited set of pre-made examples of articles converted into riddles.
        
        ![Screenshot 2023-09-09 at 10.56.24 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.56.24_PM.png)
        
2. **Pros and Cons of Few-Shot Learning:**
    - Pros:
        - Quick development process as existing data and an LLM are used with specified prompts.
        - Minimal computational costs since no training is involved.
    - Cons:
        - Requires a larger model for better performance.
        - Needs a sufficient number of good-quality examples covering various intents and article scopes.
        - Larger models may pose space and computational challenges.
            
            ![Screenshot 2023-09-09 at 10.56.36 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.56.36_PM.png)
            
3. **Implementation Steps:**
    - Construct a prompt that instructs the LLM to summarize news articles and create riddles from the summaries.
    - Provide a set of articles and their corresponding summary riddles as examples within the prompt.
    - Use a long input sequence model, which might be a large version of the LLM, to accommodate the task's complexity.
        
        ![Screenshot 2023-09-09 at 10.57.29 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.57.29_PM.png)
        

While this approach leverages few-shot learning with an existing LLM, it faces challenges related to model size and data requirements. In the next video, we'll explore an alternative approach to building the news application.

## [Fine-Tuning: Instruction-following LLMs](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@e432a61d5a32443b91810f8884d3ecb5)

In this section, we explore another approach to building the news application using pre-fine-tuned instruction-following Large Language Models (LLMs). Here are the key points:

1. **Approach Overview:**
    - This approach relies on pre-fine-tuned instruction-following LLMs, assuming the absence of pre-made examples.
    - The idea is to instruct the LLM with a specific task description and provide the news article for summarization.
        
        ![Screenshot 2023-09-09 at 10.59.19 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.59.19_PM.png)
        
2. **Pros and Cons of Zero-Shot Learning with Instruction-Following LLMs:**
    - Pros:
        - Quick development as no data collection or model fine-tuning is required.
        - Potentially excellent performance if the fine-tuned model aligns well with the task.
        - Minimal computation costs, with fees incurred only during inference.
    - Cons:
        - Task specificity depends on how well the model was fine-tuned for the given application.
        - Lack of pre-made examples may limit the model's understanding of the specific task.
        - Model size may vary, necessitating the use of a larger version for some applications.
            
            ![Screenshot 2023-09-09 at 10.59.48 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_10.59.48_PM.png)
            
3. **Implementation Steps:**
    - Construct a concise and specific prompt that describes the summarization task.
    - Input the news article to the LLM, instructing it to produce a summary based on the provided task description.
        
        ![Screenshot 2023-09-09 at 11.00.28 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.00.28_PM.png)
        

This approach leverages instruction-following LLMs for zero-shot learning. While it offers quick development and low computational costs, the effectiveness of this method depends on the quality of the fine-tuned model and how well it aligns with the intended task. In cases where the model is not well-suited, alternative approaches may be needed, as explored in the next section.

## [LLMs-as-a-Service](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@b27f998f394c4829bb3313fcfc413ed5)

In this section, we explore the option of utilizing a proprietary Large Language Model (LLM) as a service to build the news application. Here are the key points:

1. **Approach Overview:**
    - This approach involves using a proprietary LLM service, assuming no pre-made examples are available initially.
    - The focus is on incorporating the LLM service seamlessly into the application workflow.
        
        ![Screenshot 2023-09-09 at 11.01.43 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.01.43_PM.png)
        
2. **Pros and Cons of LLM as a Service:**
    - Pros:
        - Quick and easy integration into the application via API calls.
        - High-performance results, as the computation is handled on the service provider's side.
        - Minimal infrastructure management and maintenance efforts.
    - Cons:
        - Cost associated with API usage, as you pay per token sent and received.
        - Potential data privacy and security risks, as the service provider has access to the data.
        - Vendor lock-in risks, as changes in service offerings or pricing can impact the application.
            
            ![Screenshot 2023-09-09 at 11.02.01 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.02.01_PM.png)
            
3. **Implementation Steps:**
    - Construct a tokenized prompt to send to the LLM service via API.
    - Use an API key or other credentials to authenticate and authorize access to the service.
    - Receive and process the response from the LLM service.
        
        ![Screenshot 2023-09-09 at 11.03.07 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.03.07_PM.png)
        

This approach offers a straightforward and efficient way to leverage LLM capabilities, especially when high performance is required. However, it comes with cost considerations, data privacy concerns, and potential vendor lock-in risks. If this approach does not meet the specific needs of the application, fine-tuning an LLM for customization will be explored in the next section.

## [Fine-tuning: DIY](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@60b66c2c12af425a878c1410643c6e5d)

In this section, we explore the option of fine-tuning an existing Large Language Model (LLM) to create a task-specific version for the news application. Here are the key points:

1. **Considerations for Building Your Own LLM:**
    - When existing LLMs do not provide the desired results due to lack of specific training data, building a custom LLM is an option.
        
        
        ![Screenshot 2023-09-09 at 11.04.32 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.04.32_PM.png)
        
    - Two approaches are available: building a foundation model from scratch or fine-tuning an existing model. Building a foundation model from scratch is resource-intensive and typically not feasible for most developers.
        
        ![Screenshot 2023-09-09 at 11.04.44 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.04.44_PM.png)
        
        ![Screenshot 2023-09-09 at 11.05.10 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.05.10_PM.png)
        
2. **Fine-Tuning an Existing Model:**
    - Fine-tuning allows the creation of a task-specific LLM without starting from scratch.
    - Benefits of fine-tuning include tailored models, cost savings in inference, control over data and dataset, and the ability to create task-specific versions.
        
        ![Screenshot 2023-09-09 at 11.05.23 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.05.23_PM.png)
        
3. **Requirements for Fine-Tuning:**
    - Sufficient data for fine-tuning, which should cover the specific use case adequately.
    - Necessary skill sets for fine-tuning, which can be acquired through relevant courses and resources.
4. **Significance of Dolly V2:**
    - Dolly V2 is a notable development in fine-tuning LLMs, released by Databricks.
    - It utilizes an open-source dataset, making it more accessible and less restricted by proprietary data or licensing issues.
    - The availability of Dolly V2 enhances the possibilities for customizing LLMs for various applications.
        
        ![Screenshot 2023-09-09 at 11.06.01 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.06.01_PM.png)
        

Overall, fine-tuning existing LLMs, like Dolly V2, provides a practical and efficient way to create task-specific models when standard pre-trained models do not suffice for specialized applications.

### [Dolly](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@746020fe3d194dbdbc138fbf27a5b7a3)

Dolly, introduced in 2023, represents a significant innovation in large language modeling. Here are the key points about Dolly:

1. **Dolly Overview:**
    - Dolly is a large language model with 12 billion parameters.
    - It is an instruction-following LLM, designed to perform specific tasks based on provided instructions.
2. **Shift in Paradigm:**
    - Dolly's approach marks a shift in the direction of large language model development.
3. **Development of Dolly:**
    - Dolly was created by fine-tuning an open-source foundation model, the EleutherAI Pythia 12 billion parameter model.
    - It was fine-tuned on the Databricks-Dolly-15K dataset, a crucial component that makes Dolly special.
4. **Databricks-Dolly-15K Dataset:**
    - The dataset, created by Databricks employees, consists of instruction-response pairs for high-quality intellectual tasks.
    - It sets Dolly apart because it was released openly for commercial use, eliminating licensing restrictions.
5. **Dolly's Significance:**
    - Dolly itself is not a state-of-the-art model but illustrates the potential of combining open-source models with high-quality, open datasets to create commercially viable models.
        - The early months of 2023 witnessed a momentum shift towards models that excel in specific tasks, rather than pursuing broad mastery.
            - The Alpaca project produced a capable model but faced licensing restrictions, limiting its commercial use.
                
                ![Screenshot 2023-09-09 at 11.07.20 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.07.20_PM.png)
                
6. **Origin and Influence:**
    - The idea for Dolly was inspired by the Stanford Alpaca project, which used instructions to generate synthetic tasks.
7. **The Age of Small LLMs:**
    - Dolly's emergence aligns with a broader shift in the field towards smaller, task-specific LLMs.
    - Rather than chasing larger models for general mastery, the focus is on fine-tuning bespoke models for specific tasks.
        
        ![Screenshot 2023-09-09 at 11.08.43 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.08.43_PM.png)
        
8. **Future Directions:**
    - The evolution of the field beyond large models is exciting, and it remains to be seen how small LLMs will be applied in various use cases.
        
        ![Screenshot 2023-09-09 at 11.09.28 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.09.28_PM.png)
        

Dolly serves as a compelling example of how open-source models and high-quality open datasets can be combined to create practical and specialized large language models for specific applications.

## **Evaluating LLMs**

Evaluating the performance of fine-tuned Large Language Models (LLMs) is crucial, but it can be challenging to assess their effectiveness. Here are the key points regarding LLM evaluation:

![Screenshot 2023-09-09 at 11.10.56 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.10.56_PM.png)

1. **Complexity of LLM Evaluation:**
    - Evaluating LLMs is intricate due to their generative nature and the absence of traditional classification metrics.
    - Loss or validation scores may not provide meaningful insights into LLM performance.
        
        ![Screenshot 2023-09-09 at 11.11.06 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.11.06_PM.png)
        
2. **Perplexity as an Indicator:**
    - Perplexity measures the spread of probability distribution over tokens predicted by the LLM.
    - Lower perplexity indicates that the model is confident in its token predictions.
    - High accuracy and low perplexity are desirable traits for a good language model.
        
        ![Screenshot 2023-09-09 at 11.11.32 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.11.32_PM.png)
        
3. **Limitations of Perplexity:**
    - While perplexity measures confidence in token selection, it does not ensure the overall quality of generated text.
    - The repeated use of the same token, even with low perplexity and high accuracy, may result in incoherent or nonsensical text.
4. **Task-Specific Evaluation Metrics:**
    - To assess LLMs effectively, task-specific evaluation metrics are essential.
    - Different applications (e.g., translation, summarization, conversation) require tailored evaluation criteria.
    - Task-specific metrics provide insights into the quality, coherence, and relevance of generated content.
        
        ![Screenshot 2023-09-09 at 11.11.58 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.11.58_PM.png)
        
    

In summary, evaluating fine-tuned LLMs goes beyond perplexity and accuracy. Task-specific evaluation metrics are essential to assess the quality and suitability of LLM-generated content for specific applications, ensuring that the model performs effectively in its intended use case.

### [Task-specific Evaluations](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@ef64e05b12344323b35c6b38d2a5cfcb)

Task-specific evaluation metrics are essential for assessing the performance of fine-tuned Large Language Models (LLMs) in various applications. Here are the key evaluation metrics and concepts discussed:

1. **Translation Evaluation - BLEU Metric:**
    - BLEU (Bilingual Evaluation Understudy) measures the quality of machine-generated translations.
    - It calculates the overlap of unigrams (single words), bigrams (pairs of consecutive words), trigrams, and quadgrams between the output and reference translations.
    - BLEU computes a geometric mean of these n-gram scores to provide an overall translation quality score.
        
        ![Screenshot 2023-09-09 at 11.13.26 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.13.26_PM.png)
        
2. **Summarization Evaluation - ROUGE Score:**
    - ROUGE (Recall-Oriented Understudy for Gisting Evaluation) assesses the quality of text summarization.
    - Similar to BLEU, it compares the overlap of n-grams between the generated summary and reference summaries.
    - ROUGE also considers the length of the summary, favoring shorter yet informative outputs.
        
        ![Screenshot 2023-09-09 at 11.14.07 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.14.07_PM.png)
        
3. **Benchmark Datasets:**
    - Benchmark datasets, such as SQuAD (Stanford Question and Answering Dataset), are used to evaluate LLMs and compare their performance with other models.
    - These datasets provide standardized tasks and evaluation criteria to ensure fair comparisons among models.
        
        ![Screenshot 2023-09-09 at 11.14.39 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.14.39_PM.png)
        
4. **Alignment and Hallucination:**
    - Alignment assesses how well an LLM aligns with the input and provides relevant responses.
    - Hallucination refers to the generation of content not present in the input or reference data.
    - Harmlessness measures the toxicity or profanity in LLM-generated responses.
        
        ![Screenshot 2023-09-09 at 11.14.56 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.14.56_PM.png)
        
5. **Ongoing Research in Evaluation Metrics:**
    - Researchers continue to develop new evaluation metrics to address specific challenges in LLM performance assessment.
    - Alignment remains a critical focus, ensuring that LLMs provide meaningful and contextually relevant responses.

In summary, evaluation metrics like BLEU and ROUGE are essential for assessing LLMs' performance in translation and summarization tasks. Benchmark datasets and ongoing research in alignment and other metrics help advance the field and enable fair comparisons among different models.

## **Guest Lecture from Harrison Chase**

In this discussion on the evaluation of Large Language Model (LLM) chains and agents, several key points are highlighted:

![Screenshot 2023-09-09 at 11.16.30 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.16.30_PM.png)

1. **Overview of LLM Chains and Agents:**
    - LLM chains and agents use LLMs as reasoning engines to interact with external data and computation sources while relying on those sources for knowledge rather than using the LLM's internal knowledge.
    - A common example is retrieval augmented generation, where a chatbot answers questions about documents not present in the LLM's training data.
        
        ![Screenshot 2023-09-09 at 11.16.57 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.16.57_PM.png)
        
        ![Screenshot 2023-09-09 at 11.17.11 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.17.11_PM.png)
        
2. **Challenges in Evaluation:**
    - Evaluating LLM chains and agents is challenging due to the lack of readily available data.
    - It's difficult to determine ground truth answers, especially for applications involving constantly changing or up-to-date information.
        
        
        ![Screenshot 2023-09-09 at 11.18.08 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.18.08_PM.png)
        
    - Lack of suitable metrics compounds the challenge as evaluating the quality of responses generated by LLMs is complex.
        
        ![Screenshot 2023-09-09 at 11.18.41 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.18.41_PM.png)
        
3. **Potential Solutions:**
    - To address the lack of data, data sets can be generated programmatically, involving LLMs in the process. Data accumulation over time from real-world application usage is also valuable.
        
        ![Screenshot 2023-09-09 at 11.19.16 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.19.16_PM.png)
        
    - For addressing the lack of metrics, visualization tools are essential to inspect and understand the inputs and outputs at each step in the chain.
        
        ![Screenshot 2023-09-09 at 11.19.53 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.19.53_PM.png)
        
    - Language models can be employed to judge the correctness of final answers, comparing them with ground truth labels.
    - Feedback from users, collected directly (e.g., thumbs up/thumbs down) or indirectly (e.g., click-through rates on suggested links), provides valuable insights for online evaluation.
4. **Offline Evaluation:**
    - Offline evaluation occurs before the model is deployed in production.
    - It involves creating a test data set, running the LLM chain or agent against it, and visually inspecting the results.
    - Language models can be used for auto-grading, but visual inspection remains important.
        
        ![Screenshot 2023-09-09 at 11.20.33 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.20.33_PM.png)
        
5. **Online Evaluation:**
    - Online evaluation takes place after the model is deployed and actively serving users.
    - Feedback from users, both direct and indirect, helps monitor the model's ongoing performance.
    - Trends in feedback, such as increasing negative feedback, can indicate a need for model adjustments.
        
        ![Screenshot 2023-09-09 at 11.21.16 PM.png](Fine-tuning%20and%20Evaluating%20LLMs%20a5c821cf3ef943dd9b45c587f1681ddc/Screenshot_2023-09-09_at_11.21.16_PM.png)
        
6. **Ongoing Development:**
    - The field of LLM application evaluation is evolving rapidly as more applications enter production.
    - Emerging best practices and methodologies for evaluation are expected to develop further as these applications become more prevalent.

Overall, evaluating LLM chains and agents is a dynamic and evolving field with the challenge of balancing data availability and defining suitable metrics for complex, real-world applications.

## References

1. **Fine-tuned models**
    - [HF leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
    - [MPT-7B](https://www.mosaicml.com/blog/mpt-7b)
    - [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
    - [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
    - [DeepSpeed on Databricks](https://www.databricks.com/blog/2023/03/20/fine-tuning-large-language-models-hugging-face-and-deepspeed.html)
2. **Databricks’ Dolly**
    - [Dolly v1 blog](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
    - [Dolly v2 blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
    - [Dolly on Hugging Face](https://huggingface.co/databricks/dolly-v2-12b)
    - [Build your own Dolly](https://www.databricks.com/resources/webinar/build-your-own-large-language-model-dolly)
3. **Evaluation and Alignment in LLMs**
    - [HONEST](https://huggingface.co/spaces/evaluate-measurement/honest)
    - [LangChain Evaluate](https://docs.langchain.com/docs/use-cases/evaluation)
    - [OpenAI’s post on InstructGPT and Alignment](https://openai.com/research/instruction-following)
    - [Anthropic AI Alignment Papers](https://www.anthropic.com/index?subjects=alignment)

# 6. Society and LLMs

## Introduction

Module 5 focuses on the societal risks associated with Large Language Models (LLMs) and strategies to mitigate these risks. It highlights the challenges of unreliable behavior in LLMs and provides key insights into two sources of this behavior:

1. **Biases and Incorrect Information:**
    - LLMs, when trained on diverse web data, can acquire biases present in that data.
    - Web data may contain incorrect or outdated information that LLMs might inadvertently propagate.
    - Undesirable behavior includes generating biased or factually incorrect responses.
2. **Hallucination:**
    - Hallucination occurs when LLMs generate fabricated or speculative responses when they lack sufficient knowledge to provide accurate information.
    - These responses can sound confident, potentially misleading users.
    

The module underscores the importance of addressing these risks in LLM applications and outlines strategies for mitigation, including:

- **Basic Generation and Retrieval:** Combining LLMs with retrieval from a knowledge base that has undergone human review to ensure accuracy and reliability.
- **Evaluation Techniques:** Employing various evaluation methods to assess model performance, especially in cases where ground truth or quality control is challenging.
- **User Awareness:** Educating users about the limitations and potential biases of LLMs to prevent them from being misled by the generated content.

Overall, Module 5 emphasizes the responsibility of developers and organizations to be mindful of the societal implications of LLMs and to take proactive steps to mitigate risks and ensure the responsible use of these powerful language models.

Module Overview:

- **Purpose and Disclaimer:** The module begins with a disclaimer acknowledging that LLM-generated content may occasionally be offensive, biased, or harmful. However, these demonstrations are for educational purposes only.
- **Understanding LLM Merits and Risks:** The module aims to provide a comprehensive understanding of both the advantages and risks associated with Large Language Models (LLMs). It highlights the transformative potential of LLMs in various industries, emphasizing their ability to reduce manual labor, cut costs, improve content creation, enhance customer service, and aid in accessibility for learners.
    
    ![Screenshot 2023-09-09 at 11.27.20 PM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-09_at_11.27.20_PM.png)
    
- **Impact of Training Data:** The discussion shifts to the impact of training data on LLMs. It explains how LLMs learn from diverse datasets, including those from the web, which can introduce biases and inaccuracies into their responses.
- **Hallucination:** The module addresses one of the prominent challenges in LLMs, which is hallucination. Hallucination occurs when LLMs generate speculative or fabricated responses, often with unwarranted confidence.
- **Evaluation and Mitigation:** It explores methods to evaluate and mitigate hallucination, emphasizing the importance of evaluation techniques and quality control mechanisms.
- **Ethical and Responsible Usage:** The module underscores the need for ethical and responsible usage of LLMs, considering their societal impact. It discusses the responsible governance of AI systems and the importance of user awareness regarding the limitations and potential biases of LLM-generated content.

In summary, this module provides a balanced perspective on the merits and risks of LLMs, covering their transformative potential in various domains while addressing concerns related to biases, hallucination, and ethical usage.

## **Risks and Limitations**

Module Overview:

- **Understanding LLM Risks and Limitations:** This module explores the risks and limitations associated with Large Language Models (LLMs). It acknowledges that while LLMs have transformative capabilities, they also come with significant challenges and potential harm.
    
    ![Screenshot 2023-09-10 at 12.56.17 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_12.56.17_AM.png)
    
- **Data and Model Bias:** The discussion begins with a focus on the role of data in enabling LLMs' power. It explains how training data can introduce biases and inaccuracies into LLMs, impacting their responses and reliability.
- **Misuse of LLMs:** The module addresses the misuse of LLMs, whether intentional or unintentional. It discusses concerns related to LLM-generated content, attribution, copyright infringement, and the potential impact on creative industries.
    
    ![Screenshot 2023-09-10 at 1.09.37 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.09.37_AM.png)
    
- **Impact on Jobs:** It explores the potential displacement of jobs due to automation driven by LLMs. The module highlights the declining prospects for certain job roles and the adverse effects of constant exposure to toxic content for workers in specific roles.
    
    ![Screenshot 2023-09-10 at 1.10.01 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.10.01_AM.png)
    
- **Environmental and Financial Costs:** The environmental and financial costs of training large language models are discussed. It emphasizes the high costs associated with training these models and how this can limit accessibility, particularly for small businesses and individuals.
    
    ![Screenshot 2023-09-10 at 1.10.54 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.10.54_AM.png)
    
- **Data Representation and Quality:** The module delves into issues related to data representation and quality, emphasizing that big data doesn't always imply good data. It discusses the challenges of diverse data representation, data auditing, and the risk of using flawed data in LLM training.
    
    ![Screenshot 2023-09-10 at 1.11.29 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.11.29_AM.png)
    
- **Bias in LLMs:** It highlights how biases present in training data can manifest in LLM outputs, leading to toxicity, discrimination, and exclusion of certain demographic groups.
    
    
    ![Screenshot 2023-09-10 at 1.12.26 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.12.26_AM.png)
    
    ![Screenshot 2023-09-10 at 1.13.15 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.13.15_AM.png)
    
- **Information Hazards:** Information hazards are discussed, encompassing risks of compromising privacy, leaking sensitive information, and enabling malicious uses such as fraud, censorship, surveillance, and cyberattacks.
    
    ![Screenshot 2023-09-10 at 1.13.57 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.13.57_AM.png)
    
    ![Screenshot 2023-09-10 at 1.14.43 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.14.43_AM.png)
    
- **Overreliance on LLMs:** The module addresses the risk of overreliance on LLMs and the importance of responsible usage, especially in critical areas like mental health.
    
    ![Screenshot 2023-09-10 at 1.14.52 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.14.52_AM.png)
    
- **Introduction to Hallucination:** The module sets the stage for the next section, introducing the concept of hallucination and its significance in LLM behavior.

In summary, this module provides a comprehensive overview of the risks and limitations associated with LLMs, touching on data bias, misuse, job displacement, environmental costs, data quality, bias in outputs, information hazards, overreliance, and the upcoming discussion on hallucination. It emphasizes the need for responsible and thoughtful interaction with LLMs.

## [Hallucination](https://learning.edx.org/course/course-v1:Databricks+LLM101x+2T2023/block-v1:Databricks+LLM101x+2T2023+type@sequential+block@5f2cbe8c2a2742be8fbcdb9b3e098b37)

**Understanding Hallucination in LLMs:**

- **Definition of Hallucination:** Hallucination in the context of Large Language Models (LLMs) refers to the generation of content that is nonsensical or unfaithful to the source content. Despite sounding natural and confident, hallucinated content can be factually incorrect.
- **Two Types of Hallucination:**
    - **Intrinsic Hallucination:** Intrinsic hallucination occurs when the generated output directly contradicts the source content, leading to a lack of faithfulness and factual accuracy.
    - **Extrinsic Hallucination:** Extrinsic hallucination happens when the output cannot be verified against the source, making it challenging to assess its faithfulness.
        
        ![Screenshot 2023-09-10 at 1.17.20 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.17.20_AM.png)
        
- **Factors Leading to Hallucination:**
    - **Data:** Data quality and collection methods play a significant role in the likelihood of hallucination. Challenges include the difficulty of auditing large datasets, the lack of factual verification during data collection, and the desire for diverse responses in open-ended tasks.
        
        ![Screenshot 2023-09-10 at 1.18.46 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.18.46_AM.png)
        
    - **Model:** Hallucination can result from various model-related factors, including imperfect encoder learning, decoding errors, exposure bias, and parametric knowledge bias.
        
        
        ![Screenshot 2023-09-10 at 1.20.04 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.20.04_AM.png)
        
- **Evaluation of Hallucination:**
    - **Statistical Metrics:** Metrics like BLEU, ROUGE, METEOR, and PARENT can quantitatively measure hallucination by comparing the output to the source and assessing the presence of unsupported information.
    - **Model-Based Metrics:** These metrics leverage other models or tasks to evaluate hallucination. They include information extraction, question-answering-based methods, faithfulness assessment, and language-model-based evaluation.
        
        ![Screenshot 2023-09-10 at 1.21.55 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.21.55_AM.png)
        
- **Challenges in Evaluation:** Evaluating hallucination is challenging due to varying individual expectations and criteria for determining the presence of unsupported or incorrect information. While several metrics exist, none are perfect.

### **Mitigation Strategies**

**Mitigation Strategies for Hallucination and Addressing LLM Risks and Limitations:**

- **Faithful Dataset Construction:**
    - Involve humans in writing clean and faithful target data based on source text.
    - Rewrite and filter real sentences from the web, making corrections as needed.
    - Consider augmenting input data sources with additional reliable sources.
- **Architectural Research and Experimentation:**
    - Explore architectural improvements in modeling and inference methods.
    - Experiment with reinforcement learning and multi-task learning approaches.
    - Implement post-processing corrections with human involvement to reduce hallucination.
        
        ![Screenshot 2023-09-10 at 1.23.38 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.23.38_AM.png)
        
- **Combating LLM Risks and Limitations:**
    - Address data bias by examining data slices and updating data more frequently.
    - Combat toxic models through data assessment, post-processing tools, and guardrails.
    - Tackle information hazards by evaluating the sources of information and curating data for fine-tuning.
    - Deal with malicious users through regulations and governance.
        
        ![Screenshot 2023-09-10 at 1.24.15 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.24.15_AM.png)
        
        ![Screenshot 2023-09-10 at 1.25.03 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.25.03_AM.png)
        
- **Three-Layered Auditing Framework:**
    - Governance: Audit technology providers (companies offering LLMs).
    - Models: Audit LLM models before public release.
    - Application Level: Assess risks based on how users interact with the models.
        
        ![Screenshot 2023-09-10 at 1.25.14 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.25.14_AM.png)
        
- **Open Questions and Challenges:**
    - Capturing the entire landscape of model usage and interaction.
    - Auditing closed-source models and ensuring accountability.
    - Establishing arbitrary thresholds for risk acceptance.
    - Addressing deliberate misuse and navigating gray areas in creative product generation.
        
        ![Screenshot 2023-09-10 at 1.25.54 AM.png](Society%20and%20LLMs%209d4faf092f6a4f08821f781c4c93bc8c/Screenshot_2023-09-10_at_1.25.54_AM.png)
        
- **Societal Considerations:** As LLM technology advances, society must grapple with questions related to governance, auditing responsibility, misuse detection, and ethical usage.

The management of LLM risks and limitations requires a multifaceted approach involving data curation, model development, regulations, and ethical considerations to ensure responsible and safe utilization of these powerful language models.

## References

1. **Social Risks and Benefits of LLMs**
    - [Weidinger et al 2021 (DeepMind)](https://arxiv.org/pdf/2112.04359.pdf)
    - [Bender et al 2021](https://dl.acm.org/doi/10.1145/3442188.3445922)
    - [Mokander et al 2023](https://link.springer.com/article/10.1007/s43681-023-00289-2)
    - [Rillig et al 2023](https://pubs.acs.org/doi/pdf/10.1021/acs.est.3c01106)
    - [Pan et al 2023](https://arxiv.org/pdf/2305.13661.pdf)
2. **Hallucination**
    - [Ji et al 2022](https://arxiv.org/pdf/2202.03629.pdf)
3. **Bias evaluation metrics and tools**
    - [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
    - [Guardrails.ai](https://shreyar.github.io/guardrails/)
    - [Liang et al 2022](https://arxiv.org/pdf/2211.09110.pdf)
4. **Other general reading**
    - [All the Hard Stuff Nobody Talks About when Building Products with LLMs by Honeycomb](https://www.honeycomb.io/blog/hard-stuff-nobody-talks-about-llm)
    - [Science in the age of large language models by Nature Reviews Physics](https://www.nature.com/articles/s42254-023-00581-4)
    - [Language models might be able to self-correct biases—if you ask them by MIT Technology Review](https://www.technologyreview.com/2023/03/20/1070067/language-models-may-be-able-to-self-correct-biases-if-you-ask-them-to/)

# 7. LLMOps

## Introduction

**Module on LLM Ops: Getting Large Language Model Applications into Production**

- **Importance of Production-Ready LLM Applications:**
    - Reliability and readiness for production are crucial for large language model (LLM) applications.
    - Transitioning from a research or prototype stage to a production-ready application is essential for long-term success.
    - Real-world LLM applications need to handle changing user needs, evolving base models, shifting data sets, and dynamic knowledge bases.
- **Parallels with MLOps:**
    - Just as machine learning operations (MLOps) emerged to manage machine learning models in production, LLM Ops addresses the unique challenges of large language models.
    - LLM Ops involves maintaining, operating, and optimizing LLM applications to ensure their reliability and performance over time.
- **Key Aspects of LLM Ops:**
    - **Full Stack Management:** LLM Ops encompasses all components of an application stack, including the LLM, vector databases, chains, and end-to-end applications.
    - **Monitoring:** Continuous monitoring is vital for identifying issues, tracking performance, and ensuring user satisfaction.
    - **Quality Improvement:** Strategies for maintaining and enhancing LLM-generated content quality over time.
    - **Collaborative Development:** Approaches to collaborative development and version control for LLM applications.
    - **Testing:** Robust testing methodologies for LLM applications to catch errors and ensure reliability.
    - **High Performance:** Optimization techniques to achieve high performance and responsiveness in LLM applications.

LLM Ops aims to transform LLM applications from initial prototypes or research projects into reliable, production-ready systems capable of meeting users' evolving needs and expectations.

**Module 6: LLMOps - Getting LLMs into Production**

**Learning Objectives:**

- Understand how traditional MLOps principles can be adapted for Large Language Models (LLMs).
- Explore end-to-end workflows and architectures for deploying LLM-powered applications.
- Examine key considerations specific to LLMOps, including cost-performance trade-offs, deployment options, monitoring, and feedback.

**Overview:**

- The primary goal of Module 6, LLMOps, is to enable the deployment of Large Language Models (LLMs) into production.
- The module focuses on adapting traditional Machine Learning Operations (MLOps) concepts to the unique challenges posed by LLM applications.
- The key objectives include discussing the adaptation of MLOps for LLMs, reviewing end-to-end workflows and architectures, and addressing specific LLMOps concerns like cost-performance trade-offs, deployment strategies, monitoring, and feedback mechanisms.

**Background on MLOps:**

- MLOps has gained prominence in recent years as ML and AI have become integral to businesses.
- MLOps serves two main goals:
    1. Maintaining stable performance: Ensuring key performance indicators (KPIs) related to ML model accuracy, system latency, throughput, etc., meet expectations.
    2. Maintaining long-term efficiency: Automating manual processes, reducing development-to-production cycles, and ensuring compliance with requirements and regulations.
- MLOps was not widely recognized as a term just a few years ago, but it has since become a critical practice for deploying and managing ML models in production.
    
    ![Screenshot 2023-09-10 at 1.31.51 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.31.51_AM.png)
    

This module lays the foundation for understanding how traditional MLOps principles can be adapted to address the unique challenges of deploying and managing Large Language Models in production environments.

## **Traditional MLOps**

**Traditional MLOps Overview:**

- MLOps is a comprehensive approach that combines elements of DevOps, DataOps, and ModelOps to manage machine learning (ML) assets effectively. It involves processes and automation to enhance performance and long-term efficiency.
    
    ![Screenshot 2023-09-10 at 1.33.25 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.33.25_AM.png)
    
- MLOps aims to achieve two main goals:
    1. **Maintaining Stable Performance**: Ensuring that key performance indicators (KPIs) related to ML models, system latency, and throughput meet defined standards.
    2. **Maintaining Long-Term Efficiency**: Streamlining processes, automating manual tasks, and ensuring compliance with regulations to improve efficiency.
- Traditional MLOps practices encompass various aspects such as source control, testing, monitoring, CI/CD (Continuous Integration/Continuous Deployment), and more.
- Reference architecture for traditional MLOps includes:
    - **Source Control**: Managing code.
    - **Lakehouse Data Layer**: A shared data layer with controlled access.
    - **Development Environment**: Where data scientists and developers work on pipelines, including model training and feature table refresh.
    - **Staging Environment**: Where code undergoes CI tests, including unit and integration tests, to ensure it works with other pipelines and services.
    - **Production Environment**: Where pipelines and services are instantiated, and models are deployed and monitored.
- The staging environment aims to mimic the production environment as closely as possible, including the same set of services and pipelines.
- CI tests ensure that code is ready for deployment to production.
- In production, code, data, and models are orchestrated to provide services like feature table refresh, model retraining, and model deployment. Models are managed in a model registry, and CD pipelines ensure that models are deployed incrementally and monitored in production.
    
    ![Screenshot 2023-09-10 at 1.33.47 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.33.47_AM.png)
    

**Integration of LLM in MLOps:**

- The challenge of integrating Large Language Models (LLMs) into MLOps involves adapting traditional MLOps principles to address the unique characteristics and requirements of LLMs.
- LLMs introduce challenges related to handling large text-based models, and workflows need to accommodate these complexities.
- This integration requires considerations related to model size, resource requirements, data preprocessing, and real-time inference capabilities, among others.

In the next sections, the course will delve into LLMOps, focusing on how to effectively manage and deploy Large Language Models in production environments, considering their distinct requirements and challenges.

## **LLMOps**

**Key Considerations for LLMOps Integration:**

- Adapting traditional MLOps to accommodate Large Language Models (LLMs) involves several notable changes and considerations within the existing architecture:
1. **Model Training Changes**:
    - LLMs often require fine-tuning or lighter-weight training methods due to their massive size.
    - Options may include fine-tuning, pipeline tuning, or prompt engineering.
    - These are essentially pipelines or code elements that traditional MLOps can handle.
        
        ![Screenshot 2023-09-10 at 1.37.41 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.37.41_AM.png)
        
2. **Human/User Feedback Integration**:
    - Human feedback is crucial for LLMs and should be treated as an essential data source.
    - It should be considered at every stage from development to production.
    - Feedback may come from various sources, both internal and external.
        
        ![Screenshot 2023-09-10 at 1.38.13 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.38.13_AM.png)
        
3. **Monitoring and Quality Testing**:
    - While traditional monitoring can be automated, LLMs may require a constant human feedback loop.
    - Automated quality testing may be challenging and need augmentation with human evaluation.
    - Incremental rollouts, exposing the model or pipeline to a small user group for evaluation, can be more practical than batch testing.
        
        
        ![Screenshot 2023-09-10 at 1.38.38 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.38.38_AM.png)
        
4. **Production Tooling**:
    - Handling large models may require shifting from CPUs to GPUs for serving.
    - The data layer might involve new components like vector databases to handle LLM-specific needs.
        
        ![Screenshot 2023-09-10 at 1.39.12 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.39.12_AM.png)
        
5. **Cost and Performance Challenges**:
    - LLMs can introduce cost, latency, and performance trade-offs.
    - Fine-tuning and managing resources need careful consideration.
    - Comparing fine-tuned models to third-party LLM APIs involves assessing costs and performance differences.
        
        ![Screenshot 2023-09-10 at 1.39.25 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.39.25_AM.png)
        
- Despite these changes, several elements remain similar in LLMOps:
    - The separation of development, staging, and production environments, as well as access control enforcement, remains consistent.
    - Git and model registries continue to serve as conduits for managing code and models.
    - The Lakehouse data architecture remains critical.
    - Existing CI infrastructure can be reused.
    - The modular structure for developing data pipelines and services remains intact.
- In the upcoming video, further details about these changes and considerations in LLMOps will be explored.

The integration of LLMs into MLOps introduces unique challenges, particularly related to fine-tuning, human feedback, monitoring, and cost-performance trade-offs. While some aspects change, the foundational principles of MLOps remain applicable in managing LLMs throughout their lifecycle.

## **LLMOps Details**

**Key Topics in LLMOps:**

1. **Prompt Engineering and Automation**:
    - Prompt engineering involves tracking, templating, and automating prompts for LLMs.
    - Tracking queries and responses using tools like MLflow aids in development.
    - Templating standardizes prompt formats with tools like LangChain or LlamaIndex.
    - Automation, such as using DSP (Demonstrate-Search-Predict) frameworks, streamlines prompt tuning.
        
        ![Screenshot 2023-09-10 at 1.41.46 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.41.46_AM.png)
        
2. **Packaging Models for Deployment**:
    - MLflow offers a uniform format for logging models, aiding in standardizing deployment.
    - MLflow provides a model registry for tracking model versions' movement toward production.
    - Models can be deployed in various ways, including inline code, containers, batch or streaming processing, custom services, etc.
        
        ![Screenshot 2023-09-10 at 1.42.53 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.42.53_AM.png)
        
        ![Screenshot 2023-09-10 at 1.43.09 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.43.09_AM.png)
        
        ![Screenshot 2023-09-10 at 1.43.21 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.43.21_AM.png)
        
3. **Scaling Out and Distributed Computation**:
    - For training and fine-tuning, distributed frameworks like TensorFlow, PyTorch, or DeepSpeed may be used.
    - Serving and inference require scalable endpoints and pipelines.
    - Traditional scale-out frameworks like Apache Spark or Ray can be employed.
        
        ![Screenshot 2023-09-10 at 1.43.58 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.43.58_AM.png)
        
4. **Managing Cost-Performance Trade-Offs**:
    - Consider factors like query and training costs, development versus production costs, and expected query load.
    - Start with simpler models and gradually optimize for costs and performance.
    - Use smaller models, employ techniques like fine-tuning, distillation, quantization, and pruning to reduce costs.
        
        ![Screenshot 2023-09-10 at 1.44.24 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.44.24_AM.png)
        
5. **Human Feedback, Testing, and Monitoring**:
    - Plan for human feedback as an essential part of LLMOps.
    - Incorporate implicit feedback mechanisms into applications to gather user input.
    - Treat human feedback as data for both development and production phases.
    - Monitor application performance and user interactions.
        
        ![Screenshot 2023-09-10 at 1.45.42 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.45.42_AM.png)
        
6. **Deploying Models vs. Deploying Code**:
    - Distinguish between deploying pre-trained models and deploying code that generates models.
    - Use MLflow to log and manage models, making deployment processes more standardized.
        
        ![Screenshot 2023-09-10 at 1.46.30 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.46.30_AM.png)
        
7. **Service Architectures and Stability**:
    - Consider the choice between using vector databases as separate services or local tools within LLM pipelines.
    - Ensure stable behavior when offering LLM-based APIs as services, especially with complex or stochastic models.
    - Versioning endpoints and providing determinism options for users are essential for API stability.
        
        ![Screenshot 2023-09-10 at 1.48.21 AM.png](LLMOps%204772bcf3a3854c42b25b93727cea3a01/Screenshot_2023-09-10_at_1.48.21_AM.png)
        

These LLMOps topics cover various aspects of integrating LLMs into production, from prompt engineering to deployment strategies, scalability, cost management, human feedback, and ensuring stable API services. The next step is to apply these concepts to a practical, scale-out workflow in the provided code example.

## References

1. **General MLOps**
    - [“The Big Book of MLOps”](https://www.databricks.com/resources/ebook/the-big-book-of-mlops) (eBook overviewing MLOps)
        - Blog post (short) version: [“Architecting MLOps on the Lakehouse”](https://www.databricks.com/blog/2022/06/22/architecting-mlops-on-the-lakehouse.html)
        - MLOps in the context of Databricks documentation ([AWS](https://docs.databricks.com/machine-learning/mlops/mlops-workflow.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/mlops/mlops-workflow), [GCP](https://docs.gcp.databricks.com/machine-learning/mlops/mlops-workflow.html))
2. **LLMOps**
    - Blog post: Chip Huyen on “[Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)”
3. **[MLflow](https://mlflow.org/)**
    - [Documentation](https://mlflow.org/docs/latest/index.html)
        - [Quickstart](https://mlflow.org/docs/latest/quickstart.html)
        - [Tutorials and examples](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
        - Overview in Databricks ([AWS](https://docs.databricks.com/mlflow/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/mlflow/), [GCP](https://docs.gcp.databricks.com/mlflow/index.html))
4. **[Apache Spark](https://spark.apache.org/)**
    - [Documentation](https://spark.apache.org/docs/latest/index.html)
        - [Quickstart](https://spark.apache.org/docs/latest/quick-start.html)
    - Overview in Databricks ([AWS](https://docs.databricks.com/spark/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/spark/), [GCP](https://docs.gcp.databricks.com/spark/index.html))
5. **[Delta Lake](https://delta.io/)**
    - [Documentation](https://docs.delta.io/latest/index.html)
    - Overview in Databricks ([AWS](https://docs.databricks.com/delta/index.html), [Azure](https://learn.microsoft.com/en-us/azure/databricks/delta/), [GCP](https://docs.gcp.databricks.com/delta/index.html))
    - [Lakehouse Architecture (CIDR paper)](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)