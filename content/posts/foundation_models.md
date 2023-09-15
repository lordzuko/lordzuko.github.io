---
title: "Foundation Models from Ground Up"
date: 2023-08-01T10:29:21+08:00
draft: true
ShowToc: true
category: [ai]
tags: ["llms", "ai"]
description: "Understanding the core of Large Language Models, a journey through evolution of various transformer models and their applications."
---

# 1. Transformer Architecture: Attention & Transformer Fundamentals

> :memo: **NOTE** This post is still in active drafting stage. I am working on adding diagrams, figures and code snippets which will be added soon.

## Introduction

**Key Points about Transformers in Natural Language Models:**

1. **Introduction to Transformers**: Transformers are a fundamental neural network architecture used in most large language models today. They were first introduced in 2017 with the BERT paper and have since become the basis for various language models, including GPT.
2. **Standardized Building Blocks**: Transformers brought standardization to neural network architectures for natural language tasks. Before Transformers, there was a wide array of deep learning model designs, but Transformers established a common building block for these models.
3. **Powerful Architecture**: Transformers have proven to be a powerful architecture for natural language understanding and generation. They excel at capturing different interactions within input data, making them versatile for various NLP tasks.
4. **Stackable and Adaptable**: Transformers can be stacked to different depths, allowing the creation of models with varying complexities and capabilities. Their adaptability and extensibility have made them a preferred choice for many NLP applications.
5. **Focus on Training Techniques**: With the standardization of the Transformer architecture, the focus in NLP research has shifted towards innovative training techniques, data generation, and other aspects of model development.
6. **Persistent Relevance**: Despite the evolution of NLP models, the fundamental building blocks of Transformers have remained largely unchanged. Understanding the architecture in detail is crucial for interpreting model behavior and evaluating new advancements.

Transformers have revolutionized the field of natural language processing and have become a foundational technology in the development of large language models like GPT. Understanding the principles and inner workings of Transformers is essential for anyone working with modern NLP models.

## **The Transformer Block**

**Key Points from the Explanation of Transformer Blocks and Architecture:**

1. **Predicting the Next Token**: Transformers, like other language models, aim to predict the next word or token in a sequence. They achieve this by processing input tokens, transforming the information within the sequence, and using it to predict the next token.
2. **Word Embeddings**: Input tokens are converted into word embeddings, resulting in a series of word vectors. These word embeddings serve as the starting point for further processing.
3. **Transformer Block**: The core component of a Transformer is the Transformer Block. It enriches each token in the sequence with contextual information through attention mechanisms and transformations using neural networks.
    
    ![Screenshot 2023-09-10 at 2.20.36 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.20.36_AM.png)
    
4. **Sequence Enrichment**: Within a Transformer Block, tokens are enriched with contextual information using attention mechanisms. This involves measuring the importance of each word relative to others in the sequence.
    
    ![Screenshot 2023-09-10 at 2.21.54 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.21.54_AM.png)
    
5. **Position-Wise Feed Forward Network**: A key element in Transformer Blocks is the position-wise feed-forward neural network. It operates on each token individually, transforming them into the appropriate format for subsequent blocks.
    
    ![Screenshot 2023-09-10 at 2.22.39 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.22.39_AM.png)
    
6. **Residual Connections and Layer Normalization**: Residual connections ensure smooth gradient flow during training and preserve the input sequence's structural information. Layer normalization aids in training stability, especially in models with numerous layers.
    
    ![Screenshot 2023-09-10 at 2.24.03 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.24.03_AM.png)
    
7. **Input and Output**: The input to a Transformer consists of word embeddings with positional encodings. These are processed through multiple Transformer Blocks to create enriched vectors. At the output, a linear neural network combined with softmax predicts the next token or classifies sequences for specific tasks.
    
    ![Screenshot 2023-09-10 at 2.24.39 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.24.39_AM.png)
    
8. **Variations of Transformer Models**: Transformer architecture can be used in various ways, including encoder models (for processing input), decoder models (for generating the next token), and encoder-decoder models (for transforming one sequence into another, such as translation tasks).

Understanding these fundamental components of Transformers is crucial for grasping how they function and how they can be adapted for different natural language processing tasks.

## **Transformer Architectures**

**Key Points on Transformer Architecture and Variations:**

1. **Transformer Family Tree**: Transformers can be categorized into three main types: encoder-only models, decoder-only models, and encoder-decoder models. Each type is suited for different language processing tasks.
    
    ![Screenshot 2023-09-10 at 2.27.02 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.27.02_AM.png)
    
2. **Encoder-Decoder Models**: The encoder-decoder approach, introduced in the original Transformer paper, is used for tasks like machine translation. The encoder processes the input sequence, and the decoder generates the target sequence, using cross-attention to relate source and target languages.
    
    ![Screenshot 2023-09-10 at 2.27.30 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.27.30_AM.png)
    
3. **Encoder Models**: BERT (Bidirectional Encoder Representations from Transformers) is an example of an encoder model. It uses bidirectional context and masked word prediction during training. BERT is versatile and excels in various NLP tasks like question answering and named entity recognition.
    
    ![Screenshot 2023-09-10 at 2.29.24 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.29.24_AM.png)
    
4. **Decoder Models**: GPT (Generative Pre-trained Transformer) is a well-known decoder-only model. These models generate text and predict the next word in a sequence. They are used in applications like ChatGPT and language generation.
    
    ![Screenshot 2023-09-10 at 2.30.26 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.30.26_AM.png)
    
5. **Important Variables**:
    - **Vocabulary Size**: The number of tokens the Transformer is trained on.
    - **Embedding/Model Size**: The dimensions of word embeddings, influencing model size and parameters.
    - **Sequence/Context Length**: Determines the maximum sequence length the Transformer can handle.
    - **Number of Attention Heads**: Affects multi-attention mechanisms.
    - **Intermediate Feed-Forward Network Size**: Pertains to hidden layers within the feed-forward network.
    - **Number of Layers**: The count of Transformer blocks in the model.
    - **Batch Size**: The number of samples processed in one forward/backward pass during training.
    - **Training Data**: Transformers are trained on vast amounts of data, involving millions, billions, or trillions of tokens.
        
        ![Screenshot 2023-09-10 at 2.31.23 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.31.23_AM.png)
        

Understanding these variables is crucial for configuring and working with Transformer models effectively. Each type of Transformer is suited for specific natural language processing tasks, and their versatility has led to significant advancements in the field.

## **Attention Mechanism**

**Key Points on Attention Mechanism:**

1. **Components of Attention**: The attention mechanism in Transformers involves three vector families: query, key, and value vectors.
    
    ![Screenshot 2023-09-10 at 2.35.05 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.35.05_AM.png)
    
    ![Screenshot 2023-09-10 at 2.36.18 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.36.18_AM.png)
    
2. **Query Vector**: The query vector represents the current token or element of focus.
3. **Key Vectors**: Key vectors are derived from all the tokens or elements in the input sequence.
4. **Value Vectors**: Value vectors correspond to the information associated with each token or element.
5. **Matrix Multiplication**: The query vector is created by multiplying the input vector with a query matrix, and the same process is applied to generate key and value vectors.
6. **Attention Calculation**: Attention scores are computed by taking the softmax of the scaled dot product of query and key vectors. These scores indicate the importance or relevance of key vectors to the query.
7. **Scaled Dot Product**: The attention score for each key vector is obtained by multiplying the query vector with the key vector and then applying softmax to normalize the scores.
8. **Output Vector**: The output vector of the attention mechanism is a weighted combination of value vectors, where the attention scores determine the weights.
9. **Attention as Filing and Lookup**: Conceptually, attention can be thought of as a filing and lookup system. The query (current token) searches through files (key vectors) to find relevant information (value vectors) based on the computed attention weights.

Understanding attention is crucial as it is a fundamental building block of the Transformer architecture. It enables the model to focus on different parts of the input sequence when processing each token, allowing for complex interactions and contextual understanding.

## **Base/Foundation Models**

**Key Points on Building and Training Foundation Transformers:**

1. **Foundation or Base Models**: These terms refer to large language models trained from randomized weights to predict the next word. They capture syntax and some semantic understanding of language but may not exhibit task-specific behavior.
    
    ![Screenshot 2023-09-10 at 2.39.49 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.39.49_AM.png)
    
2. **Fine-Tuning**: For task-specific performance, fine-tuning a base model is typically required. Fine-tuning involves training the model on a smaller dataset related to the target task.
3. **Model Architecture**: Decide whether you need an encoder, decoder, or both, based on the tasks you want the model to perform.
    
    ![Screenshot 2023-09-10 at 2.41.23 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.41.23_AM.png)
    
4. **Number of Layers**: Determine the number of layers or blocks in your model, which affects the depth and capacity of the model.
5. **Context Size**: Consider the context length the model should handle, as it influences the amount of computation required.
6. **Data Collection and Preparation**: Gather appropriate data, including publicly available datasets like Pile and proprietary or curated datasets relevant to your use case.
    
    
    ![Screenshot 2023-09-10 at 2.41.51 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.41.51_AM.png)
    
7. **Training Process**: Train the model, which can take several weeks or months, and may require a significant number of GPUs. Common loss functions like cross-entropy and optimizers like AdamW are used.
    
    ![Screenshot 2023-09-10 at 2.42.24 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.42.24_AM.png)
    
8. **Alignment Problem**: Address issues related to model alignment, such as accuracy, toxicity, biases, and hallucinations.
    
    ![Screenshot 2023-09-10 at 2.42.51 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_2.42.51_AM.png)
    
9. **Fine-Tuning Methods**: Explore fine-tuning techniques to adapt the model to specific tasks efficiently.
10. **History of GPT**: Consider the evolution of GPT models from GPT-1 to GPT-4 and their impact on natural language understanding and generation.

Building and training foundation Transformers require careful consideration of model architecture, data, and fine-tuning strategies to achieve desired performance on specific tasks.

## **Generative Pre-trained Transformer (GPT)**

**Key Points from Module 1 - Transformers and GPT Evolution (Expanded Summary):**

1. **Introduction to Transformers**: Transformers are a fundamental architecture in natural language processing, known for their versatility in various NLP tasks.
    
    ![Screenshot 2023-09-10 at 8.32.56 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.32.56_AM.png)
    
2. **Transformer Blocks**: Transformers consist of Transformer blocks, which incorporate self-attention mechanisms and position-wise feed-forward neural networks to process input data.
3. **Types of Transformer Architectures**:
    - **Encoder Models**: Focus on encoding input data.
    - **Decoder Models**: Focus on generating output data.
    - **Encoder-Decoder Models**: Combine both encoding and decoding capabilities.
4. **GPT (Generative Pre-trained Transformer)**:
    - GPT is a family of generative models based on Transformer architecture.
    - It is pre-trained on a large corpus of text data and can be fine-tuned for various NLP tasks.
5. **Evolution of GPT**:
    - **GPT-1**: The original model with 12 Transformer blocks.
    - **GPT-2**: Increased model size with 1024-dimensional embeddings and more Transformer blocks.
    - **GPT-3**: Further scaled up with 175 billion parameters, excelling in few-shot and zero-shot learning.
    - **GPT-4**: A speculated model with even more parameters, potentially using a mixture of experts approach.
        
        ![Screenshot 2023-09-10 at 8.33.07 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.33.07_AM.png)
        
        ![Screenshot 2023-09-10 at 8.34.13 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.34.13_AM.png)
        
        ![Screenshot 2023-09-10 at 8.34.49 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.34.49_AM.png)
        
    
    **Why GPT Kept Getting Bigger**:
    
    - Increasing the number of layers in GPT models allows the attention mechanism to focus on more aspects of the input text. Each layer captures different levels of detail, from basic syntax to complex long-range dependencies.
    - As models scale up, they can process larger contexts, making them more effective at understanding and generating longer pieces of text.
    - The expansion of the model dimensions (e.g., embedding sizes) also contributes to increased capacity, although not as dramatically as the number of layers.
    - The number of attention heads per layer doesn't scale as significantly, but the total number of layers and model size plays a crucial role in the neural network's complexity and capacity.
        
        ![Screenshot 2023-09-10 at 8.35.26 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.35.26_AM.png)
        
        ![Screenshot 2023-09-10 at 8.43.07 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.43.07_AM.png)
        
6. **Data for Training**:
    - GPT models rely on large-scale datasets. GPT-1 used Book Corpus, GPT-2 used WebText, GPT-3 used WebText2, and newer models may require even larger and diverse datasets.
        
        ![Screenshot 2023-09-10 at 8.43.30 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.43.30_AM.png)
        
7. **Scaling Transformers**:
    - As models evolve, they tend to have more layers and increased capacity to handle larger contexts and more complex tasks.
    - Attention mechanisms become capable of capturing more aspects of text as layers increase.
8. **Choosing Transformer Models**:
    - Choose the appropriate Transformer architecture based on the specific NLP task and available resources.
    - Smaller models like BERT or T5 may be more suitable for some tasks and resource constraints.
        
        ![Screenshot 2023-09-10 at 8.44.54 AM.png](Transformer%20Architecture%20Attention%20&%20Transformer%20F%2029a57c10ace94f888b43eefe44c337d2/Screenshot_2023-09-10_at_8.44.54_AM.png)
        

Module 1 provided an in-depth understanding of Transformers, their evolution into GPT models, and the reasons behind the continuous increase in model size to improve their capabilities in processing and generating natural language text.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)
- [A Mathematical View of Attention Models in Deep Learning](https://people.tamu.edu/~sji/classes/attn.pdf)
- [What Is a Transformer Model?](https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/)
- [Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b)
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [BookCorpus](https://en.wikipedia.org/wiki/BookCorpus) and [Gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset) (webText)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Large Language Models and the Reverse Turing Test](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10177005/)


# 2. Efficient Fine Tuning

## Introduction

Title: Efficient Fine-Tuning of Language Models

1. **Introduction to Fine-Tuning**
    - Fine-tuning is the process of updating the weights of a pre-trained deep model with new data.
    - It's a powerful technique to adapt pre-trained models to specific tasks or domains.
    - Language models are ideal candidates for fine-tuning due to their expensive pre-training process and general knowledge.
2. **Benefits of Fine-Tuning**
    - Fine-tuning allows models to acquire domain-specific knowledge.
    - It tailors models for specific tasks like sentiment analysis, translation, or structured data extraction.
3. **General Overview of Fine-Tuning**
    - Fine-tuning involves updating a pre-trained model with new data.
    - It helps the model adapt to the target task or domain.
    - Fine-tuning is essential for achieving high-quality results in many applications.
4. **Parameter-Efficient Fine-Tuning**
    - Recent techniques focus on efficient fine-tuning, reducing the cost of updates.
    - These methods avoid updating all model parameters uniformly.
    - They are cost-effective and useful for quickly assessing a model's suitability for a task.
5. **Applications and Examples**
    - Fine-tuning improves model performance even with small, inexpensive models.
    - Examples include models trained to use software tools and APIs, where fine-tuning enhances responses significantly.
6. **The Importance of Fine-Tuning**
    - Fine-tuning is the primary approach for achieving high-quality models in various domains.
    - Even smaller models with fine-tuning can outperform larger models out of the box.
    - Fine-tuning offers the potential to significantly boost model performance in any domain.
7. **Getting Started**
    - Fine-tuning is a crucial tool for ML engineers working with language models.
    - It is essential for customizing models to specific tasks and domains.
    - Provides an opportunity to test the feasibility of model adaptation before resource-intensive fine-tuning.

In this module, we will delve deeper into the concepts of fine-tuning, efficient fine-tuning techniques, and practical applications to help you harness the full potential of language models for your specific needs.

Title: Understanding Fine-Tuning and Parameter-Efficient Approaches

1. **Introduction to Fine-Tuning and Transfer Learning**
    - Fine-tuning involves updating a pre-trained model with new data, making it adaptable to specific tasks.
    - Transfer learning is the broader concept, and fine-tuning falls under it, focusing on training the model further.
    - Fine-tuning is essential due to the high cost of training large language models (LLMs) from scratch.
        
        ![Screenshot 2023-09-10 at 9.10.53 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.10.53_AM.png)
        
2. **Leveraging Foundation Models**
    - Foundation models, such as T5, GPT-4, and BloombergGPT, can be used as-is or for feature extraction.
    - Fine-tuning involves updating model layers, adding new layers, or training on task-specific data.
        
        ![Screenshot 2023-09-10 at 9.11.34 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.11.34_AM.png)
        
3. **Why Fine-Tuning?**
    - Fine-tuning improves model performance for specific tasks.
    - It adapts models to unique styles, vocabularies, and regulatory requirements.
    - Jeremy Howard and Sebastian Ruder introduced fine-tuning techniques for NLP tasks in 2018.
        
        ![Screenshot 2023-09-10 at 9.12.14 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.12.14_AM.png)
        
4. **Full Fine-Tuning and Its Challenges**
    - Full fine-tuning updates all model weights, producing one model per task.
    - It leads to high disk space requirements and the issue of catastrophic forgetting.
    - Deployment may require numerous copies of the same foundation model.
        
        ![Screenshot 2023-09-10 at 9.13.10 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.13.10_AM.png)
        
        ![Screenshot 2023-09-10 at 9.13.26 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.13.26_AM.png)
        
5. **Parameter-Efficient Fine-Tuning (PEFT)**
    - PEFT methods aim to achieve efficient training and deployment without updating all model weights.
    - It overcomes the challenges of full fine-tuning, such as disk space and catastrophic forgetting.
        
        ![Screenshot 2023-09-10 at 9.14.15 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.14.15_AM.png)
        
6. **Examples of Fine-Tuned Models**
    - GOAT, a fine-tuned model for arithmetic tasks, outperforms few-shot learning and other large models.
    - FLAN and Dolly are fine-tuned, multitask LLMs with different foundation models and applications.
        
        ![Screenshot 2023-09-10 at 9.15.11 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.15.11_AM.png)
        
        ![Screenshot 2023-09-10 at 9.15.32 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.15.32_AM.png)
        
7. **Goals of Fine-Tuned Models**
    - Efficient training, serving, and storage are crucial for organizations with budget constraints.
    - Parameter-efficient approaches enable multitask serving without creating multiple models.
        
        ![Screenshot 2023-09-10 at 9.15.55 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.15.55_AM.png)
        
        ![Screenshot 2023-09-10 at 9.16.15 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.16.15_AM.png)
        

In this module, we've explored the concepts of fine-tuning, its importance, and the challenges it poses. We've also introduced parameter-efficient fine-tuning as a solution and provided examples of successful fine-tuned models. The goal is to achieve efficient model adaptation and deployment while delivering high-quality results for specific tasks and domains.

## **Parameter-efficient fine-tuning (PEFT) and Soft Prompt**

**Title: Parameter-Efficient Fine-Tuning (PEFT) and Soft Prompts**

**Introduction to PEFT Categories**

- PEFT, or Parameter-Efficient Fine-Tuning, aims to optimize storage, memory, computation, and performance.
- There are three categories of PEFT: additive, selective, and re-parameterization. This course focuses on additive and re-parameterization methods.
    
    ![Screenshot 2023-09-10 at 9.17.40 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.17.40_AM.png)
    

**Additive Methods in PEFT**

- Additive methods involve adding new trainable layers to the model during fine-tuning.
- Only the weights of these new layers are updated, while the foundation model weights remain frozen.
- It enhances the core Transformer block, particularly query, key, and value weight matrices.
    
    ![Screenshot 2023-09-10 at 9.18.06 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.18.06_AM.png)
    

**Soft Prompts: Removing Manual Prompt Engineering**

- Soft prompts, also known as virtual tokens, are introduced to simplify prompt engineering.
- Soft prompts have the same dimension as input embeddings and are concatenated during fine-tuning.
    
    ![Screenshot 2023-09-10 at 9.18.22 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.18.22_AM.png)
    
- Unlike manual discrete prompts, soft prompts are learned by the model through backpropagation.
- Virtual tokens can be initialized randomly or with minimal discrete prompts, both found to be effective.
    
    ![Screenshot 2023-09-10 at 9.19.43 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.19.43_AM.png)
    

**Benefits of Soft Prompt Tuning**

- Soft prompt tuning eliminates the need for manual prompt engineering.
- It allows for a flexible number of examples in the model context.
- Backpropagation helps the model find the optimal representation of task-specific virtual prompts.
- There's no need for multiple model copies, enabling multitask serving.
- Resilient to domain shift, as it prevents overfitting by keeping foundation model weights frozen.
    
    ![Screenshot 2023-09-10 at 9.20.28 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.20.28_AM.png)
    
    ![Screenshot 2023-09-10 at 9.20.54 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.20.54_AM.png)
    
    ![Screenshot 2023-09-10 at 9.21.33 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.21.33_AM.png)
    
    ![Screenshot 2023-09-10 at 9.22.07 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.22.07_AM.png)
    
    ![Screenshot 2023-09-10 at 9.22.42 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.22.42_AM.png)
    
    ![Screenshot 2023-09-10 at 9.23.01 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.23.01_AM.png)
    

**Disadvantages of Soft Prompt Tuning**

- Soft prompts are less interpretable than discrete prompts.
- Performance with soft prompt tuning can be unstable.
- Virtual tokens' meaning is often estimated through distance metrics.
    
    ![Screenshot 2023-09-10 at 9.23.47 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.23.47_AM.png)
    

**Prefix Tuning**

- Prefix tuning is a variation of soft prompt tuning where task-specific prompts are added as prefix layers to each Transformer block.
    
    ![Screenshot 2023-09-10 at 9.23.56 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.23.56_AM.png)
    

In summary, parameter-efficient fine-tuning (PEFT) encompasses strategies like soft prompts and prefix tuning. Soft prompts eliminate manual prompt engineering, offer flexibility, and improve multitask serving but can have unstable performance and limited interpretability. Prefix tuning is similar but introduces task-specific prefixes to each Transformer block. These approaches enable efficient fine-tuning while minimizing the need for full model retraining.

## **Re-parameterization: LoRA**

**Title: LoRA - Low-Rank Weight Matrix Decomposition in PEFT**

**Introduction to LoRA**

- LoRA (Low-Rank Representation) is a re-parameterization method in PEFT.
- It leverages low-rank representation to minimize the number of trainable parameters.
- LoRA decomposes the weight_delta matrix into two low-rank matrices (W_a and W_b).
    
    ![Screenshot 2023-09-10 at 9.25.42 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.25.42_AM.png)
    

**Matrix Rank Basics**

- Matrix rank is the maximum number of linearly independent columns in a matrix.
    
    ![Screenshot 2023-09-10 at 9.25.54 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.25.54_AM.png)
    
- Full-rank matrices have no redundant rows or columns.
- LoRA exploits the lower rank of the attention weight matrix compared to the weight_delta matrix.

**How Weight Matrix Decomposition Works**

- Weight_delta matrix (e.g., 100x100) can be decomposed into two matrices (W_a: 100x2 and W_b: 2x100).
- The decomposed matrices, when multiplied, retain the shape of the original weight_delta matrix.
- This decomposition significantly reduces the number of trainable parameters.
    
    ![Screenshot 2023-09-10 at 9.26.56 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.26.56_AM.png)
    
- LoRA matches fine-tuning performance while reducing the number of parameters.
- It can achieve a 96% reduction in the number of trainable parameters.
- LoRA maintains or outperforms fine-tuning with a fraction of the parameters.
    
    ![Screenshot 2023-09-10 at 9.27.12 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.27.12_AM.png)
    

**Determining Rank Sizes**

- The rank size can be treated as a hyperparameter to search over.
- Different rank sizes often produce similar validation accuracies.
- Smaller rank sizes may not work well for tasks significantly different from the base model's training.
    
    ![Screenshot 2023-09-10 at 9.27.21 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.27.21_AM.png)
    

**Advantages of LoRA**

- Locks up or freezes the majority of model weights.
- Allows for sharing or reusing the same foundation model.
- Improves training efficiency by reducing the computation of most gradients.
- Adds no additional serving latency since W_a and W_b can be merged.
- Can be combined with other PEFT methods, although not straightforward.
    
    ![Screenshot 2023-09-10 at 9.27.41 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.27.41_AM.png)
    

**Limitations of LoRA**

- Dynamic selection of weight matrices at serving time can introduce serving latency.
- Open research questions about decomposing other matrices and further reducing parameters.
- Newer PEFT techniques like IA3 can achieve even greater parameter reduction.
    
    ![Screenshot 2023-09-10 at 9.28.06 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.28.06_AM.png)
    

In summary, LoRA is a powerful technique that reduces the number of trainable parameters in fine-tuning while maintaining or even improving model performance. It leverages low-rank representation to decompose weight matrices, leading to substantial parameter efficiency gains. However, there are challenges in dynamic weight matrix selection and ongoing research to further optimize parameter-efficient fine-tuning.

## PEFT Limitations

**Title: Common Limitations of PEFT Techniques**

**Model Performance Challenges**

- PEFT techniques, including prompt tuning and LoRA, may not consistently outperform full fine-tuning due to sensitivity to hyperparameters.
- It's unclear why specific weight matrices, like attention weights, are chosen for PEFT, and the suitability of PEFT for different scenarios is not well-defined.
- PEFT focuses on storage reduction, but other aspects like memory footprint and computational efficiency need attention.

**Memory and Storage Limitations**

- While PEFT aims to reduce storage by avoiding multiple copies of the same foundation model, it doesn't address all memory and storage challenges.
- New optimizers like Lomo have been introduced to reduce memory footprint significantly.
- PEFT doesn't eliminate the need to store a single copy of the massive foundation model.
    
    ![Screenshot 2023-09-10 at 9.29.37 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.29.37_AM.png)
    

**Compute Challenges**

- PEFT doesn't always make serving or inference more computationally efficient.
- It retains the time complexity of training, requiring full forward and backward passes.
    
    ![Screenshot 2023-09-10 at 9.30.11 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.30.11_AM.png)
    

In summary, PEFT techniques have common limitations related to model performance, memory and storage, and computational efficiency. While they offer benefits in reducing parameters and storage, there are challenges in achieving stable performance and addressing all aspects of resource efficiency. Researchers are exploring new approaches to tackle these limitations.

## **Data preparation best practices**

**Title: Best Practices for Data Preparation in Fine-Tuning**

**Importance of Data Quality**

- High-performing language models (LLMs) often rely on intentionally curated, diverse, and high-quality training data.
- Examples include MPT, Llama, GPT-Neo, and GPT-J, which use large and high-quality datasets like C4 and The Pile.
- BloombergGPT achieved impressive results by curating its own dataset of financial documents.
    
    ![Screenshot 2023-09-10 at 9.31.36 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.31.36_AM.png)
    

**Data Quantity vs. Quality**

- Research suggests that even a couple of hundred high-quality labeled examples can be sufficient for some tasks.
- Increasing the quantity of data can lead to linear improvements in model performance.
- Synthetic data generation methods like synonym replacement, word deletion, word position swapping, and introducing intentional typos can help increase data diversity.
    
    ![Screenshot 2023-09-10 at 9.31.59 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.31.59_AM.png)
    
    ![Screenshot 2023-09-10 at 9.32.49 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.32.49_AM.png)
    

**Data Formatting and Tips**

- Use proper delimiters or separators to distinguish prompts and completions.
- Detailed instructions may not be necessary; clear separation between prompts and completions suffices.
- Ensure data quality by manually verifying, removing undesired content (e.g., offensive or confidential information), and choosing foundation models trained on relevant tasks.
    
    ![Screenshot 2023-09-10 at 9.33.08 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.33.08_AM.png)
    

**Imitation Models and Downstream Tasks**

- Downstream models tend to mimic the style rather than the content of LLMs.
- Choose a foundation model that aligns with your downstream tasks and data for effective fine-tuning.
    
    ![Screenshot 2023-09-10 at 9.33.29 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.33.29_AM.png)
    

**Recap of Module**

- Fine-tuning offers excellent results but can be computationally expensive.
- Parameter-efficient fine-tuning (PEFT) methods like prompt tuning (soft prompt) and LoRA (re-parameterization) reduce trainable parameters.
- Data quality and diversity are critical for successful fine-tuning.
    
    ![Screenshot 2023-09-10 at 9.34.05 AM.png](Efficient%20Fine%20Tuning%2005efa6f8f58d49feaa1ca3cf656d390b/Screenshot_2023-09-10_at_9.34.05_AM.png)
    

In conclusion, data preparation is a crucial step in fine-tuning, requiring high-quality and diverse datasets. Curated datasets and synthetic data generation can enhance model performance. It's essential to format data correctly and choose foundation models that match the downstream tasks.

## References

1. [What’s in Colossal Clean Common Crawl (C4) dataset](https://www.washingtonpost.com/technology/interactive/2023/ai-chatbot-learning/)
2. [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239)
    - LaMDA is a family of dialog models. The authors found fine-tuning the model with a classifier with some crowdsourced annotated data can improve model safety
3. [Gorilla: Large Language Model Connected with Massive APIs](https://gorilla.cs.berkeley.edu/)
4. [Interpretable Soft Prompts](https://learnprompting.org/docs/trainable/discretized)
    - By performing prompt tuning on initialized text – e.g. “classify this sentiment” – the resulting prompt embeddings might become nonsensical. But this nonsensical prompt can give better performance on the task
5. [Continual Domain-Adaptive Pre-training](https://arxiv.org/pdf/2302.03241.pdf)
6. [Foundation Models for Decision Making: Problems, Methods, and Opportunities](https://arxiv.org/pdf/2303.04129.pdf)
7. [“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426/?utm_source=substack&utm_medium=email)
    - Using a simple compressor, like gzip with a KNN classifier, can outperform BERT on text classification. The method also performs well in few-shot settings.
8. [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)
9. [Ahead of AI: LLM Tuning & Dataset Perspectives](https://magazine.sebastianraschka.com/p/ahead-of-ai-9-llm-tuning-and-dataset)
10. [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/pdf/2306.04751.pdf)
11. [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701)
    - More data for fine-tuning LLMs is not necessarily better. AlpaGasus used 9k high-quality data out of the original 52k Alpaca dataset and it performed the original Alpaca-7B model.

# 3. Deployment and Hardware Considerations

## Introduction

**Title: Deployment Optimizations for Improved Model Size and Speed**

**Overview**

- Module 3 focuses on techniques to optimize model size and speed for language models (LLMs) after training.
- Deep learning offers methods to reduce computational cost while maintaining model quality.
- Techniques include compression, quantization, pruning, model cascading, mixture of experts, and meta-optimizations.
- These optimizations are essential for interactive LLM applications and high-volume, cost-sensitive deployments.

**Key Techniques**

1. **Compression and Quantization**
    - Reduce model size by compressing weights.
    - Quantize weights to lower numerical precision (e.g., from float32 to int8) to reduce memory and computation requirements.
2. **Pruning**
    - Remove less important weights or neurons to simplify the model.
    - Pruned models have fewer parameters, which results in faster inference.
3. **Model Cascading**
    - Forward inputs to different models based on complexity.
    - Send simple inputs to a cheaper model and complex inputs to a more expensive one.
    - Optimize latency and cost while maintaining quality.
4. **Mixture of Experts**
    - Create models composed of multiple submodels (experts).
    - Route requests to different experts based on input characteristics.
    - Improve quality and efficiency for various inputs.
5. **Meta-Optimizations**
    - Explore techniques for combining models and managing their interactions.
    - Achieve lower costs and reduced latency for high-volume applications.

**Application Areas**

- Optimizations are crucial for interactive LLM applications where low latency is critical.
- High-volume applications, such as ad placement, require cost-effective deployments.
- ML engineers must balance performance, cost, and quality for successful production deployments.

In summary, this module explores various techniques and strategies for optimizing model size and speed in language models, making them more efficient and cost-effective in real-world applications. These optimizations are essential for both latency-sensitive and high-volume use cases.

**Title: Optimization Strategies for Large Language Models**

**Overview**

- Module 3 focuses on addressing challenges posed by large language models (LLMs), including memory limitations and computational costs.
- It explores various techniques and design choices to optimize LLMs for efficient use of compute resources and improved performance.

**Key Considerations**

1. **Memory Constraints**
    - LLMs with hundreds of billions of parameters often exceed the memory capacity of consumer or enterprise GPUs.
    - Larger models tend to perform better in terms of accuracy and task flexibility but suffer from slower inference and reduced updateability.
2. **Performance vs. Efficiency**
    - Developers face a trade-off between model size and speed when choosing LLMs.
    - Smaller models are faster but may sacrifice quality, while larger models offer higher quality but demand more computational resources.
        
        ![Screenshot 2023-09-10 at 11.20.18 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.20.18_AM.png)
        

**Optimization Strategies**

1. **Model Design Optimization**
    - Examine components within the Transformer architecture for potential improvements to alleviate memory constraints.
    - Seek ways to enhance LLM performance without significantly increasing model size.
2. **Mixture of Experts**
    - Utilize a combination of multiple LLMs or submodels to address different tasks and inputs.
    - Optimize cost and performance by routing requests to the most suitable expert models.
3. **Quantization**
    - Convert large LLMs into smaller versions with lower numerical precision (e.g., float32 to int8) while maintaining acceptable performance.
    - Reduce memory and computational demands for deploying LLMs.

**Benefits of Optimization**

- Efficiently manage computational resources to strike a balance between model size and performance.
- Overcome memory limitations to fit LLMs on available hardware.
- Improve model updateability and reduce inference costs.
- Enhance LLM quality and versatility while minimizing computational overhead.

In summary, Module 3 explores strategies to optimize large language models, addressing memory constraints, computational costs, and the trade-off between model size and performance. Developers can make informed design choices to leverage LLMs effectively and efficiently in various applications.

## [Improving Learning Efficiency](https://learning.edx.org/course/course-v1:Databricks+LLM102x+2T2023/block-v1:Databricks+LLM102x+2T2023+type@sequential+block@4435f45f4221485c9ad0f60062e01bdb)

**Title: Optimizing Large Language Models for Context Length and Attention Mechanism**

**Overview**

- This module delves into the challenges related to large language models (LLMs), specifically regarding context length and the attention mechanism.
- It discusses how increasing context length in LLMs impacts computational complexity and explores innovative solutions for addressing these issues.

**Key Considerations**

1. **Context Length Challenges**
    - Context length refers to the amount of information provided to an LLM for interpretation.
    - Larger context lengths improve LLM performance, but they introduce computational challenges.
    - Attention mechanism plays a crucial role in handling context length.
2. **Computational Complexity**
    - Increasing context length leads to various computational challenges:
        - Linear increase in computations for input values and position-wise feed-forward networks.
        - Quadratic increase in calculating attention scores due to the attention mechanism.
        - Deterioration in model performance when trained on shorter context lengths and tested on longer ones.
            
            ![Screenshot 2023-09-10 at 11.22.06 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.22.06_AM.png)
            

**Innovations in Attention Mechanism**

1. **ALiBi (Autoregressive Linear Basis)**
    - It uses linear scaling of query-key pairs to provide fading importance from the current token to earlier tokens.
    - ALiBi enables training on shorter contexts and expanding to longer contexts during inference.
        - ALiBi introduced a method to handle varying context lengths efficiently.
        
        ![Screenshot 2023-09-10 at 11.23.45 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.23.45_AM.png)
        
2. **FlashAttention**
    - FlashAttention is an emerging approach that avoids materializing large attention matrices.
    - It enables matrix-free operations by processing individual variables one by one.
    - By avoiding SRAM overload, it significantly accelerates calculations, especially for longer contexts.
        
        ![Screenshot 2023-09-10 at 11.25.06 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.25.06_AM.png)
        
3. **Multi-Headed Attention**
    - Multi-Headed Attention splits attention matrices into multiple heads.
    - Each head focuses on different parts of the text, providing enriched results.
    - However, this approach is slower due to multiple calculations.
4. **Grouped Query Attention**
    - Grouped Query Attention balances speed and accuracy by using different query vectors for key vectors.
    - It maintains the multi-headed, multi-focus aspects of attention while achieving faster computations.
    - LLaMa2 uses this
        
        ![Screenshot 2023-09-10 at 11.26.01 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.26.01_AM.png)
        

**Benefits of Innovations**

- Innovations in attention mechanisms and context handling allow LLMs to accommodate longer contexts and improve performance.
- Techniques like ALiBi, FlashAttention, and grouped query attention enhance LLM efficiency and maintain high-quality outputs.
- These advancements are crucial for working with increasingly larger models and tackling real-world applications.

In summary, this module explores challenges related to context length and attention mechanisms in LLMs. It highlights innovative solutions that enable efficient handling of longer contexts and faster computations, ultimately enhancing the capabilities of large language models.

## **Improving Model Footprint**

**Title: Quantization and Parameter Efficiency in Large Language Models**

**Overview**

- This module focuses on optimizing large language models (LLMs) by addressing the storage and processing of numerical values, specifically through quantization.
- Quantization involves converting floating-point numbers into integers to save memory and improve efficiency in LLMs.
- The module also discusses innovations like bf16 (brain float 16) and introduces quantized LoRA (QLoRA) as a parameter-efficient fine-tuning method.

**Key Concepts**

1. **Quantization Basics**
    - Traditional storage of numbers uses fp16 or fp32 standards, which allocate bits for exponents and mantissas.
    - BF16 (Brain Float) is an alternative format with a larger exponent range and a smaller mantissa.
    - Quantization converts floating-point numbers into integers by determining a quantization factor based on the largest number in the data.
        
        ![Screenshot 2023-09-10 at 11.29.30 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.29.30_AM.png)
        
2. **Quantization Process**
    - To quantize data, the largest number is identified, and a quantization factor is calculated.
    - All numbers in the data are multiplied by the quantization factor and rounded to obtain integers.
    - This process allows efficient storage of data using integer values.
        
        ![Screenshot 2023-09-10 at 11.30.48 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.30.48_AM.png)
        
3. **Quantization Precision**
    - Quantization introduces errors due to the loss of precision.
    - The degree of precision loss depends on the quantization factor and the original data distribution.
    - Despite precision loss, quantization can often represent data accurately for certain applications like deep learning.
4. **Quantized LoRA (QLoRA)**
    - QLoRA is an enhancement of LoRA, a parameter-efficient fine-tuning method.
    - QLoRA quantizes the Transformer to a 4-bit version and reduces the adapter size to a 16-bit representation.
    - It leverages system memory to store the optimizer state and efficiently works on smaller hardware.
        
        ![Screenshot 2023-09-10 at 11.32.02 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.32.02_AM.png)
        
5. **Quantization Innovations**
    - Innovations in quantization include 8-bit, 4-bit, and 2-bit optimizers to further reduce memory requirements.
    - Quantization is a crucial technique for accommodating large language models on consumer and enterprise hardware.

**Benefits of Quantization**

- Quantization allows LLMs to be stored and processed more efficiently by using integer values.
- It reduces memory consumption and enables the deployment of large models on resource-constrained hardware.
- Quantized LoRA (QLoRA) is a popular parameter-efficient fine-tuning method that leverages quantization for efficient inference.

In summary, this module emphasizes the importance of quantization in optimizing LLMs for memory efficiency. It introduces quantization techniques, such as bf16 and QLoRA, and highlights their benefits in handling large language models on various hardware platforms.

## **Multi-LLM Inferencing**

**Title: Leveraging Multiple Large Language Models for Inference and Training**

**Overview**

- This section explores strategies for utilizing multiple large language models (LLMs) for various purposes, including inference and training.
- The primary focus is on the concept of "mixture of experts," where inputs are routed to different experts for handling.
- It introduces the switch Transformer as an example of how this approach can efficiently distribute parameters among experts.

**Key Concepts**

1. **Mixture of Experts**
    - Mixture of experts is a technique that combines multiple versions of a smaller system, each trained for specific tasks.
    - Unlike ensemble methods, mixture of experts uses a router to determine which expert should process a given input.
    - This approach can significantly reduce parameter costs and enable the creation of extremely large models.
2. **Parameter Distribution in LLMs**
    - In LLMs, a significant portion of parameters is allocated to position-wise feed-forward neural networks within Transformer blocks.
    - The switch Transformer introduces multiple experts for these networks, which are trained during the training process.
    - The router decides which expert(s) should process a given sample, allowing for efficient parameter usage.
3. **Scaling to Trillion-Parameter Models**
    - Mixture of experts, as demonstrated by the switch Transformer, enables the creation of trillion-parameter and multi-trillion-parameter models.
    - The approach involves training multiple 100-billion-parameter models simultaneously and routing inputs to the appropriate experts.
        
        ![Screenshot 2023-09-10 at 11.34.27 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.34.27_AM.png)
        
4. **Frugal GPT and LLM Cascades for Inference**
    - Frugal GPT is an example of optimizing inference cost while maintaining accuracy.
    - LLM cascades, as implemented in Frugal GPT, involve passing a prompt to the lowest-performing model first.
    - The perplexity of model outputs is used to determine if higher-quality models are needed for subsequent steps.
    - This cascading approach reduces cost while preserving accuracy.
        
        ![Screenshot 2023-09-10 at 11.36.09 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.36.09_AM.png)
        
5. **Future Directions**
    - Leveraging multiple LLMs is an evolving area of research and industrial applications.
    - Emerging practices like LLM cascades and Frugal GPT demonstrate the potential for optimizing LLM use cases further.
    - Continued research will explore new possibilities for improving the efficiency and effectiveness of LLMs.

**Benefits of Mixture of Experts**

- Mixture of experts allows the efficient utilization of multiple LLMs, reducing parameter costs.
- The switch Transformer demonstrates how multiple experts can be trained simultaneously, enabling the creation of extremely large models.
- Frugal GPT and LLM cascades illustrate cost-effective approaches to LLM inference while maintaining accuracy.

This section highlights the significance of mixture of experts and cascading strategies in optimizing the use of LLMs for various applications, including both training and inference. It showcases innovative approaches to handling the growing scale of large language models.

## **Current Best Practices**

**Title: Best Practices for Leveraging Large Language Models (LLMs)**

**Overview**

- This section provides a comprehensive summary of best practices for effectively utilizing large language models (LLMs) in various applications, including training and inference.
- It covers optimization techniques, hardware considerations, and strategies for achieving desired performance and cost-effectiveness.

**Key Best Practices for Training LLMs**

1. **Incorporate ALiBi for Large Context Lengths**
    - ALiBi allows for training with much larger context lengths than those used during training, enabling LLMs to handle extensive inputs effectively.
2. **Utilize FlashAttention for Efficient Attention Calculation**
    - FlashAttention optimizes attention mechanisms, particularly for long contexts, by avoiding overwhelming GPU memory (SRAM) and facilitating the use of larger context lengths.
3. **Leverage Grouped Query Attention**
    - Grouped query attention helps reduce the compute resources required for attention mechanisms and minimizes the number of parameters needed, improving efficiency.
4. **Consider Mixture of Experts for Scalability**
    - For extremely large LLMs, a mixture of experts approach can efficiently distribute parameters and enable the creation of trillion-parameter models.

**Best Practices for Fine-Tuning and Inference**

1. **Use LoRA and Quantized LoRA**
    - LoRA and quantized LoRA are valuable tools for fine-tuning and inference, helping optimize LLMs for specific tasks and deployment scenarios.
2. **Explore Frugal GPT and LLM Cascades for Cost-Efficient Inference**
    - Frugal GPT and LLM cascades offer cost-effective approaches to LLM inference by cascading models based on performance and perplexity.

**GPU Memory Considerations**

- A general rule of thumb for GPU memory requirements is that doubling the number of parameters roughly corresponds to the GPU memory needed.
- For serving, a 7 billion parameter model may require approximately 14 GB of GPU memory, with larger models demanding more memory.
- Hardware advancements from NVIDIA, AMD, and others may provide improved resources for handling larger LLMs, but availability and cost considerations should be taken into account.
    
    ![Screenshot 2023-09-10 at 11.38.05 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.38.05_AM.png)
    

**Conclusion**

- This module emphasizes the need to optimize LLM deployment due to the rapidly increasing model sizes that exceed current compute capacity.
- Key optimization strategies include adapting attention mechanisms, quantization, and leveraging mixture of experts.
- Hardware advancements are expected to continue, offering more resources for working with larger LLMs.
- The importance of cost-effective inference methods like LLM cascades and Frugal GPT is highlighted.
- The module encourages exploring best practices for deploying and fine-tuning LLMs in practical applications.

This summary outlines the essential best practices and considerations for effectively working with LLMs, addressing both training and deployment challenges while maximizing performance and cost efficiency.

## **Training Large Language Models (LLMs) from Scratch**

**Title: Training Large Language Models (LLMs) from Scratch: Compute, Orchestration, and Model Insights**

**Overview:**
In this module, Abhi, the leader of the NLP team at Databricks (formerly MosaicML), shares insights into training LLMs from scratch. The focus is on infrastructure, orchestration, and the development of open-source models MPT-7B and -30B.

**Key Points:**

**Compute and Orchestration:**

- Training LLMs requires a substantial amount of compute resources, and parallelization across hundreds to thousands of GPUs is essential for efficiency.
    
    ![Screenshot 2023-09-10 at 11.41.49 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.41.49_AM.png)
    
- The MosaicML Cloud, an orchestration and sketching layer, simplifies the execution of large LLM training runs on various GPU clusters. It addresses challenges like multi-node training, automatic run resumption, object store support, and experiment tracking.
    
    ![Screenshot 2023-09-10 at 11.42.14 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.42.14_AM.png)
    

**Scaling and Performance:**

- Internode networking in cloud-based compute clusters offers high bandwidth, enabling near-linear scaling when combined with the right orchestration tools.
- Scaling out effectively allows faster model training without significantly increasing total costs.
    
    ![Screenshot 2023-09-10 at 11.42.45 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.42.45_AM.png)
    
    ![Screenshot 2023-09-10 at 11.43.13 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.43.13_AM.png)
    

**Determinism and Reliability:**

- Ensuring determinism in various components of the stack, including data streaming and training, is crucial for consistent results.
- Hardware failures are common in GPU clusters, but orchestration tools can detect and automatically resume runs, minimizing human intervention.
    
    ![Screenshot 2023-09-10 at 11.43.49 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.43.49_AM.png)
    
    ![Screenshot 2023-09-10 at 11.45.09 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.45.09_AM.png)
    

**Training Runtime:**

- MosaicML Streaming is an open-source library designed for secure data streaming from object stores to compute clusters.
    
    ![Screenshot 2023-09-10 at 11.45.57 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.45.57_AM.png)
    
- Composer, another open-source library built on PyTorch, simplifies the training of models, including LLMs, by handling mixed-precision training, distributed training, and checkpointing.
    
    ![Screenshot 2023-09-10 at 11.46.43 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.46.43_AM.png)
    
- Fully Sharded Data Parallelism (FSDP) is used to split model parameters across GPUs, enabling training of extremely large models that do not fit on a single GPU.
    
    ![Screenshot 2023-09-10 at 11.47.16 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.47.16_AM.png)
    
    ![Screenshot 2023-09-10 at 11.47.53 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.47.53_AM.png)
    
    Check the video again: How is this different from DDP
    

**LLM Foundry:**

- LLM Foundry is an open-source toolkit designed for preparing and fine-tuning large language models.
- It simplifies the process of training, fine-tuning, evaluation, and model preparation for inference, making it accessible to the community.
    
    ![Screenshot 2023-09-10 at 11.48.53 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.48.53_AM.png)
    

**MPT Models and Training Details:**

- MPT-7B and -30B were trained with a focus on data quality and diversity, including English web data, code data, scientific papers, and more.
    
    ![Screenshot 2023-09-10 at 11.49.38 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.49.38_AM.png)
    
- Determining the right proportions of data sources is essential and can be dynamically adjusted using tools like Streaming Dataset.
- The model architecture for MPT is based on the GPT-3 series, with optimizations like FlashAttention for better performance and AliBi for handling longer context lengths.
    
    ![Screenshot 2023-09-10 at 11.50.51 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.50.51_AM.png)
    
- Fine-tuning is a cost-effective way to adapt base models to specific tasks, and larger models are often easier to fine-tune.
    
    ![Screenshot 2023-09-10 at 11.51.59 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.51.59_AM.png)
    
    ![Screenshot 2023-09-10 at 11.53.13 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.53.13_AM.png)
    
- Model evaluation is evolving, with techniques like Gauntlet, which aggregates performance across multiple tasks and dimensions, becoming increasingly important.
    
    ![Screenshot 2023-09-10 at 11.53.32 AM.png](Deployment%20and%20Hardware%20Considerations%20cfdbf8361430470c91c41fe081d01ec9/Screenshot_2023-09-10_at_11.53.32_AM.png)
    

**Future of Model Evaluation:**

- As LLMs become more powerful, model evaluation will require more complex and human-involved metrics.
- Approaches like human rankings may play a significant role in assessing model performance.

In summary, building and training LLMs from scratch involve careful orchestration, scalable infrastructure, and innovative techniques. Tools like MosaicML Streaming, Composer, and LLM Foundry simplify the process, making it accessible to a broader audience of data scientists and ML engineers. Additionally, model evaluation is evolving to keep pace with the advancements in LLM capabilities.

## References

1. [We’re getting a better idea of AI’s true carbon footprint](https://www.technologyreview.com/2022/11/14/1063192/were-getting-a-better-idea-of-ais-true-carbon-footprint/)
2. [ESTIMATING THE CARBON FOOTPRINT OF BLOOM, A 176B PARAMETER LANGUAGE MODEL](https://arxiv.org/pdf/2211.02001.pdf)
3. [Mosaic LLMs (Part 2): GPT-3 quality for <$500k](https://www.mosaicml.com/blog/gpt-3-quality-for-500k) and [ChatGPT and generative AI are booming, but the costs can be extraordinary](https://www.cnbc.com/2023/03/13/chatgpt-and-generative-ai-are-booming-but-at-a-very-expensive-price.html)
4. [When AI’s Large Language Models Shrink](https://spectrum.ieee.org/large-language-models-size)


# 4. Beyond Text-Based LLMs: Multi-Modality

## Introduction

**Title: Beyond Text: Multimodal Language Models**

**Overview:**
In this session on multimodal language models, the focus shifts beyond text data to enable models to process various modalities of data, such as images, structured data, sensor readings, audio, and more. The goal is to harness the potential of AI models to handle diverse data types and generate meaningful outputs.

**Key Points:**

**Multimodal Potential:**

- Multimodal language models have the capability to process a wide range of data types beyond text, including visual data (images), structured data, sensor data, time series data, audio, and more.
- The transformer architecture, with its flexible input token representation and attention mechanism, can be adapted to work with different modalities, enabling joint representations for diverse data types.

**Common Multimodal AI Types:**

- **Vision Transformers:** These models specialize in processing images and are a common example of multimodal AI. They can be used to understand and generate textual descriptions of images, among other tasks.
- **Structured Data Integration:** Multimodal models can process structured data, such as sensor readings or tabular data, alongside text and other modalities to gain insights or make predictions.
- **Video Analysis:** Multimodal models can handle video data, which combines visual and temporal information, making them suitable for tasks like action recognition or video summarization.

**Training and Tuning Techniques:**

- Effective training and tuning techniques are crucial for multimodal models. These include strategies for learning joint representations, handling different data distributions, and optimizing performance on specific tasks.
- Multimodal models may require careful data preprocessing and feature engineering to align modalities and make the most of the available information.

**Emerging Alternatives and Enhancements:**

- Researchers are exploring alternative architectures and enhancements to the transformer model to achieve similar or better performance while reducing computation and data requirements.
- These developments aim to make multimodal AI more accessible and efficient, opening up new possibilities for applications across various domains.

**Exciting Future Potential:**

- The ability to work with multiple data modalities seamlessly and generate complex outputs is one of the most promising areas in AI.
- Multimodal language models have the potential to transform a wide range of industries, from healthcare (analyzing medical images and patient records) to aviation (processing sensor data from aircraft) and beyond.

In summary, multimodal language models represent a significant step forward in AI's evolution, allowing models to leverage diverse data types to perform a wide range of tasks. The flexibility of the transformer architecture makes it a powerful tool for processing multimodal data and creating innovative AI applications.

**Title: Exploring Multi-Modal Language Models**

**Overview:**
This module explores the exciting field of multi-modal language models, which go beyond text data to process various data modalities, including images, audio, and more. It highlights key models, trends, applications, and the flexibility of Transformer architectures.

**Key Points:**

**Multi-Modal Landscape:**

- Multi-modal language models can process diverse data types, such as text, images, audio, and structured data, making them highly versatile for various applications.
- These models mimic human perception by handling different modalities simultaneously, enabling richer interactions and understanding of the world.

**Notable Models:**

- **Whisper:** Converts speech audio into text, enabling speech recognition tasks.
- **DALL·E:** Generates images from textual descriptions, showcasing the model's creative abilities.
- **CLIP:** Associates textual and visual data, allowing the model to predict relevant text descriptions for images.
    
    ![Screenshot 2023-09-10 at 12.20.39 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.20.39_PM.png)
    
- **VideoLlama:** An example of multi-modal applications where users can interact with video content using text-based queries and prompts.
- **MiniGPT-4:** Demonstrates the model's ability to explain memes, generate code, and engage in creative tasks.
    
    ![Screenshot 2023-09-10 at 12.21.00 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.21.00_PM.png)
    

**Chain of Thought Reasoning:**

- Multi-modal language models can exhibit chain of thought reasoning by processing multi-modal information in context. For example, explaining events between video frames or finding common properties between images.
    
    ![Screenshot 2023-09-10 at 12.21.38 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.21.38_PM.png)
    

**Simultaneous Multi-Modality:**

- These models can process and generate responses using multiple modalities simultaneously. For instance, they can analyze both images and audio while responding to text-based questions or prompts.

**Agent-Like Abilities:**

- Multi-modal language models can act as agents that leverage other tools or models to complete tasks, showcasing their integration capabilities.
    
    ![Screenshot 2023-09-10 at 12.21.54 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.21.54_PM.png)
    

**Transformer Architectures:**

- While Transformer architectures were originally designed for text processing, they have been adapted to handle multi-modal data by accepting inputs from various modalities.
- The module delves into how Transformers can be extended to process non-textual data, highlighting their flexibility and adaptability.
    
    ![Screenshot 2023-09-10 at 12.22.05 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.22.05_PM.png)
    

**Limitations and Alternatives:**

- The limitations of multi-modal language models are discussed, along with potential alternative architectures that could address these limitations.

**Future Possibilities:**

- The module emphasizes the wide-ranging possibilities of multi-modal applications across industries and domains, transforming how humans interact with AI systems.

In summary, multi-modal language models represent a groundbreaking frontier in AI, enabling models to process diverse data types and perform complex tasks across various modalities. The flexibility of Transformer architectures and their integration with different data sources open up new horizons for AI-driven applications and interactions.

## [Transformers Beyond Text](https://learning.edx.org/course/course-v1:Databricks+LLM102x+2T2023/block-v1:Databricks+LLM102x+2T2023+type@sequential+block@e0778cbfa7d2400db4e774301105cc70)

**Title: Leveraging Transformer Architectures for Multi-Modal Data**

**Overview:**
This section explores the adaptability of Transformer architectures for multi-modal data processing. It covers the use of Transformers in computer vision and audio analysis, emphasizing the role of Vision Transformers (ViT) and Speech Transformers. The discussion also addresses the challenges and trends in multi-modal modeling.

**Key Points:**

**Transformer Versatility:**

- Transformer architectures, originally designed for text processing, exhibit remarkable adaptability for handling diverse data modalities, including images, audio, and more.
    
    ![Screenshot 2023-09-10 at 12.23.44 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.23.44_PM.png)
    
- Cross-attention mechanisms enable Transformers to bridge different modalities, such as text, images, audio, and time series data, allowing for multi-modal processing.
    
    ![Screenshot 2023-09-10 at 12.23.55 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.23.55_PM.png)
    

**Computer Vision with Transformers:**

- Vision Transformers (ViT) represent images as a sequence of patches.
    
    ![Screenshot 2023-09-10 at 12.24.11 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.24.11_PM.png)
    
- Each patch is linearly projected into a D-dimensional vector, creating patch embeddings.
    
    
    ![Screenshot 2023-09-10 at 12.24.37 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.24.37_PM.png)
    
    ![Screenshot 2023-09-10 at 12.25.17 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.25.17_PM.png)
    
    ![Screenshot 2023-09-10 at 12.26.54 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.26.54_PM.png)
    
- Positional embeddings are added to help the model infer spatial relationships.
    
    ![Screenshot 2023-09-10 at 12.27.20 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.27.20_PM.png)
    
- The entire sequence of patch embeddings, along with a CLS token, is input into a standard Transformer encoder.
- Pre-training on ImageNet data helps ViT achieve impressive performance, surpassing convolutional neural networks (CNNs) in computational efficiency and accuracy on larger datasets.
- ViT is four times faster to train than ResNets.
    
    ![Screenshot 2023-09-10 at 12.29.14 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.29.14_PM.png)
    
    ![Screenshot 2023-09-10 at 12.29.34 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.29.34_PM.png)
    

**Audio Processing with Transformers:**

- Audio can be represented as a sequence of fixed-length audio frames.
    
    ![Screenshot 2023-09-10 at 12.30.27 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.30.27_PM.png)
    
- Transformers can be applied to audio analysis, with convolutional neural network layers often used to reduce input dimensions.
    
    ![Screenshot 2023-09-10 at 12.30.45 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.30.45_PM.png)
    
- The Speech Transformer combines convolutional layers and Transformers for audio processing.
- There is a limited number of multi-modal audio-visual models, with most audio models focused on text-to-speech, speech-to-text, or speech-to-speech tasks.
    
    ![Screenshot 2023-09-10 at 12.31.28 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.31.28_PM.png)
    
- Multi-modal modeling faces challenges due to the scarcity of high-quality multi-modal data compared to single-modal data.
- Combining different modalities, such as text, images, and audio, requires innovative research to effectively leverage Transformers for multi-modal applications.
- The adoption of Transformer architectures in computer vision and audio analysis has driven research into improved architectures and model variations, like SwinTransformer and MLP-mixer.

**Conclusion:**
Transformer architectures, initially developed for text, have been successfully extended to process multi-modal data, including images and audio. Vision Transformers (ViT) have proven to be computationally efficient for computer vision tasks, outperforming traditional convolutional neural networks (CNNs) on larger datasets. Audio processing with Transformers is still emerging, with fewer multi-modal models available. Overcoming data challenges and refining multi-modal modeling techniques are areas of ongoing research and development.

In summary, Transformers have brought versatility to multi-modal data processing, enabling AI models to handle diverse data types and paving the way for more sophisticated multi-modal applications in the future.

## **Training data for multi-modal language models**

**Title: Challenges in Collecting Multi-Modal Training Data**

**Overview:**
This section discusses the complexities and challenges associated with gathering training data for multi-modal language models. It highlights the difficulties in collecting text-to-audio and text-to-video data, which often require manual curation. The examples provided illustrate the meticulous work required for annotating and structuring such data. Additionally, the section introduces the LAION-5B dataset as a valuable open-source resource for multi-modal training.

**Key Points:**

**Challenges in Data Collection:**

- Data collection for text-to-image and image-to-text tasks is more accessible compared to text-to-audio and text-to-video data.
- Researchers often need to manually curate multi-modal data, which is a time-consuming process.
    
    ![Screenshot 2023-09-10 at 12.32.37 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.32.37_PM.png)
    
- Annotators may be required to provide detailed video descriptions frame by frame, structured scene descriptions, or dense scene descriptions.
- Data can be organized into JSON or tabular formats for effective utilization.
    
    ![Screenshot 2023-09-10 at 12.33.02 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.33.02_PM.png)
    
- High-quality examples are essential as models require a groundwork of well-curated data before generating examples.
    
    ![Screenshot 2023-09-10 at 12.33.12 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.33.12_PM.png)
    

**LAION-5B Dataset:**

- LAION-5B is one of the most significant open-source image-text datasets, designed for research purposes.
- It comprises 5.85 billion CLIP-filtered image-text pairs, with 2.3 billion in English and 2.2 billion in other languages.
- Notably, LAION-5B includes many copyrighted images, and LAION does not claim ownership over them.
    
    ![Screenshot 2023-09-10 at 12.33.39 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.33.39_PM.png)
    

**Importance of Data Curation:**

- Data curation is a crucial and non-trivial aspect of producing high-quality multi-modal models.
- High-quality training data is essential for model performance in tasks involving multi-modal inputs.

**Few-Shot Learning as a Solution:**

- The challenge of limited data resources can be mitigated through few-shot learning techniques, which will be explored in the next video.

**Conclusion:**
Gathering high-quality training data for multi-modal language models, particularly for text-to-audio and text-to-video tasks, is a labor-intensive and time-consuming process. The availability of datasets like LAION-5B aids researchers in advancing multi-modal modeling. To address data limitations, few-shot learning techniques provide a valuable solution, which will be elaborated upon in the subsequent video.

## X-shot learning

**Title: Few-Shot Learning in Multi-Modal Language Models**

**Overview:**
This section explores the emergence of few-shot learning techniques in multi-modal language models, focusing on both computer vision and audio domains. It discusses two significant models: CLIP, which excels in zero-shot learning for vision and text, and Flamingo, a multi-modal few-shot learning model. Additionally, the section introduces Whisper as an exemplary zero-shot audio model.

![Screenshot 2023-09-10 at 12.35.32 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.35.32_PM.png)

**Key Points:**

**Few-Shot Learning in Computer Vision (CLIP):**

- CLIP (Contrastive Language Image Pairing) learns visual representations from extensive natural language data.
- It trains separate image and text encoders using a contrastive loss, predicting correct text-image pairs.
- CLIP excels in zero-shot learning by maximizing similarity between words and visual information.
- Demonstrates strong performance on various non-ImageNet datasets, including challenging scenarios.
    
    ![Screenshot 2023-09-10 at 12.35.43 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.35.43_PM.png)
    
    ![Screenshot 2023-09-10 at 12.36.22 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.36.22_PM.png)
    

**Flamingo: A Multi-Modal Few-Shot Learning Model:**

- Flamingo is a family of visual language models capable of processing interleaved visual data and text, generating free-form text.
- It employs a Perceiver Resampler to convert variable-sized visual features into a fixed number of tokens.
    
    ![Screenshot 2023-09-10 at 12.37.11 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.37.11_PM.png)
    
- Flamingo leverages a pre-trained language model (Chinchilla) to harness generative language abilities.
- Bridges vision and language models by freezing their weights and linking them through learnable architectures.
- Supports a wide range of tasks, from open-ended tasks like visual question answering to close-ended tasks like classification.
- Outperforms other few-shot learning approaches with as few as 4 examples, thanks to curated high-quality datasets.
    
    ![Screenshot 2023-09-10 at 12.37.54 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.37.54_PM.png)
    
    ![Screenshot 2023-09-10 at 12.38.06 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.38.06_PM.png)
    
    ![Screenshot 2023-09-10 at 12.38.26 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.38.26_PM.png)
    

**Flamingo Model Output Examples:**

- Flamingo can infer object details, perform reasoning, and maintain consistent response formats when given input prompts.
- It handles image-text input and queries, providing responses consistent with previous response formats.
- Capable of processing series of video frames and answering questions about the frames.
    
    ![Screenshot 2023-09-10 at 12.39.00 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.39.00_PM.png)
    

**Zero-Shot Audio Model (Whisper):**

- Whisper, an audio model, utilizes an encoder-decoder Transformer architecture and convolutional neural networks to reduce audio dimensions.
    
    ![Screenshot 2023-09-10 at 12.39.15 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.39.15_PM.png)
    
- Splits input audio into 30-second frames and excels in zero-shot settings.
- Achieves lower average word error rates compared to models fine-tuned on LibriSpeech, demonstrating human-level robustness.
    
    ![Screenshot 2023-09-10 at 12.39.33 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.39.33_PM.png)
    

**Challenges and Future Directions:**

- Despite significant advancements, multi-modal language models still face various challenges.
- Upcoming sections will delve into these challenges and explore emerging architectures that may replace or enhance the ubiquitous attention mechanism.

## **Challenges and alternative architectures**

**Title: Challenges and Future Directions in Multi-Modal Language Models**

**Overview:**
This section discusses the challenges and limitations faced by multi-modal language models (LLMs) and explores emerging architectures and approaches to address them. It highlights issues such as hallucination, bias, copyright concerns, and common sense deficiencies, while also presenting alternative architectural directions.

**Key Points:**

**Challenges in Multi-Modal Language Models (LLMs):**

- LLMs share limitations with traditional language models, including hallucination, sensitivity to prompts, context limitations, and inference compute costs.
- Concerns about bias, toxicity, and fairness persist in LLMs, as evidenced by examples where models produce biased outputs.
- Copyright issues, as seen in datasets like LAION, pose challenges regarding image ownership and usage rights.
    
    ![Screenshot 2023-09-10 at 12.41.38 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.41.38_PM.png)
    
- Multi-modal LLMs may struggle with common-sense reasoning, generating nonsensical outputs in response to text-image prompts.
    
    ![Screenshot 2023-09-10 at 12.42.02 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.42.02_PM.png)
    

**Alternative Architectures to Attention Mechanism:**

- Reinforcement Learning with Human Feedback (RLHF) incorporates human feedback to train a reward model, which assesses the quality of model outputs.
- KL loss ensures that fine-tuned models don't deviate significantly from their pre-trained counterparts.
- Proximal Policy Optimization (PPO) updates large language models based on reward signals from the reward model.
    
    ![Screenshot 2023-09-10 at 12.42.40 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.42.40_PM.png)
    

**Emerging Architectural Directions:**

- Hyena Hierarchy uses convolutional neural networks (CNNs) instead of Transformers, showing promise as a few-shot model for languages and matching Vision Transformers' performance.
    
    ![Screenshot 2023-09-10 at 12.43.18 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.43.18_PM.png)
    
- Retentive networks introduce a retention mechanism that combines recurrence and attention, achieving higher computational efficiency without sacrificing model performance.
    
    ![Screenshot 2023-09-10 at 12.43.56 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.43.56_PM.png)
    

**Emerging Applications in Multi-Modal Language Models:**

- The field continues to evolve with emerging applications, indicating its ongoing growth and relevance.
- Specific applications and use cases for multi-modal language models are continually being explored and developed, expanding their practical utility.

**Conclusion:**

- The challenges and opportunities in multi-modal language models underscore the dynamic nature of this field, with ongoing research and development shaping its future.

**Note:** The section ends with a reference to emerging applications in the field, serving as a segue to the next video.

## **Emerging applications**

**Title: Emerging Applications and Conclusion in Multi-Modal Language Models**

**Overview:**
In this section, the presenter discusses exciting emerging applications of multi-modal language models (LLMs) and recaps the key takeaways from the module. The applications range from generating 3D objects and videos from text to using LLMs for robotics, code generation, language diversity, audio processing, biological research, and even household robots.

**Key Points:**

**Emerging Applications of Multi-Modal Language Models:**

1. **DreamFusion:** Generates 3D objects from text captions, showcasing the capabilities of LLMs in creative applications.
2. **Meta's Make-A-Video Application:** Allows users to generate videos from text inputs, demonstrating the versatility of LLMs in multimedia content creation.
3. **PaLM-E-bot:** Google integrates its PaLM language model with robotics applications, indicating the expansion of LLMs into the robotics domain.
4. **Code Generation with AlphaCode:** LLMs are increasingly being used to generate code, addressing various programming challenges, such as optimizing pizza slice preparation times.
5. **Multi-Lingual Models like Bactrian-X:** Focuses on lower-resource languages, highlighting the potential of LLMs to bridge language gaps.
    
    ![Screenshot 2023-09-10 at 12.45.52 PM.png](Beyond%20Text-Based%20LLMs%20Multi-Modality%20cca5b0c8c6bf4cfebe2afa98168d5c34/Screenshot_2023-09-10_at_12.45.52_PM.png)
    
6. **Textless NLP Application:** Enables speech generation from raw audio without text transcription, simplifying multilingual audio processing.
7. **Biological Research:** Transformer architectures are making their way into biological research, demonstrating their adaptability to sequences like protein sequences.
8. **Household Robots:** Speculates about the possibility of having household robots capable of playing games, chatting, and performing tasks like stacking blocks.

**Recap of Module's Key Takeaways:**

- Multi-modal language models are gaining prominence in research and applications.
- Transformers, as versatile sequence processing architectures, can handle various non-text inputs like audio and images.
- LLMs inherit limitations from pre-trained large language models, including issues related to bias, hallucination, and sensitivity to prompts.
- Emerging alternative architectures like RLHF, Hyena Hierarchy, and retentive networks offer potential solutions to existing challenges.
- The field of multi-modal language models continues to evolve, with exciting applications on the horizon.

**Conclusion:**

- The module ends with an invitation for learners to explore practical applications in the demo notebook and hands-on experience in the lab notebook.

**Note:** This section serves as a conclusion to the module, summarizing key points and encouraging learners to engage further in practical exercises.

## References

1. [Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts](https://arxiv.org/pdf/2307.11661.pdf)
2. [EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/abs/2305.07027)
3. [Key-Locked Rank One Editing for Text-to-Image Personalization](https://arxiv.org/pdf/2305.01644.pdf)
    - This paper describes text-to-image generation being done with a model that is 100KB in size. Maybe size isn't everything.
4. [AudioCraft by MetaAI](https://audiocraft.metademolab.com/)
    - MetaAI just released this code base for generative audio needs in early August 2023. It can model audio sequences and capture the long-term dependencies in the audio.
5. [X-ray images with LLMs and vision encoders](https://arxiv.org/abs/2308.01317)
6. [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
7. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
    - This is the original paper on Stable Diffusion.
8. [The Illustrated Stable Diffusion by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)
    - This blog post illustrates how stable diffusion model works.
9. [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)
    - This paper describes how to add diffusion models to Vision Transformer.
10. [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) by Huyen Chip