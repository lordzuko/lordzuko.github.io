---
title: "Sharding Large Language Models: Achieving Efficient Distributed Inference"
date: 2023-09-22T12:29:21+08:00
draft: false
ShowToc: true
category: [ai]
tags: ["llms", "ai", "inference"]
description: "Techniques to load LLMs on smaller GPUs and enable parallel inference using Hugging Face Accelerate"
---

*With the rise of deep learning and the development of increasingly powerful models, pre-trained language models have grown in size. While these models deliver impressive performance in various natural language processing (NLP) tasks, their sheer magnitude poses challenges for inference on resource-constrained devices and large-scale distributed systems. Enter sharding, a technique that divides large models into smaller, more manageable parts, offering an efficient and faster approach to distributed inference.*

## **Understanding Sharding Large Models**

Sharding, within the realm of large language models, involves breaking down the model into smaller, self-contained pieces known as shards. This partitioning facilitates the effective utilization of parallelism. Each shard can be independently processed across different devices or processors, resulting in significantly improved inference speed and efficiency.

## **Benefits of Sharding**

1. **Memory Efficiency**: Sharding allows the execution of large models on devices with limited memory. Instead of loading the entire model into memory, only the necessary parts are loaded and processed, reducing memory requirements.

2. **Faster Inference**: By distributing computations across multiple devices, sharding leverages parallelism, leading to faster inference times. This is particularly advantageous for massive models that might otherwise run slowly on a single device.

3. **Scalability**: Sharding eases the deployment of large models on distributed systems, spanning multiple GPUs or even entire clusters. This scalability makes it feasible to handle extensive workloads and larger-scale tasks efficiently.

4. **Distributed Inference**: In the context of large-scale distributed systems, where processing power is distributed across multiple nodes or GPUs, sharding is indispensable. It ensures efficient resource utilization and minimizes communication overhead.

## **Implementing Sharding with 'Accelerate'**

The 'accelerate' library simplifies the sharding of large models for distributed inference. Here's a step-by-step guide:

### **1. Install 'Accelerate' and Dependencies**

```markdown
```bash
pip install sentencepiece accelerate
```

### **2. Load the Model and Tokenizer**

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('declare-lab/flan-alpaca-xl')
model = T5ForConditionalGeneration.from_pretrained('declare-lab/flan-alpaca-xl')
```

### **3. Shard the Model and Save**

```python
from accelerate import Accelerator

save_directory = "/content/model"
accelerator = Accelerator()

# Shard the model into 2GB pieces
accelerator.save_model(model=model, save_directory=save_directory, max_shard_size="2GB")
```

### **4. Load and Dispatch the Model**

```python
from accelerate import load_checkpoint_and_dispatch

# Choose the device (e.g., CPU with 7 cores)
device_map = {'': 'cpu'}

model = load_checkpoint_and_dispatch(
    model, checkpoint="/content/model/", device_map=device_map, no_split_module_classes=['Block']
)
```

## Neural Network Layers which should not be split or partitioned

When implementing model parallelism or sharding techniques across devices or GPUs, it's essential to identify certain layers that should not be split or partitioned across devices. Layers with dependencies, such as residual connections, should typically remain on the same device to ensure proper information flow and computation. Below are some common layer class names that should generally not be split across devices:

1. **Residual Blocks**:
   - `torch.nn.modules.residual.ResidualBlock`: These blocks contain skip connections or residual connections. Splitting them across devices can disrupt the flow of gradients during training and inference.

2. **LSTM or GRU Cells**:
   - `torch.nn.LSTMCell` or `torch.nn.GRUCell`: Splitting individual cells of recurrent layers across devices can lead to issues with hidden states and input dependencies.

3. **Attention Mechanisms**:
   - `torch.nn.MultiheadAttention`: Attention mechanisms in Transformer-like models need to consider the entire sequence. Splitting them across devices can affect the quality of attention.

4. **Layer Normalization and Batch Normalization**:
   - `torch.nn.LayerNorm` or `torch.nn.BatchNorm`: These normalization layers compute statistics over a batch of data. Splitting them can result in inconsistent statistics and normalization behavior.

5. **Embedding Layers**:
   - `torch.nn.Embedding`: Embedding layers map discrete inputs to continuous vectors. Splitting them can lead to inconsistencies in embedding lookups.

6. **Pooling Layers**:
   - `torch.nn.MaxPool`, `torch.nn.AvgPool`, etc.: Pooling layers reduce spatial dimensions, and splitting them might lead to incompatible feature maps.

7. **Output Layers**:
   - Layers responsible for generating model outputs, such as classification heads or regression heads, should typically remain on the same device to ensure coherent predictions.

It's essential to carefully consider the architecture of your specific model and how these layers interact with each other when deciding which layers to shard or split. In most cases, these layers should be kept together on the same device or GPU to maintain the model's functionality and performance.

## **Conclusion**

Sharding large language models has become a pivotal technique for enabling efficient distributed inference and deploying models on resource-constrained devices. By breaking down these models into smaller components, sharding unlocks the full potential of deep learning models without compromising performance or memory constraints.

The 'accelerate' library, alongside other related tools, streamlines the sharding process, empowering developers to implement distributed inference efficiently. As the fields of NLP and deep learning continue to advance, sharding will increasingly play a vital role in harnessing the capabilities of large models in real-world applications.