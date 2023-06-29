---
katex: true
markup: mmark
title: "Prompting"
description: "Understanding basics of Generative Modeling"
dateString: June 2023
draft: false
tags: ["NLU", "Prompting"]
weight: 107
---

# Prompting

Readings:
* Sections 1-3 [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language](https://arxiv.org/abs/2107.13586) [Processing](https://arxiv.org/abs/2107.13586), Liu et al. (2021)
* [Language models are unsupervised multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), Radford et al. (2019)
* [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)- GPT

### A short overview of change in language modeling paradigm
* At the start we used to have `fully-supervised learning` modeling. Here, we were doing task specific supervised modeling
	* `feature-engineering`: domain knowledge requirement
	* then, we moved towards `architecture-engineering`, for automatic learning of features for the supervised task.
* Then we moved towards `pretrain-finetune` paradigm. `pre-train` a LM on a general purpose dataset as that is available in abundance. Adapt the `pre-trained` LM to downstream tasks by introducing additional parameters and `finetune` using task-specific objective functions.
	* focus is shifted to `objective-engineering` i.e. the training objectives to be used at both `pretraining` and `finetuning` stage. We add a pre-training task which is similar to downstream task, this improves performance on the downstream task later.
* Now we are moving towards `pre-train, prompt, predict` paradigm. Instead of adapting pre-trained LMs to downstream tasks via `objective-engineering`, we are reformulating the downstream tasks to look more like those solved during the original LM training with the help of textual `prompt`
	* Eg:`“I felt so ___” `, and ask the LM to fill the blank with an emotion-bearing word. Or if we choose the prompt `“English: I missed the bus today. French: ”`), an LM may be able to fill in the blank with a French translation.

![image](/posts/prompting/1.png)
### Prompting Basics
![image](/posts/prompting/2.png)
* **Terminologies**:
	* `prefix prompt`: variety of prompt where the input text comes entirely before $\bf{z}$
	* `cloze prompt`: the first variety of prompt with a slot to fill in the middle of the text

 * **Prompt Addition**: $f_{prompt}(x)$ is applied on $\bf{x}$ to to generate $\mathbf{x}' = f_{prompt}(x)$
	1. Apply a template, which is a textual string that has two slots: an input slot [X] for input x and an answer slot [Z] for an intermediate generated answer text z that will later be mapped into y.
	2. Fill slot [X] with the input text $\bf{x}$.
* **Answer Search**: 
	* we search for the highest-scoring text $\bf{z}ˆ$ that maximizes the score of the LM. We first define $Z$ as a set of permissible values for $\bf{z}$.
		$$
	\hat{z} = \underset{z \epsilon Z}{search} P(f_{fill}(x', z);\theta)
		$$
	* $Z$ could take variety of input:
		* **classification**: could be a small subset of the words `{“excellent”, “good”, “OK”, “bad”, “horrible”}` or `{++, +, ~, -, --}`
		* **regression**: continuous values, constants 

* **Answer Mapping**: we would like to go from the highest-scoring answer $zˆ$ to the highest-scoring output $yˆ$. This is trivial for cases, where answer itself is the output, however for cases where multiple result could result in the same output, we need a mapping function:
	* sentiment-bearing words (e.g. “excellent”, “fabulous”, “wonderful”) to represent a single class (e.g. “++”)

* **Design Considerations for Prompting**

![image](/posts/prompting/3.png)

### Pre-trained Language Models

Systematic view of various pre-trained LMs:
* **main training objective**
	* auxiliary training objective
* **type of text noising**
* **Directionality**: attention mask



#### Main Training Objective

The main training objective of the pre-trained LMs plays an important role in determining its applicability to particular prompting tasks.

* **Standard Language Model (SLM)**
	* Autoregressive prediction (left to right)
		* These are particularly suitable for `prefix prompts`
	
* **Denoising Objective**:
	* Noising function: $\tilde{f} = f_{noise}(x)$
	* Task to predict: $P(x|\tilde{x})$
	* These types of reconstruction objectives are suitable for `cloze prompts`
	* Two common types of denoising objectives
		* **Corrupted Text Reconstruction (CTR)**: the processed text to its uncorrupted state by calculating *loss over only the noised parts* of the input sentence
		* **Full Text Reconstruction (FTR)**: reconstruct the text by *calculating the loss over the entirety of the input texts* whether it has been noised or not
	* **Noising Functions**
		* the specific type of corruption applied to obtain the noised text $\tilde{x}$ has an effect on the efficacy of the learning algorithm
		* **prior knowledge can be incorporated by controlling the type of noise**, e.g. *the noise could focus on entities of a sentence, which allows us to learn a pre-trained model with particularly high predictive performance for entities*

![image](/posts/prompting/4.png)


* **SLM** or **FTR** objectives are maybe more suitable for *generation tasks*
* tasks such as *classification* can be formulated using models trained with any of these objectives

* **Auxiliary Training Objective**:
	* improve models’ ability to perform certain varieties of downstream tasks.
	* **Next Sentence Prediction**: Next Sentence Prediction: do two segments appear consecutively - better sentence representations - `BERT`
	* **Discourse Relation Prediction**: predict rhetorical relations between sentences - better semantics - `ERNIE [Sun et al., 2020]`
	* **Image Region Prediction**: predict the masked regions of an image - for better visual-linguistic tasks - `VL-BERT [Su et al., 2020]`



#### Directionality (Type of attention masking)

* pre-trained LM can be different based on the directionality of the calculation of representations

* **Bidirectional:** full attention no masking 
* **Left-to-right:** diagonal attention masking 
* Mix the two strategies

![image](/posts/prompting/5.png)

#### Typical Pre-training Methods

Following is a representation of popular pre-training methods:
![image](/posts/prompting/6.png)

![image]("/posts/prompting/7.png")
* **Left-to-Right Language Model**
	* Popular backbone for many prompting methods. Representative examples of modern pre-trained left-to-right LMs include **GPT-3** , and **GPT-Neo**
	* Generally large and difficult to train - generally not available to public, thus `pretraining and finetuning`  is generally not possible
	* Useful for **generative tasks**
* **Masked Language Models**
	* Take advantage of full context. When the focus is shifted on generating optimal representation for downstream tasks.
	* BERT is a popular example which aims to predict masked text pieces based on surrounded context
	* In prompting methods, MLMs are generally most suitable for **natural language understanding or analysis tasks** (e.g., text classification, natural language inference , and extractive question answering).
	* Suitable for `cloze prompting`. 
	* `pretraining-finetuning` is generally possible
* **Prefix and Encoder-Decoder**
	* Useful for conditional text-generation tasks such as **translation** and **summarization**
		* such tasks need a pre-trained model both capable of endcoding the text and generating the output
	* (1) using an encoder with **fully-connected mask** (full-attention, no masking) to encode the source $x$ first and then (2) decode the target $y$ **auto-regressively** (from the left to right)
	* **In Prefix-LM**: Encoder-Decoder weights are shared. So same parameters are used to encode $x$ and $y$
		* Eg: UniLM 1-2, ERNIE-M
	* **In Encoder-Decoder**: Weights are different for E & D. $x$ is encoded using encoder weight whereas, $y$ is encoded using decoder weight. 
		* Eg: T5, BART
	* These models were typically used for **text generation purposes**, however, recently they are **being used for non-generation tasks** such as QA, Information Extraction etc. 


### Prompt Engineering

* Creating a promtping function $f{prompt}(x)$
* Manual template engineering
* Automated template learning of discrete prompts: 
	* Prompt mining ”[X] middle words [Z]” 
	* Paraphrase existing prompts - select the ones with highest accuracy 
* Continuous prompts: perform prompting directly in the embedding space of the model 
	* Initialise with discrete prompt, fne tune on task 
	* Template embeddings have their own parameters that can be tuned

### Training

* Promptless fne-tuning (BERT, ELMO) 
* Tuning free prompting: zero-shot (GPT3) 
* Fix prompt tune LM (T5) 
* Additional prompt parameteres: 
	* Fix LM tune prompt
	* Tune LM and prompt (high resource)