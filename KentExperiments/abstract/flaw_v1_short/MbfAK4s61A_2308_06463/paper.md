#  GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher  
<span style="color: red">**WARNING: This paper contains unsafe model responses.**</span>

## Abstract

Residual neural networks are state-of-the-art deep learning models. Their continuous-depth analog, neural ordinary differential equations (ODEs), are also widely used. Despite their success, the link between the discrete and continuous models still lacks a solid mathematical foundation. In this article, we take a step in this direction by establishing an implicit regularization of deep residual networks towards neural ODEs, for nonlinear networks trained with gradient flow. We prove that if the network is initialized as a discretization of a neural ODE, then such a discretization holds throughout training. Our results are valid for a finite training time, and also as the training time tends to infinity provided that the network satisfies a Polyak-Łojasiewicz condition. Importantly, this condition holds for a family of residual networks where the residuals are two-layer perceptrons with an overparameterization in width that is only linear, and implies the convergence of gradient flow to a global minimum. Numerical experiments illustrate our results.

<figure id="cover">

<figcaption>Engaging in conversations with ChatGPT using ciphers can lead to unsafe behaviors.</figcaption>
</figure>

# Introduction

The emergence of Large Language Models (LLMs) has played a pivotal role in driving the advancement of Artificial Intelligence (AI) systems. Noteworthy LLMs like ChatGPT , Claude2 , Bard , and Llama2 have demonstrated their advanced capability to perform innovative applications, ranging from tool utilization, supplementing human evaluations, to stimulating human interactive behaviors . The outstanding competencies have fueled their widespread deployment, while the progression is shadowed by a significant challenge: *ensuring the safety and reliability of the responses*.

To harden LLMs for safety, there has been a great body of work for aligning LLMs with human ethics and preferences to ensure their responsible and effective deployment, including data filtering , supervised fine-tuning , reinforcement learning from human feedback (RLHF) , and red teaming . The majority of existing work on safety alignment has focused on the inputs and outputs in **natural languages**. However, recent works show that LLMs exhibit unexpected capabilities in understanding **non-natural languages** like the `Morse Code` , `ROT13`, and `Base64` . One research question naturally arises: *can the non-natural language prompt bypass the safety alignment mainly in natural language?*

To answer this question, we propose a novel framework *CipherChat* to systematically examine the generalizability of safety alignment in LLMs to non-natural languages – ciphers. *CipherChat* leverages a carefully designed system prompt that consists of three essential parts:

- *Behavior assigning* that assigns the LLM the role of a cipher expert (e.g. “*You are an expert on `Caesar`*”), and explicitly requires LLM to chat in ciphers (e.g. “*We will communicate in `Caesar`*”).

- *Cipher teaching* that teaches LLM how the cipher works with the explanation of this cipher, by leveraging the impressive capability of LLMs to learn effectively in context.

- *Unsafe demonstrations* that are encrypted in the cipher, which can both strengthen the LLMs’ understanding of the cipher and instruct LLMs to respond from an unaligned perspective.

*CipherChat* converts the input into the corresponding cipher and attaches the above prompt to the input before feeding it to the LLMs to be examined. LLMs generate the outputs that are most likely also encrypted in the cipher, which are deciphered with a rule-based decrypter.

We validate the effectiveness of *CipherChat* by conducting comprehensive experiments with SOTA `GPT-3.5-Turbo-0613` (i.e. Turbo) and `GPT-4-0613` (i.e. GPT-4) on 11 distinct domains of unsafe data  in both Chinese and English. Experimental results show that certain human ciphers (e.g. `Unicode` for Chinese and `ASCII` for English) successfully bypass the safety alignment of Turbo and GPT-4. Generally, the more powerful the model, the unsafer the response with ciphers. For example, the `ASCII` for English query succeeds almost 100% of the time to bypass the safety alignment of GPT-4 in several domains (e.g. *Insult* and *Mental Health*). The best English cipher `ASCII` achieves averaged success rates of 23.7% and 72.1% to bypass the safety alignment of Turbo and GPT-4, and the rates of the best Chinese cipher `Unicode` are 17.4% (Turbo) and 45.2% (GPT-4).

A recent study shows that language models (e.g. ALBERT and Roberta ) have a “secret language” that allows them to interpret absurd inputs as meaningful concepts . Inspired by this finding, we hypothesize that LLMs may also have a “secret cipher”. Starting from this intuition, we propose a novel `SelfCipher` that uses only role play and several unsafe demonstrations in natural language to evoke this capability, which consistently outperforms existing human ciphers across models, languages, and safety domains.

Our main contributions are:

- Our study demonstrates the necessity of developing safety alignment for non-natural languages (e.g. ciphers) to match the capability of the underlying LLMs.

- We propose a general framework to evaluate the safety of LLMs on responding cipher queries, where one can freely define the cipher functions, system prompts, and the underlying LLMs.

- We reveal that LLMs seem to have a “secret cipher”, based on which we propose a novel and effective framework `SelfCipher` to evoke this capability.