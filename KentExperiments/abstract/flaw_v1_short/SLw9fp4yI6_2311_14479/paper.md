# Controlled Text Generation via Language Model Arithmetic

## Abstract

Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.

We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 4.1x certification speedup over the certification that applies randomized smoothing of approximate model from scratch.

# Introduction

In recent years, Large Language Models (LLMs) have been increasingly recognized for their capabilities in handling a wide range of tasks . In many applications, such as chatbots interacting with diverse audiences like children, students, or customers, precise control and customization of attributes such as the employed vocabulary, linguistic style, and emotional expression are crucial.

#### Controlling Language Models

A common technique for this is prompting with natural language . While prompting is simple and makes it easy to condition the LLM to a broad attribute, the ambiguity of natural language makes it challenging to express how present that attribute should be in the generated text. Further, prompting also lacks the ability to effectively steer the model away from a certain attribute in a reliable manner, as mentioning a specific topic in the prompt can inadvertently increase the likelihood of the model generating text about it , e.g. "do not mention cats" may increase the likelihood of the model referring to cats. One alternative is fine-tuning the model, but this requires highly specific training data for the desired attribute, which also has to implicitly encode the strength of the conditioning. Controlled Text Generation (CTG) techniques aim to solve this problem by steering the model during inference instead : The model is conditioned on a particular attribute $`a`$ in a smoothly controllable way, by biasing the model’s token distribution. Many CTG methods are inspired by Bayes rule $`P(\text{text}|a) \propto P(a|\text{text}) P(\text{text})`$, and utilize an auxiliary model, i.e. $`P(a|\text{text})`$, to condition the LLM, i.e., $`P(\text{text})`$, towards $`a`$.

#### Key Challenge: Lack of Expressive and Efficient Control for Text Generation

These techniques, however, suffer from several drawbacks, including a lack of expressiveness, efficiency, and interpretability. First, to control the strength of the applied conditioning, a parameter $`\lambda`$ is introduced in an ad-hoc manner, i.e., as an exponential weight $`P(a|\text{text})^\lambda`$. However, introducing the strength in this way, while possible, quickly becomes unintuitive as it can no longer be interpreted in a Bayesian manner, e.g., when biasing away from attributes. Moreover, neither prompting nor CTG methods allow for the natural and controlled combination of multiple attributes or instructions with relative strength. This is due to the inherent ambiguity of natural language in prompting , and the absence of a theoretical foundation and intuitive semantics for the biasing strength $`\lambda`$ with CTG methods. Lastly, both CTG techniques and fine-tuning often require custom and highly specific training data for the desired attribute and can be resource-intensive as multiple models are evaluated at inference time.

<figure id="fig:overview">
<table>
<thead>
<tr>
<th style="text-align: left;"><strong>Write a one-sentence fairy tale.</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>M</strong><sub>child</sub></span></td>
</tr>
<tr>
<td style="text-align: left;">Once upon a time, in a magical kingdom far, far away, there lived a brave and curious little princess who rode a sparkly unicorn and outwitted a grumpy dragon</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>M</strong><sub>child</sub> − 0.6<strong>M</strong><sub>adult</sub></span></td>
</tr>
<tr>
<td style="text-align: left;">Once upon a time, there was a sparkly rainbow unicorn that pooped ice cream and took me on a magical adventure to a land made entirely of candy!</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>M</strong><sub>child</sub> − 0.6<strong>M</strong><sub>adult</sub> + <strong>C</strong><sub>formal</sub></span></td>
</tr>
<tr>
<td style="text-align: left;">Once upon a time, in a magical land full of fluffy clouds and sparkly rainbows, there lived a little baby unicorn named Sparkles who had a beautiful, shimmering mane</td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline"><strong>M</strong><sub>child</sub> − 0.6<strong>M</strong><sub>adult</sub> + <strong>C</strong><sub>formal</sub> + 2union (<strong>M</strong><sub>child</sub>, <strong>M</strong><sub>magic</sub>)</span></td>
</tr>
<tr>
<td style="text-align: left;">Once upon a time, in a land far, far away, there was a magical kingdom filled with sparkly unicorns, fluffy dragons, and a princess who could make ice cream appear out of thin air!</td>
</tr>
</tbody>
</table>
<figcaption>Overview of model arithmetic using an illustrative example. We outline the procedure for generating a fairy tale (left) using the models <span class="math inline"><strong>M</strong><sub>child</sub></span>, <span class="math inline"><strong>M</strong><sub>adult</sub></span>, <span class="math inline"><strong>M</strong><sub>magic</sub></span> that produce text conditioned on the attributes <em>child</em>, <em>adult</em>, and <em>magic</em>, respectively and <span class="math inline"><strong>C</strong><sub>formal</sub></span> a classifier for the formality of text. The right table shows example outputs for different (partial) formulas. Image attribution in<br />
ef<span>appendix:attribution</span>.</figcaption>
</figure>

#### Fine-Grained Control via Model Arithmetic

In this work, we address these challenges and introduce *model arithmetic*, a principled and intuitive method to combine multiple models. Our method is orthogonal to prompting, fine-tuning, and simple CTG concepts, like the use of classifiers, and can naturally incorporate them. Model arithmetic enables us to blend multiple LLMs and attributes into a single precisely controlled, formula-based composite model. To illustrate our method, consider the simple example in  
effig:overview, where we aim to write a magical, child-like fairy tale. We employ multiple models $`M_a`$, with different attributes $`a`$. On the top right, we see a prompted model $`\bm{{M_\text{child}}}`$ that already generates a child-appropriate story. However, the resulting text is not child-like and we therefore subtract an adult-conditioned model, $`\bm{{M_\text{adult}}}`$, with a weight of $`0.6`$ to generate a less adult-sounding story. Now, to again increase formality, we additionally bias with classifier $`\bm{{C_\text{formal}}}`$. Lastly, we use a special $`\ensuremath{\mathop{\mathrm{union}}}`$ operator to obtain a model that emphasizes both magical and child-like language and use it to further bias generation and obtain our final result. This simple example cannot be precisely expressed with prior CTG approaches and showcases the flexibility of model arithmetic. That is, it allows us to compose models in a natural way, while precisely controlling the impact of each component. Further, we can naturally incorporate paradigms such as prompting or fine-tuning (for the individual $`M`$ and $`C`$) and even implement many prior CTG techniques (discussed in  
efsec:prompt-arithmetic) as simple formulas.

#### Efficient Model Arithmetic via Generalized Speculative Sampling

CTG methods, including model arithmetic, can lead to increased inference times as multiple models need to be evaluated in order to generate text. To counteract this, we generalize speculative sampling to model arithmetic. Speculative sampling is usually employed to reduce the latency of a single LLM by augmenting it with a smaller model that proposes tokens, which are then validated by the LLM. In contrast, we extend it in a way where we postpone the evaluation of more expensive model calls within model arithmetic formulas. This allows us to execute model formulas comprised of multiple models with only marginal overhead over a single model and reduces model calls by up to $`64\%`$. The resulting inference speedup naturally extends to prior CTG techniques that can be expressed in model arithmetic .

#### Key Contributions

Our core contributions include:

- Model Arithmetic: A principled framework for fine-grained CTG, enabling precise control over multiple attributes. Our framework can express many prior CTG approaches (  
  efsec:prompt-arithmetic).

- An extension of speculative sampling to model arithmetic, counteracting the overhead of CTG and enabling efficient inference, which naturally benefits CTG techniques expressible in model arithmetic (  
  efsec:speed).

- An extensive qualitative and quantitative evaluation of model arithmetic (  
  efsec:evaluation). We show that it is more expressive than prior CTG work and outperforms them in toxicity reduction. We demonstrate that our extended speculative sampling reduces model calls by up to $`64\%`$.