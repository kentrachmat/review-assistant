# Take a Step Back: Evoking Reasoning via Abstraction in Large Language ModelsStep-Back Prompting Enables Reasoning Via Abstraction in Large Language ModelsTake a Step Back: Evoking Reasoning via Abstraction in Large Language Models

## Abstract

We present <span class="smallcaps">Step-Back Prompting</span>, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide reasoning, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of <span class="smallcaps">Step-Back Prompting</span> with PaLM-2L, GPT-4 and Llama2-70B models, and observe substantial performance gains on various challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, <span class="smallcaps">Step-Back Prompting</span> improves PaLM-2L performance on MMLU (Physics and Chemistry) by $`7\%`$ and $`11\%`$ respectively, TimeQA by $`27\%`$, and MuSiQue by $`7\%`$.

*The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise. — Edsger W. Dijkstra*  

# Introduction

The field of natural language processing (NLP) is witnessing a ground-breaking revolution because of the Transformer-based   large language models (LLMs)  . Scaling up the model size and pre-training corpus  has brought remarkable improvement in model capabilities and sample efficiency with insights from the scaling law , as well as emergent abilities  such as multi-step reasoning  and instruction following .

<figure id="fig:result_summary">
<img src="./figures/res_sum.png"" />
<figcaption>Strong Performance of <span class="smallcaps">Step-Back Prompting</span>: our proposed Abstraction-and-Reasoning scheme leads to a substantial improvement in a wide range of challenging tasks in STEM, Knowledge QA and Multi-Hop Reasoning requiring complex (often multi-hop) reasoning. </figcaption>
</figure>

Despite the great advancements, complex multi-step reasoning remains challenging for even the state-of-the-art LLMs. show that process-supervision with step-by-step verification is a promising remedy to improve the correctness of intermediate reasoning steps. Techniques such as Chain-of-Thought were introduced to produce a coherent series of intermediate reasoning steps to increase the success rate of following the right decoding path. Inspired by the fact that when faced with challenging tasks humans often step back and do abstractions to arrive at high-level principles to guide the process, we propose <span class="smallcaps">Step-Back Prompting</span> to ground reasoning on abstractions to reduce the chance of making errors in the intermediate reasoning steps.

<figure id="fig:sbt_diagram">
<img src="./figures/Figure1_SBT_Diagram.png"" />
<figcaption>Illustration of <span class="smallcaps">Step-Back Prompting</span> with two steps of Abstraction and Reasoning guided by concepts and principles. <em>Top</em>: an example of MMLU high-school physics <span class="citation" data-cites="hendrycks2020measuring"></span> where the first principle of Ideal Gas Law is retrieved via abstraction. <em>Bottom</em>: an example from TimeQA <span class="citation" data-cites="chen2021dataset"></span> where the high-level concept of education history is a result of the abstraction. <em>Left</em>: PaLM-2L <span class="citation" data-cites="anil2023palm"></span> fails to answer the original question. Chain-of-Thought prompting <span class="citation" data-cites="wei2022chain kojima2022large"></span> ran into errors during intermediate reasoning steps (highlighted as red). <em>Right</em>: PaLM-2L <span class="citation" data-cites="anil2023palm"></span> successfully answers the question via <span class="smallcaps">Step-Back Prompting</span>. </figcaption>
</figure>

Among many of the cognitive skills, abstraction is ubiquitous to humans’ ability to process vast amounts of information and derive general principles. For example, Kepler compressed thousands of measurements into Kepler’s three laws of planetary motion, which precisely describe the orbits of planets around the Sun . In critical decision-making, humans find abstraction to be helpful since it provides a broader view of the environment. This work explores how LLMs can tackle complex tasks involving many low-level details through a two-step process of abstraction-and-reasoning. The first step is to show LLMs how to step back through in-context learning – prompting them to derive high-level abstractions such as concepts and principles for a specific example. The second step is to leverage the reasoning ability to reason on top of the high-level concepts and principles. We use few-shot exemplar demonstrations to execute <span class="smallcaps">Step-Back Prompting</span> on LLMs.

We experiment across a range of tasks involving domain specific reasoning such as Physics and Chemistry, knowledge-intensive question answering requiring factual knowledge, multi-hop commonsense reasoning. We observe significant performance improvements (up to $`27\%`$) in PaLM-2L  demonstrating the efficacy of <span class="smallcaps">Step-Back Prompting</span> in tackling complex tasks, which are otherwise challenging due to the amount of details needed for reasoning. Figure <a href="#fig:result_summary" data-reference-type="ref" data-reference="fig:result_summary">1</a> shows a summary of all the key results presented in this paper. Some the tasks are very challenging: both PaLM-2L and GPT-4 achieve only $`\sim40\%`$ accuracy on TimeQA and MuSiQue. Chain-of-Thought prompting leads to a minor improvement on a few tasks, while <span class="smallcaps">Step-Back Prompting</span> improves the performance of PaLM-2L across the board: $`7\%`$ and $`11\%`$ on MMLU Physics and Chemistry, $`27\%`$ on TimeQA, and $`7\%`$ on MuSiQue.

We conduct a variety of analyses and find that <span class="smallcaps">Step-Back Prompting</span> leads to strong performance improvements (up to $`36\%`$) over chain-of-thought (CoT) prompting  and “take-a-deep-breath” (TDB) prompting . We perform a qualitative evaluation where we find that Step-Back fixes a large portion of errors of the base model (up to $`\sim`$ $`40\%`$) while introducing a small portion of new errors (max $`\sim`$ $`12\%`$). We also conduct an error analysis and find that majority of the errors made by <span class="smallcaps">Step-Back Prompting</span> is attributed to the intrinsic limitations of reasoning capabilities of LLMs while abstraction skills are relatively easy to demonstrate to LLMs, pointing out the direction for future improvements of methods alike <span class="smallcaps">Step-Back Prompting</span>.

# <span class="smallcaps">Step-Back Prompting</span>

<span class="smallcaps">Step-Back Prompting</span> is motivated by the observation that many tasks contain a lot of details, and it is hard for LLMs to retrieve relevant facts to tackle the task. As shown in the first example (top) in Figure <a href="#fig:sbt_diagram" data-reference-type="ref" data-reference="fig:sbt_diagram">2</a>, for a Physics question of “*What happens to the pressure, P, of an ideal gas if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8 ?*”, the LLM can deviate from the first principle of Ideal Gas Law when reasoning directly on the question. Similarly, a question of “*Estella Leopold went to which school between Aug 1954 and Nov 1954?*” is very hard to address directly given the detailed time range constraint. In both cases, asking a step-back question helps the model to solve the problem effectively.

We define a step-back question as a derived question from the original question at a higher level of abstraction. For instance, instead of directly asking “*which school Estella Leopold went to during a specific period*”, a step-back question (Figure <a href="#fig:sbt_diagram" data-reference-type="ref" data-reference="fig:sbt_diagram">2</a> bottom) would ask about the “*education history*”, which is a high-level concept encompasses the original question. Answering the step-back question of “*Estella Leopold’s education history*” in this case will provide all the necessary information to reason about “*which school Estella Leopold went to during a specific period*”. The premise is that the step-back question is typically much easier. Grounding the reasoning on top of such abstractions helps to avoid reasoning errors in the intermediate steps such as the example shown in Figure <a href="#fig:sbt_diagram" data-reference-type="ref" data-reference="fig:sbt_diagram">2</a> (left) from Chain-of-Thought. In short, <span class="smallcaps">Step-Back Prompting</span> consists two simple steps:

- **Abstraction**: Instead of addressing the question directly, we first prompt the LLM to ask a generic step-back question about a higher-level concept or principle, and retrieve relevant facts about the high-level concept or principle. The step-back question is unique for each task in order to retrieve the most relevant facts.

- **Reasoning**: Grounded on the facts regarding the high-level concept or principle, the LLM can reason about the solution to the original question. We term this as *Abstraction-grounded Reasoning*.

In the following sections, we present an empirical study of <span class="smallcaps">Step-Back Prompting</span> on a range of challenging tasks covering STEM, Knowledge QA, and Multi-Hop Reasoning involving complex reasoning.

# Experimental Setup

Here we define the tasks and models we experiment with. We also describe our evaluation metric and the baselines we consider.

## Tasks

We experiment with the following diverse tasks: (a) STEM, (b) Knowledge QA, and (c) Multi-Hop Reasoning. We describe below the datasets we consider (see Appendix <a href="#app:dataset_details" data-reference-type="ref" data-reference="app:dataset_details">11</a> for more details).

- **STEM**: We evaluate MMLU and GSM8K for STEM tasks. MMLU contains a series of benchmarks across diverse domains to evaluate the model’s language understanding. We consider the high school physics and chemistry portions of MMLU because of the deep reasoning involved.

- **Knowledge QA**: We consider TimeQA  since it contains complex queries that require challenging time-sensitive knowledge. We also experiment with SituatedQA , another challenging open-retrieval QA dataset requiring the model to answer questions given temporal or geographical contexts.

- **Multi-Hop Reasoning**: We experiment with MuSiQue , a hard multihop reasoning dataset created via composable pairs of single-hop questions, and StrategyQA with open-domain questions that demand some strategy to solve.

## Models

We use the following state-of-the-art LLMs: instruction-tuned PaLM-2L , GPT-4 , and Llama2-70B .

## Evaluation

Conventional evaluation metrics such as accuracy, F1 score have limitations specifically for evaluating the generations of state-of-the-art LLMs since these models often generate long-form answers which are hard to capture. We instead conduct an evaluation using the PaLM-2L model where we few-shot prompt the model to identify equivalence between target answers and the model predictions. Few-shot examples, prompts and other details used for this evaluation are in Appendix <a href="#app:eval" data-reference-type="ref" data-reference="app:eval">12</a>.

## Baseline Methods

- **PaLM-2L, PaLM-2L 1-shot**: PaLM-2L is either queried directly with the question or has a single demonstration exemplar of question-answer included in the prompt.

- **PaLM-2L + CoT, PaLM-2L + CoT 1-shot**: PaLM-2L model is queried with zero-shot CoT prompting : “*Let’s think step by step*” is appended to the question. For 1-shot, One demonstration example of a question and answer pair is provided in the prompt, where the answer is in the style of CoT .

- **PaLM-2L + TDB**: Zero-shot prompting with “*Take a deep breath and work on this problem step-by-step.*” prepended to the question.

- **PaLM-2L + RAG**: For Sections <a href="#sec:timeqa" data-reference-type="ref" data-reference="sec:timeqa">5</a> and <a href="#sec:multihop_reasoning" data-reference-type="ref" data-reference="sec:multihop_reasoning">6</a>, we use retrieval-augmented generation (RAG) where the retrieved passage is used as context by the LLM.

- **GPT-4 and Llama2-70B**: we run GPT-4 and Llama2-70B on MMLU tasks for all methods. In addition, we also run GPT-4 on all baselines for all tasks.

We do not use RAG for STEM tasks, because of the inherent reasoning nature of the tasks contrary to the other fact-seeking datasets. All inferences are done using greedy decoding.

<div class="center">

<div id="table:mmlu_physics">

| Method                        |   MMLU Physics   |  MMLU Chemistry  |
|:------------------------------|:----------------:|:----------------:|
| PaLM-2L                       |   66.4% (0.8%)   |   70.9% (0.9%)   |
| PaLM-2L 1-shot                |    64% (1.6%)    |   75.6% (0.4%)   |
| PaLM-2L + CoT                 |     65% (2%)     |   75.3% (1.5%)   |
| PaLM-2L + CoT 1-shot          |   61.5% (1.8%)   |    76.6% (1%)    |
| PaLM-2L + TDB                 |   65.7% (0.7%)   |   73.8% (1.1%)   |
| PaLM-2L + Step-Back (ours)    | **73.2%** (1.9%) | **81.8**% (1.4%) |
| GPT-4                         |   69.4% (2.0%)   |   80.9% (0.7%)   |
| GPT-4 1-shot                  |   78.4% (2.4%)   |   80.5% (1.6%)   |
| GPT-4 + CoT                   |   82.9% (0.5%)   |   85.3% (1.0%)   |
| GPT-4 + CoT 1-shot            |   79.3% (1.0%)   |   82.8% (0.5%)   |
| GPT-4 + TDB                   |   74.4% (4.0%)   |   81.5% (1.3%)   |
| GPT-4 + Step-Back (ours)      | **84.5%** (1.2%) | **85.6**% (1.4%) |
| Llama2-70B                    |   51.9% (3.6%)   |   55.7% (2.1%)   |
| Llama2-70B 1-shot             |   57.3% (1.6%)   |   58.5% (2.5%)   |
| Llama2-70B + CoT              |   59.3% (2.0%)   |   64.1% (1.2%)   |
| Llama2-70B + CoT 1-shot       |   59.6% (2.0%)   | **68.1%** (1.4%) |
| Llama2-70B + TDB              |   60.4% (2.1%)   |   63.6% (1.9%)   |
| Llama2-70B + Step-Back (ours) | **64.8%** (1.5%) |   66.7% (1.6%)   |

Strong performance of <span class="smallcaps">Step-Back Prompting</span> on MMLU tasks across three model families. CoT: zero-shot Chain of Thought prompting , TDB: Take a Deep Breath prompting .

</div>

</div>

# STEM

We evaluate <span class="smallcaps">Step-Back Prompting</span> on STEM tasks to gauge the efficacy of our method on reasoning in highly specialized domains. We explain below our experimental setup, result, and analysis of applying <span class="smallcaps">Step-Back Prompting</span> on the MMLU high-school Physics and Chemistry, and GSM8K benchmarks.

## <span class="smallcaps">Step-Back Prompting</span>

Questions in the MMLU benchmarks require deeper reasoning. Furthermore, they also require understanding and application of formulae which are often physics and chemistry principles and concepts. In this case, we first demonstrate to the model abstraction skills in the form of concepts and first principles such as *Newton’s first law of motion*, *Doppler effect*, and *Gibbs free energy* etc. The implicit step-back question here is “*what are the physics or chemistry principles and concepts involved in solving this task?*”. We provide demonstrations to the model to recite the relevant principles for solving the task from its own knowledge (see Appendix <a href="#app:fewmmlu" data-reference-type="ref" data-reference="app:fewmmlu">13.1</a> for few-shot exemplars).

<figure id="fig:mmlu_shot_abalation">
<img src="./figures/mmlu_shot_ablation.png"" style="width:50.0%" />
<figcaption>Ablation study of <span class="smallcaps">Step-Back Prompting</span> accuracy using PaLM-2L on MMLU high-school Physics against the number of few shot exemplars: robust performance with respect to a varying number of shots. </figcaption>
</figure>

## Results

Table  <a href="#table:mmlu_physics" data-reference-type="ref" data-reference="table:mmlu_physics">1</a> illustrates model performance across various setups across three model families: PaLM-2L, GPT-4, and Llama2-70B. Average accuracy over 5 evaluation runs is reported along with standard deviations (in the parentheses). PaLM-2L baseline performance is $`66.4\%`$ and 70.9% on Physics and Chemistry, respectively. We find that CoT and TDB zero-shot prompting do not significantly increase model performance, which could be due to the inherent difficulty and deep reasoning associated with these tasks. PaLM-2L 1-shot and PaLM-2L + CoT 1-shot do not improve against the baseline much either, highlighting the challenge of demonstrating the reasoning steps to the model. In contrast, <span class="smallcaps">Step-Back Prompting</span> significantly improves model performance: +7% and +11% compared to PaLM-2L. Similarly, with GPT-4 and Llama2-70B models, <span class="smallcaps">Step-Back Prompting</span> is very competitive among all the baseline methods we tested, showing that <span class="smallcaps">Step-Back Prompting</span> is model-agnostic. We present the results of GSM8K in Appendix <a href="#app:gsm8k" data-reference-type="ref" data-reference="app:gsm8k">10.1</a>.

## Ablation and Analysis

**Few-shot Ablation**: First, in Figure <a href="#fig:mmlu_shot_abalation" data-reference-type="ref" data-reference="fig:mmlu_shot_abalation">3</a>, we observe that <span class="smallcaps">Step-Back Prompting</span> is robust to the number of few-shot exemplars of (question, principles) pairs used as demonstrations. Adding more demonstration examples beyond a single example does not lead to further improvements. This indicates that the task of retrieving the relevant principles and concepts is relatively easy through in-context learning and a single demonstration suffices. Therefore, we use a single exemplar for few-shot prompting throughout the paper except the ablation studies.

**Error Analysis**: Comparing the predictions of <span class="smallcaps">Step-Back Prompting</span> to the baseline PaLM-2L model for MMLU high-school Physics: we find that <span class="smallcaps">Step-Back Prompting</span> corrects $`20.5\%`$ errors from the baseline while introducing $`11.9\%`$ errors.

To further understand where the errors come from in <span class="smallcaps">Step-Back Prompting</span>, we annotate all the wrong predictions of <span class="smallcaps">Step-Back Prompting</span> in the test set, and categorize them into 5 classes (see Appendix <a href="#app:mmlu_errors" data-reference-type="ref" data-reference="app:mmlu_errors">14.1</a> for examples in each class):

- **Principle Error**: The error happens at the step of Abstraction, where the first principles generated by models are wrong or incomplete.

- **Factual Error**: There is at least one factual error when the model recites its own factual knowledge

- **Math Error**: There is at least one math error in the intermediate steps when math calculations are involved in deriving the final answer.

- **Context Loss**: There is at least one error where the model response loses context from the question, and deviates from addressing the original question

- **Reasoning Error**: We define Reasoning Error as when the model makes at least one error in the intermediate Reasoning steps before arriving at the final answer.

<figure id="fig:mmlu_ea">
<img src="./figures/MMLU_EA.png"" style="width:50.0%" />
<figcaption>Error Analysis of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics: five classes of errors Step-Back makes with Reasoning being the dominating class. </figcaption>
</figure>

All five types of errors are happening during the Reasoning step except *Principle Error* which points to the failure of the Abstraction step. As shown in Figure <a href="#fig:mmlu_ea" data-reference-type="ref" data-reference="fig:mmlu_ea">4</a> (right), *Principle Error* comprises only a small fraction of the errors the model makes: more than $`90\%`$ of the errors happen at the Reasoning step. Among the four error types during Reasoning, *Reasoning Error* and *Math Error* are the major error categories. This corroborates with the finding in the ablation study above that very few exemplars are needed to demonstrate to LLMs the Abstraction skill. Reasoning step is still the bottleneck of how well <span class="smallcaps">Step-Back Prompting</span> can perform tasks such as MMLU requiring complex reasoning. For MMLU Physics specifically, the Reasoning and Math skills are critical for solving the problems successfully: even if the first principles are retrieved correctly, deep reasoning and math are involved to derive a correct final answer through a typical multi-step reasoning process.

<div class="center">

<div id="table:cbqa">

| Method | TimeQA | TQA Easy | TQA Hard | SituatedQA |
|:---|:--:|:--:|:--:|:--:|
| PaLM-2L | 41.5% | 42.6% | 40.4% | 54.3% (0.3%) |
| PaLM-2L 1-shot | 40.7% | 41.7% | 39.1% | 51.8% (0.6%) |
| PaLM-2L + CoT | 40.8% | 41.8% | 39.8% | 56.4% (0.2%) |
| PaLM-2L + CoT 1-shot | 38.1% | 39.3% | 36.8% | 54% (0.8%) |
| PaLM-2L + TDB | 40.9% | 42.6% | 39.1% | 54% (0.5%) |
| PaLM-2L + RAG | 57.4% | 67.8% | 46.8% | 59.3% (0.4%) |
| PaLM-2L + Step-Back (ours) | 66% | 70.4% | 61.6% | 57.5% (0.3%) |
| PaLM-2L + Step-Back + RAG (ours) | **68.7%** | **75.2%** | **62.3%** | 61% (0.4%) |
| GPT-4 | 45.6% | 48.9% | 42.6% | **63.2%** (0.4%) |

Strong performance of <span class="smallcaps">Step-Back Prompting</span> on Knowledge QA tasks. CoT: Chain of Thought prompting, TDB: Take a Deep Breath prompting, RAG: retrieval-augmented generation. <span class="smallcaps">Step-Back Prompting</span> results in significant performance improvements.

</div>

</div>

# Knowledge QA

We evaluate <span class="smallcaps">Step-Back Prompting</span> on question-answering benchmarks requiring intensive factual knowledge. Knowledge QA has been challenging for LLMs. In this section, we first describe the experimental setup, followed by results and analysis on <span class="smallcaps">Step-Back Prompting</span>.

## <span class="smallcaps">Step-Back Prompting</span>

We evaluate <span class="smallcaps">Step-Back Prompting</span> on TimeQA  and SituatedQA  in the Knowledge QA category. We first show the LLMs how to do Abstraction through in-context demonstrations. The step-back question “*What was Estella Leopold’s education history*” in Figure <a href="#fig:sbt_diagram" data-reference-type="ref" data-reference="fig:sbt_diagram">2</a> is generated by the LLM through few-shot demonstrations (see Appendix <a href="#app:fewtimeqa" data-reference-type="ref" data-reference="app:fewtimeqa">13.2</a> for details). Given the knowledge-intensive nature of these queries, we use retrieval augmentation (RAG) in combination with <span class="smallcaps">Step-Back Prompting</span>. The step-back question is used to retrieve relevant facts, which work as additional context (see Table <a href="#tab:timeqa_final_prompt" data-reference-type="ref" data-reference="tab:timeqa_final_prompt">14</a> for the prompt) to ground the final reasoning step.

## Results

We evaluate the models on the test set of TimeQA. As shown in Table <a href="#table:cbqa" data-reference-type="ref" data-reference="table:cbqa">2</a>, the baseline models of GPT-4 and PaLM-2L achieved $`45.6\%`$ and $`41.5\%`$, highlighting the difficulty of the task. Applying either CoT or TDB zero-shot (and one-shot) prompting to the baseline model shows no improvement. In contrast, augmenting the baseline model by regular retrieval augmentation (RAG) improves the accuracy to $`57.4\%`$, highlighting the fact-intensive nature of the task. The result of Step-Back + RAG shows the effectiveness of going back to a high-level concept, which enables much more reliable retrieval augmentation: the accuracy on TimeQA achieves a remarkable $`68.7\%`$.

Next, we segment TimeQA into the Easy and Hard difficulty levels provided in the original dataset. As expected, all methods perform worse on the Hard subset. While RAG can improve the Easy accuracy from $`42.6\%`$ to $`67.8\%`$, the improvement is much smaller on the Hard accuracy: $`40.4\%`$ to $`46.8\%`$. This is where <span class="smallcaps">Step-Back Prompting</span> shines by retrieving facts regarding high-level concepts to ground the final reasoning: Step-Back + RAG further improves the Hard accuracy to $`62.3\%`$, outperforming GPT-4’s $`42.6\%`$ from GPT-4. We hypothesize that facts regarding the high-level concepts (such as *education history*) are much more accessible than the low-level details.

On the SituatedQA benchmark, we observe a moderate quality gain from $`54.3\%`$ to our best method of Step-Back + RAG ($`61\%`$) with a small gap to GPT-4’s $`63.2\%`$. Similar to TimeQA, prompting techniques such as CoT and TDB don’t help significantly for SituatedQA.

<figure id="fig:timeqa_error_analysis">
<img src="./figures/TimeQA_Merged_EA.png"" style="width:90.0%" />
<figcaption>Ablation and error analysis of <span class="smallcaps">Step-Back Prompting</span> on TimeQA. <em>Left</em>: ablation against the number of few-shot exemplars. <em>Right</em>: four classes of errors Step-Back makes with Reasoning and RAG being the dominant error sources. </figcaption>
</figure>

## Ablation and Analysis

**Few-shot Ablation**: We observe in Figure <a href="#fig:timeqa_error_analysis" data-reference-type="ref" data-reference="fig:timeqa_error_analysis">5</a> (left) that the performance of <span class="smallcaps">Step-Back Prompting</span> on TimeQA is robust to the number of exemplars used in demonstration, highlighting again the sample efficiency of in-context learning Abstraction skills for models like PaLM-2L.

**Error Analysis:** Figure <a href="#fig:timeqa_error_analysis" data-reference-type="ref" data-reference="fig:timeqa_error_analysis">5</a> (right) shows the breakdown of all the remaining errors made by <span class="smallcaps">Step-Back Prompting</span> on TimeQA. Similar to Section <a href="#sec:stem_analysis" data-reference-type="ref" data-reference="sec:stem_analysis">4.3</a>, we categorize the errors into

- **StepBack**: The step-back question generated is not helpful in solving the task.

- **RAG**: RAG fails to retrieve relevant information despite that the step-back question is on target.

- **Scoring Error**: The evaluation by the judge model made a mistake.

- **Reasoning Error**: The retrieved context is relevant, but the model still fails to reason through the context to arrive at the right answer.

We find that the StepBack rarely fails. In contrast, we find more than half of the errors are due to reasoning errors. Additionally, $`45\%`$ of errors are due to failure in retrieving the right information despite that Abstraction provided by step-back makes it a much easier task. This reflects the difficulty level of the TimeQA task. Additional error analysis of TimeQA is in Appendix <a href="#app:timeqaerrora" data-reference-type="ref" data-reference="app:timeqaerrora">10</a>.

<div class="center">

<div id="table:strategyqa">

|     | Method                           |     MuSiQue      |   StrategyQA   |
|:----|:---------------------------------|:----------------:|:--------------:|
|     | PaLM-2L                          |    35.5% (3%)    |  82.8% (0.7%)  |
|     | PaLM-2L 1-shot                   |   29.0% (0.5%)   |  76.6% (0.5%)  |
|     | PaLM-2L + CoT                    |   38.7% (3.2%)   |  83.6% (0.4%)  |
|     | PaLM-2L + CoT 1-shot             |   38.5% (2.2%)   |  76.8% (1.4%)  |
|     | PaLM-2L + TDB                    |   39.0% (2.3%)   |  82.7% (0.9%)  |
|     | PaLM-2L + RAG                    |   39.6% (2.8%)   |  84.2% (0.5%)  |
|     | PaLM-2L + Step-Back (ours)       |   42.6% (3.1%)   |  82.7% (0.4%)  |
|     | PaLM-2L + Step-Back + RAG (ours) | **42.8**% (2.0%) | **86.4%** (1%) |
|     | GPT-4                            |   38.5% (0.2%)   |  78.3% (1.1%)  |

Results of <span class="smallcaps">Step-Back Prompting</span> on Multi-Hop Reasoning. CoT: Chain of Thought prompting, TDB: Take a Deep Breath prompting, RAG: retrieval augmentation generation. The average accuracy is over 5 evaluation runs with the standard deviations included in the parentheses.

</div>

</div>

# Multi-Hop Reasoning

We evaluate <span class="smallcaps">Step-Back Prompting</span> on challenging Multi-Hop reasoning benchmark MuSiQue and StrategyQA . We follow the same protocol as Section <a href="#sec:timeqa" data-reference-type="ref" data-reference="sec:timeqa">5</a> to implement <span class="smallcaps">Step-Back Prompting</span>.

Table  <a href="#table:strategyqa" data-reference-type="ref" data-reference="table:strategyqa">3</a> shows performance of various baselines on the dev set of MuSiQue and StrategyQA. Baseline performance of PaLM-2L and GPT-4 are low ($`35.5\%`$ and $`38.5\%`$ for PaLM-2L and GPT-4 respectively) in MuSiQue since it is a hard multihop reasoning benchmark. In contrast, StrategyQA has stronger baselines ($`82.8\%`$ and $`78.3\%`$ for PaLM-2L and GPT-4 respectively) probably because it is a binary classification task. CoT and TDB improve model performance a bit in the case of MuSiQue ($`\sim`$ 3% and 3.5% respectively) which can be attributed to the inherent reasoning nature of this task where these methods are shown to be helpful. In the case of StrategyQA, there is no significant performance gain with CoT and TDB which could be due to the high baseline performance in this task, with limited scope for these prompting methods to improve performance. Often, 1-shot performance is significantly lower than their zero-shot methods, which could be attributed to potential example bias . RAG improves model performance ($`\sim`$ 4% and 2% for MuSiQue and StrategyQA respectively.) <span class="smallcaps">Step-Back Prompting</span> with the power of abstraction produces the best performance of all methods: $`42.8\%`$ in MuSiQue and $`86.4\%`$ in StrategyQA, significantly outperforming GPT-4 on both tasks. We present a detailed error analysis on StrategyQA in Appendix <a href="#app:startegyqaerrora" data-reference-type="ref" data-reference="app:startegyqaerrora">10.3</a>.

# Discussion

Abstraction helps humans to solve complex tasks by removing irrelevant details and distilling high-level concepts and principles to guide the problem-solving process. <span class="smallcaps">Step-Back Prompting</span> breaks complex tasks such as knowledge-intensive QA, multi-hop reasoning, and science questions into two separate steps of Abstraction and Reasoning. We demonstrate through empirical experiments that Abstraction is an easy skill for the LLMs such as PaLM-2L via sample-efficient in-context learning. Grounding on the high-level concepts and principles, LLMs can leverage their intrinsic Reasoning capabilities to derive the solution. This reduces the chance of reasoning failures in the intermediate steps and is shown to improve the performance on a wide range of complex reasoning tasks. Despite the success, through error analysis, we find that Reasoning is still one of the hardest skills for LLMs to acquire: it is still the dominant failure mode even after the large reduction of task complexity by <span class="smallcaps">Step-Back Prompting</span>.

Nevertheless, Abstraction is neither necessary nor possible in all scenarios. For instance, the task can be as simple as *who was the president of the United States in 2000?*, in which case there is no such need to step back and ask a high-level question as the answer to such questions is readily available. Questions such as *what is the speed of light?* point to the first principles themselves. Doing Abstraction in this case would not make a difference either.

# Related Work

## Prompting

Few-shot prompting  has significantly improved model performance across a range of tasks without requiring updating any model parameters. Our work <span class="smallcaps">Step-Back Prompting</span> is in the same category as the chain-of-thought prompting   and scratchpad  owing to its simplicity and generic nature. But our approach is focused on the key idea of abstraction which is inspired from the fact that taking a step back often helps humans in performing complex tasks. Our work is also related to the recitation-augmented language models ; however in contrast to their work, we explicitly perform step-back and abstraction, with optional use of retrieval augmentation depending on the nature of the task at hand.

## Decomposition

Decomposing a task into simpler tasks and solving these tasks to complete the original task has been an effective way  to improve model performance on complex tasks. Several prompting methods have been successful in this regard. Our work <span class="smallcaps">Step-Back Prompting</span>, in contrast, is on making the question more abstract and high-level, which is different from decomposition that is often a low-level breakdowns of the original question. For instance, a generic question for *which employer did Steve Jobs work for in 1990?* could be *what is the employment history of Steve Jobs?* While decomposition would lead to sub-questions such as *What was Steve Jobs doing in 1990?*, *Was Steve Jobs employed in 1990?* and *If Steve Jobs was employed, who was his employer?* Furthermore, abstract questions such as *what is the employment history of Steve Jobs?* are often generic in nature to have a many-to-one mapping since many questions (e.g. *which employer did Steve Jobs work for in 1990?* and *which employer did Steve Jobs work for in 2000?*) can have the same abstract question. This is in contrast to decomposition where there is often a one-to-many mapping since there are multiple decomposed sub-problems necessary to solve a given question.

# Conclusion

We introduce <span class="smallcaps">Step-Back Prompting</span> as a simple yet generic method to elicit deep reasoning via abstraction in large language models. Experimentation on LLMs across fact-seeking, commonsense reasoning and domain-specific reasoning benchmarks shows that <span class="smallcaps">Step-Back Prompting</span> significantly improves model performance. We hypothesize that abstraction helps models to hallucinate less and reason better, probably reflecting the true nature of the model which are often hidden while responding to the original question without abstraction. We hope our work will inspire more human-inspired approaches to elicit the hidden potential of large language models.

# Acknowledgement

We thank Andrew Dai, Adams Yu and Hannah Rashkin of Google DeepMind for their insightful feedback on this paper.

# References

<div class="thebibliography">

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al Palm 2 technical report *arXiv preprint arXiv:2305.10403*, 2023. **Abstract:** We introduce PaLM 2, a new state-of-the-art language model that has better multilingual and reasoning capabilities and is more compute-efficient than its predecessor PaLM. PaLM 2 is a Transformer-based model trained using a mixture of objectives. Through extensive evaluations on English and multilingual language, and reasoning tasks, we demonstrate that PaLM 2 has significantly improved quality on downstream tasks across different model sizes, while simultaneously exhibiting faster and more efficient inference compared to PaLM. This improved efficiency enables broader deployment while also allowing the model to respond faster, for a more natural pace of interaction. PaLM 2 demonstrates robust reasoning capabilities exemplified by large improvements over PaLM on BIG-Bench and other reasoning tasks. PaLM 2 exhibits stable performance on a suite of responsible AI evaluations, and enables inference-time control over toxicity without additional overhead or impact on other capabilities. Overall, PaLM 2 achieves state-of-the-art performance across a diverse set of tasks and capabilities. When discussing the PaLM 2 family, it is important to distinguish between pre-trained models (of various sizes), fine-tuned variants of these models, and the user-facing products that use these models. In particular, user-facing products typically include additional pre- and post-processing steps. Additionally, the underlying models may evolve over time. Therefore, one should not expect the performance of user-facing products to exactly match the results reported in this report. (@anil2023palm)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al Language models are few-shot learners *Advances in neural information processing systems*, 33: 1877–1901, 2020. **Abstract:** Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general. (@brown2020language)

Wenhu Chen, Xinyi Wang, and William Yang Wang A dataset for answering time-sensitive questions *arXiv preprint arXiv:2108.06314*, 2021. **Abstract:** Time is an important dimension in our physical world. Lots of facts can evolve with respect to time. For example, the U.S. President might change every four years. Therefore, it is important to consider the time dimension and empower the existing QA models to reason over time. However, the existing QA datasets contain rather few time-sensitive questions, hence not suitable for diagnosing or benchmarking the model’s temporal reasoning capability. In order to promote research in this direction, we propose to construct a time-sensitive QA dataset. The dataset is constructed by 1) mining time-evolving facts from WikiData and aligning them to their corresponding Wikipedia page, 2) employing crowd workers to verify and calibrate these noisy facts, 3) generating question-answer pairs based on the annotated time-sensitive facts. Our dataset poses challenges in the aspect of both temporal understanding and temporal reasoning. We evaluate different SoTA long-document QA systems like BigBird and FiD on our dataset. The best-performing model FiD can only achieve 46\\}% accuracy, still far behind the human performance of 87\\}%. We demonstrate that these models are still lacking the ability to perform consistent temporal reasoning. Therefore, we believe that our dataset could serve as a benchmark to develop NLP models more sensitive to temporal shifts. The dataset and code are released in~\\}url{https://github.com/wenhuchen/Time-Sensitive-QA}. (@chen2021dataset)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al Palm: Scaling language modeling with pathways *arXiv preprint arXiv:2204.02311*, 2022. **Abstract:** Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies. (@chowdhery2022palm)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova Bert: Pre-training of deep bidirectional transformers for language understanding *arXiv preprint arXiv:1810.04805*, 2018. **Abstract:** We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement). (@devlin2018bert)

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies *Transactions of the Association for Computational Linguistics*, 9: 346–361, 2021. **Abstract:** A key limitation in current datasets for multi-hop reasoning is that the required steps for answering the question are mentioned in it explicitly. In this work, we introduce StrategyQA, a question answering (QA) benchmark where the required reasoning steps are implicit in the question, and should be inferred using a strategy. A fundamental challenge in this setup is how to elicit such creative questions from crowdsourcing workers, while covering a broad range of potential strategies. We propose a data collection procedure that combines term-based priming to inspire annotators, careful control over the annotator population, and adversarial filtering for eliminating reasoning shortcuts. Moreover, we annotate each question with (1) a decomposition into reasoning steps for answering it, and (2) Wikipedia paragraphs that contain the answers to each step. Overall, StrategyQA includes 2,780 examples, each consisting of a strategy question, its decomposition, and evidence paragraphs. Analysis shows that questions in StrategyQA are short, topic-diverse, and cover a wide range of strategies. Empirically, we show that humans perform well (87%) on this task, while our best baseline reaches an accuracy of $\\}sim$66%. (@geva2021did)

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt Measuring massive multitask language understanding *arXiv preprint arXiv:2009.03300*, 2020. **Abstract:** We propose a new test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model’s academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings. (@hendrycks2020measuring)

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al Training compute-optimal large language models *arXiv preprint arXiv:2203.15556*, 2022. **Abstract:** We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4$\\}times$ more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher. (@hoffmann2022training)

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei Scaling laws for neural language models *arXiv preprint arXiv:2001.08361*, 2020. **Abstract:** We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence. (@kaplan2020scaling)

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal Decomposed prompting: A modular approach for solving complex tasks *arXiv preprint arXiv:2210.02406*, 2022. **Abstract:** Few-shot prompting is a surprisingly powerful way to use Large Language Models (LLMs) to solve various tasks. However, this approach struggles as the task complexity increases or when the individual reasoning steps of the task themselves are hard to learn, especially when embedded in more complex tasks. To address this, we propose Decomposed Prompting, a new approach to solve complex tasks by decomposing them (via prompting) into simpler sub-tasks that can be delegated to a library of prompting-based LLMs dedicated to these sub-tasks. This modular structure allows each prompt to be optimized for its specific sub-task, further decomposed if necessary, and even easily replaced with more effective prompts, trained models, or symbolic functions if desired. We show that the flexibility and modularity of Decomposed Prompting allows it to outperform prior work on few-shot prompting using GPT3. On symbolic reasoning tasks, we can further decompose sub-tasks that are hard for LLMs into even simpler solvable sub-tasks. When the complexity comes from the input length, we can recursively decompose the task into the same task but with smaller inputs. We also evaluate our approach on textual multi-step reasoning tasks: on long-context multi-hop QA task, we can more effectively teach the sub-tasks via our separate sub-tasks prompts; and on open-domain multi-hop QA, we can incorporate a symbolic information retrieval within our decomposition framework, leading to improved performance on both tasks. Datasets, Code and Prompts available at https://github.com/allenai/DecomP. (@khot2022decomposed)

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa Large language models are zero-shot reasoners *Advances in neural information processing systems*, 35: 22199–22213, 2022. **Abstract:** Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-by-step answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs’ ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding "Let’s think step by step" before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large InstructGPT model (text-davinci-002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars. (@kojima2022large)

Royi Lachmy, Valentina Pyatkin, Avshalom Manevich, and Reut Tsarfaty Draw me a flower: Processing and grounding abstraction in natural language *Transactions of the Association for Computational Linguistics*, 2022. **Abstract:** Abstract Abstraction is a core tenet of human cognition and communication. When composing natural language instructions, humans naturally evoke abstraction to convey complex procedures in an efficient and concise way. Yet, interpreting and grounding abstraction expressed in NL has not yet been systematically studied in NLP, with no accepted benchmarks specifically eliciting abstraction in NL. In this work, we set the foundation for a systematic study of processing and grounding abstraction in NLP. First, we deliver a novel abstraction elicitation method and present Hexagons, a 2D instruction-following game. Using Hexagons we collected over 4k naturally occurring visually-grounded instructions rich with diverse types of abstractions. From these data, we derive an instruction-to-execution task and assess different types of neural models. Our results show that contemporary models and modeling practices are substantially inferior to human performance, and that model performance is inversely correlated with the level of abstraction, showing less satisfying performance on higher levels of abstraction. These findings are consistent across models and setups, confirming that abstraction is a challenging phenomenon deserving further attention and study in NLP/AI research. (@lachmy2022)

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe Let’s verify step by step *arXiv preprint arXiv:2305.20050*, 2023. **Abstract:** In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model. (@lightman2023let)

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing *ACM Computing Surveys*, 55 (9): 1–35, 2023. **Abstract:** This article surveys and organizes research works in a new paradigm in natural language processing, which we dub “prompt-based learning.” Unlike traditional supervised learning, which trains a model to take in an input x and predict an output y as P ( y\|x ), prompt-based learning is based on language models that model the probability of text directly. To use these models to perform prediction tasks, the original input x is modified using a template into a textual string prompt x′ that has some unfilled slots, and then the language model is used to probabilistically fill the unfilled information to obtain a final string x̂ , from which the final output y can be derived. This framework is powerful and attractive for a number of reasons: It allows the language model to be pre-trained on massive amounts of raw text, and by defining a new prompting function the model is able to perform few-shot or even zero-shot learning, adapting to new scenarios with few or no labeled data. In this article, we introduce the basics of this promising paradigm, describe a unified set of mathematical notations that can cover a wide variety of existing work, and organize existing work along several dimensions, e.g., the choice of pre-trained language models, prompts, and tuning strategies. To make the field more accessible to interested beginners, we not only make a systematic review of existing works and a highly structured typology of prompt-based concepts but also release other resources, e.g., a website NLPedia–Pretrain including constantly updated survey and paperlist. (@liu2023pre)

Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi Reframing instructional prompts to gptk’s language In *Findings of the Association for Computational Linguistics: ACL 2022*, pp. 589–612, 2022. **Abstract:** What kinds of instructional prompts are easier to follow for Language Models (LMs)? We study this question by conducting extensive empirical analysis that shed light on important features of successful instructional prompts. Specifically, we study several classes of reframing techniques for manual reformulation of prompts into more effective ones. Some examples include decomposing a complex task instruction into multiple simpler tasks or itemizing instructions into sequential steps. Our experiments compare the zero-shot and few-shot performance of LMs prompted with reframed instructions on 12 NLP tasks across 6 categories. Compared with original instructions, our reframed instructions lead to significant improvements across LMs with different sizes. For example, the same reframed prompts boost few-shot performance of GPT3-series and GPT2-series by 12.5% and 6.7% respectively averaged over all tasks. Furthermore, reframed instructions reduce the number of examples required to prompt LMs in the few-shot setting. We hope these empirically-driven techniques will pave the way towards more effective future prompting algorithms. (@mishra2022reframing)

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi Cross-task generalization via natural language crowdsourcing instructions In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3470–3487, 2022. **Abstract:** Humans (e.g., crowdworkers) have a remarkable ability in solving different tasks, by simply reading textual instructions that define them and looking at a few examples. Despite the success of the conventional supervised learning on individual datasets, such models often struggle with generalization across tasks (e.g., a question-answering system cannot solve classification tasks). A long-standing challenge in AI is to build a model that learns a new task by understanding the human-readable instructions that define it. To study this, we introduce NATURAL INSTRUCTIONS, a dataset of 61 distinct tasks, their human-authored instructions, and 193k task instances (input-output pairs). The instructions are obtained from crowdsourcing instructions used to create existing NLP datasets and mapped to a unified schema. Using this meta-dataset, we measure cross-task generalization by training models on seen tasks and measuring generalization to the remaining unseen ones. We adopt generative pre-trained language models to encode task-specific instructions along with input and generate task output. Our results indicate that models benefit from instructions when evaluated in terms of generalization to unseen tasks (19% better for models utilizing instructions). These models, however, are far behind an estimated performance upperbound indicating significant room for more progress in this direction. (@mishra2022cross)

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al Show your work: Scratchpads for intermediate computation with language models *arXiv preprint arXiv:2112.00114*, 2021. **Abstract:** Large pre-trained language models perform remarkably well on tasks that can be done "in one pass", such as generating realistic text or synthesizing computer programs. However, they struggle with tasks that require unbounded multi-step computation, such as adding integers or executing programs. Surprisingly, we find that these same models are able to perform complex multi-step computations – even in the few-shot regime – when asked to perform the operation "step by step", showing the results of intermediate computations. In particular, we train transformers to perform multi-step computations by asking them to emit intermediate computation steps into a "scratchpad". On a series of increasingly complex tasks ranging from long addition to the execution of arbitrary programs, we show that scratchpads dramatically improve the ability of language models to perform multi-step computations. (@nye2021show)

OpenAI Gpt-4 technical report *arXiv preprint arXiv:2303.08774*, 2023. **Abstract:** We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained with no more than 1/1,000th the compute of GPT-4. (@gpt4)

Mihir Parmar, Swaroop Mishra, Mor Geva, and Chitta Baral Don’t blame the annotator: Bias already starts in the annotation instructions In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pp. 1771–1781, 2023. **Abstract:** In recent years, progress in NLU has been driven by benchmarks. These benchmarks are typically collected by crowdsourcing, where annotators write examples based on annotation instructions crafted by dataset creators. In this work, we hypothesize that annotators pick up on patterns in the crowdsourcing instructions, which bias them to write many similar examples that are then over-represented in the collected data. We study this form of bias, termed instruction bias, in 14 recent NLU benchmarks, showing that instruction examples often exhibit concrete patterns, which are propagated by crowdworkers to the collected data. This extends previous work (Geva et al., 2019) and raises a new concern of whether we are modeling the dataset creator’s instructions, rather than the task. Through a series of experiments, we show that, indeed, instruction bias can lead to overestimation of model performance, and that models struggle to generalize beyond biases originating in the crowdsourcing instructions. We further analyze the influence of instruction bias in terms of pattern frequency and model size, and derive concrete recommendations for creating future NLU benchmarks. (@parmar2023don)

Pruthvi Patel, Swaroop Mishra, Mihir Parmar, and Chitta Baral Is a question decomposition unit all we need? In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 4553–4569, 2022. **Abstract:** Large Language Models (LMs) have achieved state-of-the-art performance on many Natural Language Processing (NLP) benchmarks. With the growing number of new benchmarks, we build bigger and more complex LMs. However, building new LMs may not be an ideal option owing to the cost, time and environmental impact associated with it. We explore an alternative route: can we modify data by expressing it in terms of the model’s strengths, so that a question becomes easier for models to answer? We investigate if humans can decompose a hard question into a set of simpler questions that are relatively easier for models to solve. We analyze a range of datasets involving various forms of reasoning and find that it is indeed possible to significantly improve model performance (24% for GPT3 and 29% for RoBERTa-SQuAD along with a symbolic calculator) via decomposition. Our approach provides a viable option to involve people in NLP research in a meaningful way. Our findings indicate that Human-in-the-loop Question Decomposition (HQD) can potentially provide an alternate path to building large LMs. (@patel2022question)

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis Measuring and narrowing the compositionality gap in language models *arXiv preprint arXiv:2210.03350*, 2022. **Abstract:** We investigate the ability of language models to perform compositional reasoning tasks where the overall solution depends on correctly composing the answers to sub-problems. We measure how often models can correctly answer all sub-problems but not generate the overall solution, a ratio we call the compositionality gap. We evaluate this ratio by asking multi-hop questions with answers that require composing multiple facts unlikely to have been observed together during pretraining. In the GPT-3 family of models, as model size increases we show that the single-hop question answering performance improves faster than the multi-hop performance does, therefore the compositionality gap does not decrease. This surprising result suggests that while more powerful models memorize and recall more factual knowledge, they show no corresponding improvement in their ability to perform this kind of compositional reasoning. We then demonstrate how elicitive prompting (such as chain of thought) narrows the compositionality gap by reasoning explicitly. We present a new method, self-ask, that further improves on chain of thought. In our method, the model explicitly asks itself (and answers) follow-up questions before answering the initial question. We finally show that self-ask’s structured prompting lets us easily plug in a search engine to answer the follow-up questions, which additionally improves accuracy. (@press2022measuring)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu Exploring the limits of transfer learning with a unified text-to-text transformer *The Journal of Machine Learning Research*, 21 (1): 5485–5551, 2020. **Abstract:** Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code. (@raffel2020exploring)

John L Russell Kepler’s laws of planetary motion: 1609–1666. *The British journal for the history of science*, 2 (1): 1–24, 1964. **Abstract:** Synopsis Historians of seventeenth-century science have frequently asserted that Kepler’s laws of planetary motion were largely ignored between the time of their first publication (1609, 1619) and the publication of Newton’s Principia (1687). In fact, however, they were more widely known and accepted than has been generally recognized. Kepler’s ideas were, indeed, rather slow in establishing themselves, and until about 1630 there are few references to them in the literature of the time. But from then onwards, interest in them increased fairly rapidly. In particular, the principle of elliptical orbits had been accepted by most of the leading astronomers in France before 1645 and in England by about 1655. It also received quite strong support in Holland. The second law had a more chequered history. It was enunciated in its exact form by a few writers and was used in practice by some others without being explicitly formulated, but the majority, especially after 1645, preferred one or another of several variant forms which were easier to use but only approximately correct. The third law attracted less interest than the others, chiefly perhaps because it had no satisfactory theoretical basis, but it was correctly stated by at least six writers during the period under review. Between about 1630 and 1650 Kepler’s Epitome Astronomiae Copernicanae (in which all three laws were clearly formulated) was probably the most widely read work on theoretical astronomy in northern and western Europe, while his Rudolphine Tables, which were based upon the first two laws, were regarded by the majority of astronomers as the most accurate planetary tables available. Kepler’s work certainly did not receive all the recognition it deserved, but the extent to which it was neglected has been much exaggerated. (@russell1964kepler)

Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou Recitation-augmented language models *arXiv preprint arXiv:2210.01296*, 2022. **Abstract:** We propose a new paradigm to help Large Language Models (LLMs) generate more accurate factual knowledge without retrieving from an external corpus, called RECITation-augmented gEneration (RECITE). Different from retrieval-augmented language models that retrieve relevant documents before generating the outputs, given an input, RECITE first recites one or several relevant passages from LLMs’ own memory via sampling, and then produces the final answers. We show that RECITE is a powerful paradigm for knowledge-intensive NLP tasks. Specifically, we show that by utilizing recitation as the intermediate step, a recite-and-answer scheme can achieve new state-of-the-art performance in various closed-book question answering (CBQA) tasks. In experiments, we verify the effectiveness of \\}method~on four pre-trained models (PaLM, UL2, OPT, and Codex) and three CBQA tasks (Natural Questions, TriviaQA, and HotpotQA). Our code is available at "https://github.com/Edward-Sun/RECITE". (@sun2022recitation)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom Llama 2: Open foundation and fine-tuned chat models 2023. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@touvron2023llama)

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal Musique: Multihop questions via single-hop question composition *Transactions of the Association for Computational Linguistics*, 10: 539–554, 2022. **Abstract:** Abstract Multihop reasoning remains an elusive goal as existing multihop benchmarks are known to be largely solvable via shortcuts. Can we create a question answering (QA) dataset that, by construction, requires proper multihop reasoning? To this end, we introduce a bottom–up approach that systematically selects composable pairs of single-hop questions that are connected, that is, where one reasoning step critically relies on information from another. This bottom–up methodology lets us explore a vast space of questions and add stringent filters as well as other mechanisms targeting connected reasoning. It provides fine-grained control over the construction process and the properties of the resulting k-hop questions. We use this methodology to create MuSiQue-Ans, a new multihop QA dataset with 25K 2–4 hop questions. Relative to existing datasets, MuSiQue-Ans is more difficult overall (3× increase in human–machine gap), and harder to cheat via disconnected reasoning (e.g., a single-hop model has a 30-point drop in F1). We further add unanswerable contrast questions to produce a more stringent dataset, MuSiQue-Full. We hope our datasets will help the NLP community develop models that perform genuine multihop reasoning.1 (@trivedi2022musique)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin Attention is all you need *Advances in neural information processing systems*, 30, 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le Finetuned language models are zero-shot learners *arXiv preprint arXiv:2109.01652*, 2021. **Abstract:** This paper explores a simple method for improving the zero-shot learning abilities of language models. We show that instruction tuning – finetuning language models on a collection of tasks described via instructions – substantially improves zero-shot performance on unseen tasks. We take a 137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call FLAN, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 20 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of finetuning datasets, model scale, and natural language instructions are key to the success of instruction tuning. (@wei2021finetuned)

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al Emergent abilities of large language models *arXiv preprint arXiv:2206.07682*, 2022. **Abstract:** Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models. (@wei2022emergent)

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al Chain-of-thought prompting elicits reasoning in large language models *Advances in Neural Information Processing Systems*, 35: 24824–24837, 2022. **Abstract:** We explore how generating a chain of thought – a series of intermediate reasoning steps – significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier. (@wei2022chain)

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen Large language models as optimizers *arXiv preprint arXiv:2309.03409*, 2023. **Abstract:** Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to our main application in prompt optimization, where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. Code at https://github.com/google-deepmind/opro. (@yang2023large)

Michael Zhang and Eunsol Choi Situatedqa: Incorporating extra-linguistic contexts into qa In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 7371–7387, 2021. **Abstract:** Answers to the same question may change depending on the extra-linguistic contexts (when and where the question was asked). To study this challenge, we introduce SituatedQA, an open-retrieval QA dataset where systems must produce the correct answer to a question given the temporal or geographical context. To construct SituatedQA, we first identify such questions in existing QA datasets. We find that a significant proportion of information seeking questions have context-dependent answers (e.g. roughly 16.5% of NQ-Open). For such context-dependent questions, we then crowdsource alternative contexts and their corresponding answers. Our study shows that existing models struggle with producing answers that are frequently updated or from uncommon locations. We further quantify how existing models, which are trained on data collected in the past, fail to generalize to answering questions asked in the present, even when provided with an updated evidence corpus (a roughly 15 point drop in accuracy). Our analysis suggests that open-retrieval QA benchmarks should incorporate extra-linguistic context to stay relevant globally and in the future. Our data, code, and datasheet are available at https://situatedqa.github.io/. (@zhang2021situatedqa)

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh Calibrate before use: Improving few-shot performance of language models In *International Conference on Machine Learning*, pp. 12697–12706. PMLR, 2021. **Abstract:** GPT-3 can perform numerous tasks when provided a natural language prompt that contains a few training examples. We show that this type of few-shot learning can be unstable: the choice of prompt format, training examples, and even the order of the training examples can cause accuracy to vary from near chance to near state-of-the-art. We demonstrate that this instability arises from the bias of language models towards predicting certain answers, e.g., those that are placed near the end of the prompt or are common in the pre-training data. To mitigate this, we first estimate the model’s bias towards each answer by asking for its prediction when given the training prompt and a content-free test input such as "N/A". We then fit calibration parameters that cause the prediction for this input to be uniform across answers. On a diverse set of tasks, this contextual calibration procedure substantially improves GPT-3 and GPT-2’s average accuracy (up to 30.0% absolute) and reduces variance across different choices of the prompt. (@zhao2021calibrate)

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al Least-to-most prompting enables complex reasoning in large language models *arXiv preprint arXiv:2205.10625*, 2022. **Abstract:** Chain-of-thought prompting has demonstrated remarkable performance on various natural language reasoning tasks. However, it tends to perform poorly on tasks which requires solving problems harder than the exemplars shown in the prompts. To overcome this challenge of easy-to-hard generalization, we propose a novel prompting strategy, least-to-most prompting. The key idea in this strategy is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. Solving each subproblem is facilitated by the answers to previously solved subproblems. Our experimental results on tasks related to symbolic manipulation, compositional generalization, and math reasoning reveal that least-to-most prompting is capable of generalizing to more difficult problems than those seen in the prompts. A notable finding is that when the GPT-3 code-davinci-002 model is used with least-to-most prompting, it can solve the compositional generalization benchmark SCAN in any split (including length split) with an accuracy of at least 99% using just 14 exemplars, compared to only 16% accuracy with chain-of-thought prompting. This is particularly noteworthy because neural-symbolic models in the literature that specialize in solving SCAN are trained on the entire training set containing over 15,000 examples. We have included prompts for all the tasks in the Appendix. (@zhou2022least)

</div>

# GSM8K Results, and Error Analysis

## GSM8K Results

We present in Table <a href="#table:gsm8k" data-reference-type="ref" data-reference="table:gsm8k">4</a> the results of <span class="smallcaps">Step-Back Prompting</span> on GSM8K along with other strong baselines from PaLM-2L runs. We observe that <span class="smallcaps">Step-Back Prompting</span> achieved competitive performance together with zero-shot CoT and 1-shot standard prompting. We hypothesize that the simplicity of principles (e.g. addition, subtraction, etc.) in GSM8K makes it not absolutely necessary to retrieve the principles first before reasoning. Nonetheless, we still find that <span class="smallcaps">Step-Back Prompting</span> is the most competitive among all the prompting methods we tested, including the “Take a Deep Breath” prompting optimized for GSM8K in and Decomposed Prompting in  .

<div class="center">

<div id="table:gsm8k">

| Method                     |      GSM8K       |
|:---------------------------|:----------------:|
| PaLM-2L                    |   75.8% (0.2%)   |
| PaLM-2L 1-shot             | **84.5%** (0.4%) |
| PaLM-2L + CoT              | **84.4%** (0.2%) |
| PaLM-2L + CoT 1-shot       |    81% (0.2%)    |
| PaLM-2L + TDB              |   82.2% (0.2%)   |
| PaLM-2L + DP               |  82.2% (0.08%)   |
| PaLM-2L + Step-Back (ours) | **84.3%** (0.2%) |

Step-Back Prompting on GSM8K. CoT: zero-shot Chain of Thought prompting , TDB: Take a Deep Breath prompting , DP: Decomposed Prompting . The Table reports the average accuracy over 5 evaluation runs, with standard deviations in the parentheses.

</div>

</div>

## TimeQA Error Analysis

We conduct error analysis to understand where <span class="smallcaps">Step-Back Prompting</span> fixes the errors the baseline models make. Figure <a href="#fig:figure_timeqa_ea" data-reference-type="ref" data-reference="fig:figure_timeqa_ea">6</a> shows that compared to the predictions of baseline PaLM-2L, <span class="smallcaps">Step-Back Prompting</span> can fix $`39.9\%`$ of the predictions where the baseline prediction is wrong, while causing $`5.6\%`$ errors.Furthermore, Step-Back + RAG fixes $`21.6\%`$ errors coming from RAG. The $`\%`$ of errors introduced by <span class="smallcaps">Step-Back Prompting</span> to RAG is still relatively low ($`6.3\%`$). Together, this shows that the <span class="smallcaps">Step-Back Prompting</span> is helpful most of the time, signifying the need and effectiveness of doing Abstraction before directly addressing the original question.

<figure id="fig:figure_timeqa_ea">
<img src="./figures/Figure2_TimeQA_EA.png"" />
<figcaption>Error Analysis of Step-Back Prompting on TimeQA. <em>Left</em>: Step-Back + RAG vs Baseline predictions. <em>Right</em>: Step-Back RAG vs RAG predictions. Step-Back + RAG can fix <span class="math inline">39.9%</span> of the predictions where the baseline prediction is wrong while causing <span class="math inline">5.6%</span> errors. Furthermore, Step-Back + RAG fixes <span class="math inline">21.6%</span> errors coming from RAG. The <span class="math inline">%</span> of errors introduced by <span class="smallcaps">Step-Back Prompting</span> to RAG is still relatively low (<span class="math inline">6.3%</span>).</figcaption>
</figure>

## StrategyQA Error Analysis

Figure <a href="#fig:figure4_sqa_ea" data-reference-type="ref" data-reference="fig:figure4_sqa_ea">7</a> shows the error analysis of StrategyQA on the predictions of Step-Back + RAG against the baseline model and the raw retrieval augmentation variant of PaLM-2L. Compared to the baseline, Step-Back + RAG can turn $`15.4\%`$ wrong predictions into correct predictions, while leading to $`6.1\%`$ errors the other way around. Furthermore, Step-Back + RAG fixes $`12.7\%`$ errors coming from RAG. The errors introduced to RAG by Step-Back are just $`4.4\%`$.

<figure id="fig:figure4_sqa_ea">
<img src="./figures/Figure4_SQA_EA.png"" />
<figcaption>Error Analysis of Step-Back Prompting on StrategyQA. <em>Left</em>: Step-Back + RAG vs Baseline predictions. <em>Right</em>: Step-Back + RAG vs RAG predictions. Step-Back + RAG is able to turn <span class="math inline">15.4%</span> wrong predictions into correct predictions, while leading to <span class="math inline">6.1%</span> errors the other way around. Furthermore, Step-Back + RAG fixes <span class="math inline">12.7%</span> errors coming from RAG. The errors introduced to RAG by Step-Back are just <span class="math inline">4.4%</span>.</figcaption>
</figure>

# Dataset Details

Table <a href="#app:eval_stats" data-reference-type="ref" data-reference="app:eval_stats">5</a> shows the split and number of examples used for evaluations in TimeQA, StrategyQA, MMLU, and GSM8K.

<div class="center">

<div id="app:eval_stats">

| Domain              |          Dataset           | Split | Number of Examples |
|:--------------------|:--------------------------:|:-----:|:------------------:|
| STEM                |  MMLU high-school Physics  | Test  |        151         |
|                     | MMLU high-school Chemistry | Test  |        203         |
|                     |           GSM8K            | Test  |        1319        |
| Knowledge QA        |           TimeQA           | Test  |        5226        |
|                     |        TimeQA Easy         | Test  |        2613        |
|                     |        TimeQA Hard         | Test  |        2613        |
|                     |         SituatedQA         | Test  |        2901        |
| Multi-hop Reasoning |          MuSiQue           |  Dev  |        2417        |
|                     |         StrategyQA         |  Dev  |        229         |

Stats of the evaluation datasets used in this paper.

</div>

</div>

# Evaluation Details

## Few-shot Examples for Evaluation with PaLM-2L

Given the model free-form outputs and the target label, we use one positive and one negative output as few-shot examples to demonstrate to the scoring model how to score the output. Table  <a href="#tab:fewevalpalm" data-reference-type="ref" data-reference="tab:fewevalpalm">6</a> illustrates the prompt we used for the scoring model. We parse out the “Yes" or “No" answer from the scoring model output as a TRUE or FALSE score of the model output.

<div id="tab:fewevalpalm">

<table>
<caption>Illustration of few shot evaluation with the PaLM-2L model. </caption>
<thead>
<tr>
<th style="text-align: left;">Are the following two answers to the given question equivalent? Do not consider whether the answers are right or wrong, but only whether they are equivalent. Directly state "Yes" or "No".<br />
<strong>Question</strong>: Which title was conferred to Anna Muzychuk in 2007?<br />
<strong>Answer 1</strong>: Anna Muzychuk was conferred the title of International Master (IM) in 2007. She earned the title by scoring three norms in rapid chess tournaments.<br />
<strong>Answer 2</strong>: International Master<br />
<strong>Answer 1 (short)</strong>: International Master<br />
<strong>Answer 2 (short)</strong>: International Master<br />
<strong>Are the two answers equivalent?</strong> Yes<br />
<strong>Question</strong>: What state is Seattle located in?<br />
<strong>Answer 1</strong>: Seattle is in Washington State.<br />
<strong>Answer 2</strong>: The answer is George Washington.<br />
<strong>Answer 1 (short)</strong>: Washington State<br />
<strong>Answer 2 (short)</strong>: George Washington<br />
<strong>Are the two answers equivalent?</strong> No<br />
<strong>Question</strong>: <span class="math inline">&lt;</span>Question<span class="math inline">&gt;</span><br />
<strong>Answer 1</strong>: <span class="math inline">&lt;</span>Model Output<span class="math inline">&gt;</span><br />
<strong>Answer 2</strong>: <span class="math inline">&lt;</span>Target Label<span class="math inline">&gt;</span></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

</div>

## Hyper-parameters for Evaluation with PaLM-2L

We use PaLM-2L as the scoring model for evaluation. We experiment with different sampling temperatures, and find that $`T=1`$ gives us a highly-accurate evaluation. For example, we sampled $`100`$ test examples and the model predictions, and manually rated the correctness of the model scoring. We found that out of 4 trials, the model scoring agrees with human ratings $`97\%`$, $`98\%`$, $`99\%`$ and $`99\%`$ of the time.

# Prompts and Few shot Examples

## STEM

For MMLU high-school Physics and Chemistry, we first prompt the model to generate the first principles behind the question. Using the generated first principles, we further prompt the model to generate the final answer through few-shot demonstrations The prompt generating first principles is shown in Table <a href="#tab:mmlu_principle_prompt" data-reference-type="ref" data-reference="tab:mmlu_principle_prompt">7</a> for MMLU high-school Physics and Chemistry.

<div id="tab:mmlu_principle_prompt">

<table>
<caption>Prompt of extracting the underlying principles involved in MMLU physics and chemistry questions.</caption>
<thead>
<tr>
<th style="text-align: left;">MMLU Physics/Chemistry First-Principle Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">You are an expert at Physics/Chemistry. You are given a Physics/Chemistry problem. Your task is to extract the Physics/Chemistry concepts and principles involved in solving the problem. Here are a few examples:<br />
<br />
Question: <span class="math inline">&lt;</span>Question Example1<span class="math inline">&gt;</span><br />
Principles Involved: <span class="math inline">&lt;</span>Principles Example1<span class="math inline">&gt;</span><br />
...<br />
Question: <span class="math inline">&lt;</span>Question Example5<span class="math inline">&gt;</span><br />
Principles Involved: <span class="math inline">&lt;</span>Principles Example5<span class="math inline">&gt;</span><br />
Question: <span class="math inline">&lt;</span>Question<span class="math inline">&gt;</span><br />
Principles Involved:</td>
</tr>
</tbody>
</table>

</div>

After extracting the first principles of solving a particular question, we formulate the prompt in Table <a href="#tab:mmlu_answer_prompt" data-reference-type="ref" data-reference="tab:mmlu_answer_prompt">8</a> to query the model for the final answer.

<div id="tab:mmlu_answer_prompt">

<table>
<caption>Prompt of querying the model for final answer with first principles behind the question in MMLU high-school Physics and Chemistry.</caption>
<thead>
<tr>
<th style="text-align: left;">MMLU Physics/Chemistry Final Answer Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">You are an expert at Physics/Chemistry. You are given a Physics/Chemistry problem and a set of principles involved in solving the problem. Solve the problem step by step by following the principles. Here are a few examples:<br />
<br />
Question: <span class="math inline">&lt;</span>Question Example1<span class="math inline">&gt;</span><br />
Principles: <span class="math inline">&lt;</span>Principles Example1<span class="math inline">&gt;</span><br />
Answer: <span class="math inline">&lt;</span>Answer Example1<span class="math inline">&gt;</span><br />
...<br />
Question: <span class="math inline">&lt;</span>Question Example5<span class="math inline">&gt;</span><br />
Principles: <span class="math inline">&lt;</span>Principles Example5<span class="math inline">&gt;</span><br />
Answer: <span class="math inline">&lt;</span>Answer Example5<span class="math inline">&gt;</span><br />
Question: <span class="math inline">&lt;</span>Question<span class="math inline">&gt;</span><br />
Principles: <span class="math inline">&lt;</span>Principles<span class="math inline">&gt;</span><br />
Answer:</td>
</tr>
</tbody>
</table>

</div>

Tables <a href="#tab:mmlu_stepback_exemplars_1" data-reference-type="ref" data-reference="tab:mmlu_stepback_exemplars_1">9</a>-<a href="#tab:mmlu_stepback_exemplars_chem_1" data-reference-type="ref" data-reference="tab:mmlu_stepback_exemplars_chem_1">10</a> show one demonstration exemplar of Question-Principles-Answer triplets for MMLU high-school Physics and Chemistry, respectively. For GSM8K, given the simplicity of the principles, we directly combine the principles and the solution in the demonstration exemplar. Table <a href="#tab:gsm8k_stepback_prompt" data-reference-type="ref" data-reference="tab:gsm8k_stepback_prompt">11</a> shows the exemplar we used in the paper.

<div id="tab:mmlu_stepback_exemplars_1">

<table>
<caption>Few-shot demonstration exemplar for MMLU high-school Physics. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">A spherical conductor carries a net charge. How is this charge distributed on the sphere?<br />
0) The charge is evenly distributed on the surface.<br />
1) The charge resides on the surface only; the distribution of charge on the surface depends on what other charged objects are near the sphere.<br />
2) The charge moves continually within the sphere.<br />
3) The charge is distributed uniformly throughout the sphere.</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Coulomb’s Law: the force between two charged particles is proportional to the product of their charges and inversely proportional to the square of the distance between them, F = k * q1 * q2 / r, where F is the electric force, k is a constant, q1 and q2 are the charges the particles carry, and r is the distance between them.</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Coulomb’s Law, we can solve the problem as following:<br />
Step 1: Apply Coulomb’s Law to find out how charges are distributed on the surface.<br />
In the case of a spherical conductor, the charges on the surface will repel each other. The further apart the charges are, the less force they will exert on each other. Therefore, the charges will distribute themselves evenly on the surface of the sphere, as this is the configuration that minimizes the repulsive force between them.<br />
</p>
<p>Step 2: Apply Coulomb’s Law to find out what happens if there are other charges present.<br />
The distribution of charge on the surface may also be affected by the presence of other charged objects near the sphere. For example, if a negatively charged object is brought near a positively charged sphere, the negative charges on the sphere will be repelled and will move to the opposite side of the sphere. This will result in a non-uniform distribution of charge on the surface of the sphere.<br />
</p>
<p>Therefore, the correct answer is 1) The charge resides on the surface only; the distribution of charge on the surface depends on what other charged objects are near the sphere.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmlu_stepback_exemplars_chem_1">

<table>
<caption>Few-shot demonstration exemplar for MMLU high-school Chemistry. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">A sample of an unknown chloride compound was dissolved in water, and then titrated with excess Pb(NO3)2 to create a precipitate. After drying, it is determined there are 0.0050 mol of precipitate present. What mass of chloride is present in the original sample?<br />
0) 0.177 g<br />
1) 0.355 g<br />
2) 0.522 g<br />
3) 0.710 g</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Precipitation reactions: Precipitation reactions occur when two soluble salts are mixed and form an insoluble product, called a precipitate. The precipitate can be separated from the solution by filtration or centrifugation.<br />
Molar mass: The molar mass of a substance is the mass of one mole of that substance. The molar mass is expressed in grams per mole (g/mol).<br />
Limiting reactant: The limiting reactant is the reactant that is completely consumed in a chemical reaction. The amount of product formed is determined by the amount of limiting reactant.</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Assuming the unknown chloride compound is MCl, where M represents the metal cation, the balanced chemical equation for the precipitation reaction is:<br />
<span class="math inline"><em>P</em><em>b</em>(<em>N</em><em>O</em><sub>3</sub>)<sub>2</sub>(<em>a</em><em>q</em>)</span> + <span class="math inline">2<em>M</em><em>C</em><em>l</em>(<em>a</em><em>q</em>) → <em>P</em><em>b</em><em>C</em><em>l</em><sub>2</sub>(<em>s</em>)</span> + <span class="math inline">2<em>M</em><em>N</em><em>O</em><sub>3</sub>(<em>a</em><em>q</em>)</span><br />
</p>
<p>Since <span class="math inline"><em>P</em><em>b</em>(<em>N</em><em>O</em><sub>3</sub>)<sub>2</sub></span> is in excess, MCl is the limiting reactant. The stoichiometry of the reaction indicates that 2 moles of MCl produce 1 mole of <span class="math inline"><em>P</em><em>b</em><em>C</em><em>l</em><sub>2</sub></span> precipitate. Therefore, 0.0050 mol of <span class="math inline"><em>P</em><em>b</em><em>C</em><em>l</em><sub>2</sub></span> corresponds to 0.010 mol of MCl.<br />
</p>
<p>The mass of chloride in the original sample can be calculated using the molar mass of chloride (35.45 g/mol):<br />
0.010 mol Cl × 35.45 g/mol = 0.355 g Cl<br />
</p>
<p>The correct answer is 1) 0.355 g.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:gsm8k_stepback_prompt">

<table>
<caption>Step-Back few-shot demonstration exemplar for GSM8K. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Let us find and apply the math principles to solve the problem step by step:<br />
</p>
<p>Step 1. Addition: Calculate the number of truck stamps.<br />
Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps: there are 11 + 9 = 20 truck stamps.<br />
</p>
<p>Step 2. Subtraction: Calculate the number of rose stamps.<br />
Bella bought 13 fewer rose stamps than truck stamps: there are 20 - 13 = 7 rose stamps.<br />
</p>
<p>Step 3. Addition: Calculate the total number of stamps in all three colors.<br />
Bella bought 11 snowflake stamps, 20 truck stamps, 7 rose stamps: there are 11 + 20 + 7 = 38 stamps in total.<br />
</p>
<p>Conclusion: Bella bought 38 stamps in all.</p></td>
</tr>
</tbody>
</table>

</div>

## Knowledge QA

We use the following prompting in Table <a href="#tab:timeqa_stepback_prompt" data-reference-type="ref" data-reference="tab:timeqa_stepback_prompt">12</a> to demonstrate to the LLM on asking a step-back question for TimeQA and SituatedQA including up to 5 exemplar demonstrations of pairs of Original Question and Step-back Question.

<div id="tab:timeqa_stepback_prompt">

<table>
<caption>Prompt of asking step-back question in Knowledge QA tasks.</caption>
<thead>
<tr>
<th style="text-align: left;">Knowledge QA Step-Back Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:<br />
<br />
Original Question: <span class="math inline">&lt;</span>Original Question Example1<span class="math inline">&gt;</span><br />
Stepback Question: <span class="math inline">&lt;</span>Stepback Question Example1<span class="math inline">&gt;</span><br />
...<br />
Original Question: <span class="math inline">&lt;</span>Original Question Example5<span class="math inline">&gt;</span><br />
Stepback Question: <span class="math inline">&lt;</span>Stepback Question Example5<span class="math inline">&gt;</span><br />
Original Question: <span class="math inline">&lt;</span>Original Question<span class="math inline">&gt;</span><br />
Stepback Question:</td>
</tr>
</tbody>
</table>

</div>

Table <a href="#tab:timeqa_stepback_exemplars" data-reference-type="ref" data-reference="tab:timeqa_stepback_exemplars">13</a> shows 5 exemplars from the Train split of TimeQA and SituatedQA as demonstrations of asking step-back questions.

<div id="tab:timeqa_stepback_exemplars">

| dataset | Original Question | Step-back Question |
|:---|:---|:---|
| TimeQA | Which position did Knox Cunningham hold from May 1955 to Apr 1956? | Which positions have Knox Cunningham held in his career? |
| TimeQA | Who was the spouse of Anna Karina from 1968 to 1974? | Who were the spouses of Anna Karina? |
| TimeQA | Which team did Thierry Audel play for from 2007 to 2008? | Which teams did Thierry Audel play for in his career? |
| TimeQA | What was the operator of GCR Class 11E from 1913 to Dec 1922? | What were the operators of GCR Class 11E in history? |
| TimeQA | Which country did Sokolovsko belong to from 1392 to 1525? | Which countries did Sokolovsko belong to in history? |
| SituatedQA | when was the last time a team from canada won the stanley cup as of 2002 | which years did a team from canada won the stanley cup as of 2002 |
| SituatedQA | when did england last get to the semi final in a world cup as of 2019 | which years did england get to the semi final in a world cup as of 2019? |
| SituatedQA | what is the biggest hotel in las vegas nv as of November 28, 1993 | what is the size of the hotels in las vegas nv as of November 28, 1993 |
| SituatedQA | who has scored most runs in t20 matches as of 2017 | What are the runs of players in t20 matches as of 2017 |
| SituatedQA | who is the highest paid player in the nba this season as of 2017 | what is the salary of the high paid players in the nba this season as of 2017 |

Few-shot demonstration exemplars for asking step-back questions in TimeQA and SituatedQA.

</div>

The step-back question is extracted from the model output using the prompt. Using the step-back question, we do retrieval augmentation. Using both the retrieval augmentations from the original question and the step-back question, we formulate the final prompt to query the model for the final answer, as shown in Table <a href="#tab:timeqa_final_prompt" data-reference-type="ref" data-reference="tab:timeqa_final_prompt">14</a>.

<div id="tab:timeqa_final_prompt">

<table>
<caption>Prompt of querying the model for final answer with additional contexts from original and step-back retrieval augmentations in TimeQA and SituatedQA</caption>
<thead>
<tr>
<th style="text-align: left;">Knowledge QA Final-Answer Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.<br />
<br />
<span class="math inline">&lt;</span>Passage from original retrieval augmentation<span class="math inline">&gt;</span><br />
<span class="math inline">&lt;</span>Passage from step-back retrieval augmentation<span class="math inline">&gt;</span><br />
<br />
Original Question: <span class="math inline">&lt;</span>Original Question<span class="math inline">&gt;</span><br />
Answer:</td>
</tr>
</tbody>
</table>

</div>

## Multi-Hop Reasoning

For Multi-Hop Reasoning, we use the same prompting template as in Knowledge QA to ask the step-back question, and query for the final answer given the retrieval augmentations. Table <a href="#tab:strategyqa_stepback_exemplars" data-reference-type="ref" data-reference="tab:strategyqa_stepback_exemplars">15</a> shows 5 demonstration exemplars for asking step-back questions from the Train split of MuSiQue and StrategyQA.

<div id="tab:strategyqa_stepback_exemplars">

| dataset | Original Question | Step-back Question |
|:---|:---|:---|
| MuSiQue | at year saw the creation of the region where the county of Hertfordshire is located? | which region is the county of Hertfordshire located? |
| MuSiQue | Jan Šindel’s was born in what country? | what is Jan Šindel’s personal history? |
| MuSiQue | When was the abolishment of the studio that distributed The Game? | which studio distributed The Game? |
| MuSiQue | What city is the person who broadened the doctrine of philosophy of language from? | who broadened the doctrine of philosophy of language |
| MuSiQue | When was the baseball team winning the world series in 2015 baseball created? | which baseball team won the world series in 2015 baseball? |
| StrategyQA | Could the members of The Police perform lawful arrests? | what can the members of The Police do? |
| StrategyQA | Would a Monoamine Oxidase candy bar cheer up a depressed friend? | What are the effects of Monoamine Oxidase? |
| StrategyQA | Would a dog respond to bell before Grey seal? | Would a dog respond to bell before Grey seal? |
| StrategyQA | Is shrimp scampi definitely free of plastic? | what is shrimp scampi made of? |
| StrategyQA | Do the anchors on Rede Globo speak Chinese? | What languages do the anchors on Rede Globo speak? |

Few-shot demonstration exemplars for asking step-back questions in MuSiQue and StrategyQA.

</div>

## Baseline Prompts

For standard zero-shot and few-shot prompting of the baseline model, we formulate the prompt using the template in Table <a href="#tab:baseline_answer_prompt" data-reference-type="ref" data-reference="tab:baseline_answer_prompt">16</a> with up to 1 exemplars.

<div id="tab:baseline_answer_prompt">

<table>
<caption>Prompt of querying the baseline model for final answer with few-shot demonstration exemplars.</caption>
<thead>
<tr>
<th style="text-align: left;">Baseline few-shot Prompt</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">You are an expert of world knowledge and physics. Your task is to solve the following question. Here are a few examples:<br />
<br />
Question: <span class="math inline">&lt;</span>Question Example<span class="math inline">&gt;</span><br />
Answer: <span class="math inline">&lt;</span>Answer Example<span class="math inline">&gt;</span><br />
Question: <span class="math inline">&lt;</span>Question<span class="math inline">&gt;</span><br />
Answer:</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:baseline_exemplars">

<table>
<caption>Demonstration exemplar for standard prompting. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU Physics</td>
<td style="text-align: left;">A point charge, Q1 = +1 mC, is fixed at the origin. How much work is required to move another charge, Q2 = +8 <span class="math inline"><em>μ</em></span>C, from the point (0, 4 meters) to the point (3 meters, 0)?<br />
0) 3.5 J 1) 6.0 J 2) 22.5 J 3) 40 J</td>
<td style="text-align: left;"><p>The work required to move a charge in an electric field is equal to the change in potential energy of the charge. The potential energy of a charge in an electric field is given by: U = k * Q1 * Q2 / r where: k is the Coulomb constant, Q1 and Q2 are the charges, r is the distance between the charges. In this case, the initial potential energy of Q2 is: U1 = k * Q1 * Q2 / 4 meters<br />
</p>
<p>The final potential energy of Q2 is: U2 = k * Q1 * Q2 / 3 meters<br />
</p>
<p>The change in potential energy is: <span class="math inline"><em>Δ</em></span>U = U2 - U1 = k * Q1 * Q2 * (1/3 meters - 1/4 meters)<br />
</p>
<p>Plugging in the values, we get: <span class="math inline"><em>Δ</em></span>U = <span class="math inline">(9 × 10<sup>9</sup><em>N</em> * <em>m</em><sup>2</sup>/<em>C</em><sup>2</sup>) × (1 × 10<sup>−3</sup><em>C</em>) * (8 × 10<sup>−6</sup><em>C</em>) * (1/3<em>m</em> − 1/4<em>m</em>) = 6<em>J</em></span>. The answer is 1) 6.0 J.</p></td>
</tr>
<tr>
<td style="text-align: left;">MMLU Chemistry</td>
<td style="text-align: left;">A sample of an unknown chloride compound was dissolved in water, and then titrated with excess Pb(NO3)2 to create a precipitate. After drying, it is determined there are 0.0050 mol of precipitate present. What mass of chloride is present in the original sample?<br />
0) 0.177 g 1) 0.355 g 2) 0.522 g 3) 0.710 g</td>
<td style="text-align: left;"><p>0.0050 moles of lead chloride precipitate contain 0.010 moles of chloride ions due to the 2:1 mole ratio between chloride ions and lead chloride.<br />
</p>
<p>The mass of 0.010 moles of chloride ions is 0.355 g.<br />
</p>
<p>Therefore, the mass of chloride present in the original sample is 0.355 g.</p></td>
</tr>
<tr>
<td style="text-align: left;">GSM8K</td>
<td style="text-align: left;">Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?</td>
<td style="text-align: left;">The number of truck stamps is 11 + 9 = 20. The number of rose stamps is 20 − 13 = 7. Bella bought 11 + 20 + 7 = 38 stamps in all.</td>
</tr>
<tr>
<td style="text-align: left;">TimeQA</td>
<td style="text-align: left;">Who was the spouse of Anna Karina from 1968 to 1974?</td>
<td style="text-align: left;">Anna Karina’s spouse from 1968 to 1974 was French actor Pierre Fabre.</td>
</tr>
<tr>
<td style="text-align: left;">SituatedQA</td>
<td style="text-align: left;">what is the biggest hotel in las vegas nv as of November 28, 1993</td>
<td style="text-align: left;">The biggest hotel in Las Vegas, Nevada as of November 28, 1993 was the Excalibur Hotel &amp; Casino.</td>
</tr>
<tr>
<td style="text-align: left;">MuSiQue</td>
<td style="text-align: left;">What year saw the creation of the region where the county of Hertfordshire is located?</td>
<td style="text-align: left;">1994</td>
</tr>
<tr>
<td style="text-align: left;">StrategyQA</td>
<td style="text-align: left;">Would a Monoamine Oxidase candy bar cheer up a depressed friend?</td>
<td style="text-align: left;">A Monoamine Oxidase (MAO) candy bar would not cheer up a depressed friend. MAO is an enzyme that breaks down neurotransmitters such as serotonin, dopamine, and norepinephrine. These neurotransmitters are associated with mood, so inhibiting MAO can lead to an increase in their levels.</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:cot_exemplars_physics">

<table>
<caption>Demonstration exemplar for CoT prompting for MMLU Physics. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU Physics</td>
<td style="text-align: left;">A point charge, Q1 = +1 mC, is fixed at the origin. How much work is required to move another charge, Q2 = +8 <span class="math inline"><em>μ</em></span>C, from the point (0, 4 meters) to the point (3 meters, 0)?<br />
0) 3.5 J<br />
1) 6.0 J<br />
2) 22.5 J<br />
3) 40 J</td>
<td style="text-align: left;"><p>Sure, let’s think step by step.<br />
</p>
<p>1. Calculate the electric potential energy of Q2 at the initial position. The electric potential energy (U) of a point charge Q2 at a distance r from another point charge Q1 is given by the following equation: U = k * Q1 * Q2 / r, where k is Coulomb’s constant.<br />
</p>
<p>Therefore, the electric potential energy of Q2 at the initial position is:<br />
<span class="math inline"><em>U</em><sub>initial</sub></span> = <span class="math inline">(9 × 10<sup>9</sup><em>N</em><em>m</em><sup>2</sup>/<em>C</em><sup>2</sup>) * (1 × 10<sup>−3</sup><em>C</em>) * (8 × 10<sup>−6</sup><em>C</em>)/4<em>m</em></span> = 18 J<br />
</p>
<p>2. Calculate the electric potential energy of Q2 at the final position. The electric potential energy of Q2 at the final position is:<br />
<span class="math inline"><em>U</em><sub>final</sub></span> = <span class="math inline">(9 × 10<sup>9</sup><em>N</em><em>m</em><sup>2</sup>/<em>C</em><sup>2</sup>) * (1 × 10<sup>−3</sup><em>C</em>) * (8 × 10<sup>−6</sup><em>C</em>)/3<em>m</em></span> = 24 J<br />
</p>
<p>3. Calculate the work required to move Q2 from the initial position to the final position. The work (W) required to move a charge from one position to another is equal to the change in electric potential energy (<span class="math inline"><em>Δ</em></span>U) between the two positions.<br />
W = <span class="math inline"><em>Δ</em></span>U = <span class="math inline"><em>U</em><sub>final</sub></span> - <span class="math inline"><em>U</em><sub>initial</sub></span> = 24 J - 18 J = 6 J<br />
</p>
<p>Therefore, the answer is 1) 6.0 J.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:cot_exemplars_chem">

<table>
<caption>Demonstration exemplar for CoT prompting for MMLU Chemistry. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU Chemistry</td>
<td style="text-align: left;">A sample of an unknown chloride compound was dissolved in water, and then titrated with excess Pb(NO3)2 to create a precipitate. After drying, it is determined there are 0.0050 mol of precipitate present. What mass of chloride is present in the original sample?<br />
0) 0.177 g<br />
1) 0.355 g<br />
2) 0.522 g<br />
3) 0.710 g</td>
<td style="text-align: left;"><p>Here’s the step-by-step solution:<br />
Identify the precipitate: The precipitate formed when a chloride compound reacts with lead nitrate (Pb(NO3)2) is lead chloride (PbCl2).<br />
</p>
<p>Determine the mole ratio: The balanced chemical equation for the reaction between a chloride compound (represented as MCl, where M is the unknown metal) and lead nitrate is: <span class="math inline">2<em>M</em><em>C</em><em>l</em></span> + <span class="math inline"><em>P</em><em>b</em>(<em>N</em><em>O</em><sub>3</sub>)<sub>2</sub> → 2<em>M</em><em>N</em><em>O</em><sub>3</sub></span> + <span class="math inline"><em>P</em><em>b</em><em>C</em><em>l</em><sub>2</sub></span><br />
From the balanced equation, you can see that 2 moles of chloride ions (Cl-) react with 1 mole of lead nitrate to produce 1 mole of lead chloride precipitate. Additionally, 1 mole of lead chloride contains 2 moles of chloride ions.<br />
</p>
<p>Calculate the moles of chloride ions: Since 0.0050 moles of lead chloride precipitate were formed, there must have been 2 × 0.0050 = 0.010 moles of chloride ions in the original sample.<br />
</p>
<p>Calculate the mass of chloride ions: The molar mass of chlorine (Cl) is 35.45 g/mol. Therefore, the mass of 0.010 moles of chloride ions is:<br />
mass = moles × molar mass = 0.010 mol × 35.45 g/mol = 0.355 g<br />
</p>
<p>Therefore, the mass of chloride present in the original sample is 0.355 g. The correct answer is option 1.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:cot_exemplars_gsm8k">

<table>
<caption>Demonstration exemplar for CoT prompting for GSM8K. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">GSM8K</td>
<td style="text-align: left;">Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?</td>
<td style="text-align: left;"><p>Here are the steps to solve the problem:<br />
</p>
<p>Step 1: Find the number of truck stamps. Adding 9 to the number of snowflake stamps gives the number of truck stamps: 11 + 9 = 20<br />
</p>
<p>Step 2: Find the number of rose stamps. Subtracting 13 from the number of truck stamps gives the number of rose stamps: 20 - 13 = 7<br />
</p>
<p>Step 3: Add the number of snowflake, truck, and rose stamps to find the total number of stamps Bella bought: 11 + 20 + 7 = 38<br />
</p>
<p>Solution: Bella bought 38 stamps in all.</p></td>
</tr>
</tbody>
</table>

</div>

## Chain of Thought (CoT)

For zero-shot CoT prompting, we simply append *Let’s think step by step.* to the question to query the model.

For few-shot CoT prompting, we use the same template as the Baseline prompting in Sec. <a href="#app:baseline_prompt" data-reference-type="ref" data-reference="app:baseline_prompt">13.4</a> by replacing the few-shot examples using CoT responses, as shown in Tables <a href="#tab:cot_exemplars_physics" data-reference-type="ref" data-reference="tab:cot_exemplars_physics">18</a>, <a href="#tab:cot_exemplars_chem" data-reference-type="ref" data-reference="tab:cot_exemplars_chem">19</a>, <a href="#tab:cot_exemplars_gsm8k" data-reference-type="ref" data-reference="tab:cot_exemplars_gsm8k">20</a>, <a href="#tab:cot_exemplars_knowledge" data-reference-type="ref" data-reference="tab:cot_exemplars_knowledge">21</a>, and <a href="#tab:cot_exemplars_multihop" data-reference-type="ref" data-reference="tab:cot_exemplars_multihop">22</a>.

<div id="tab:cot_exemplars_knowledge">

<table>
<caption>Demonstration exemplar for CoT prompting for TimeQA and SituatedQA. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">TimeQA</td>
<td style="text-align: left;">Who was the spouse of Anna Karina from 1968 to 1974?</td>
<td style="text-align: left;"><p>Step 1: Identify Anna Karina’s spouses:Jean-Luc Godard (1961–1965).<br />
Pierre Fabre (1968–1974)<br />
Daniel Duval (1978–1981)<br />
Dennis Berry (1982–1994)<br />
</p>
<p>Step 2: Determine which spouse was married to Anna Karina from 1968 to 1974.<br />
Jean-Luc Godard was married to Anna Karina from 1961 to 1965, so he is not the correct answer.<br />
Pierre Fabre was married to Anna Karina from 1968 to 1974, so he is the correct answer.<br />
Daniel Duval was married to Anna Karina from 1978 to 1981, so he is not the correct answer.<br />
Dennis Berry was married to Anna Karina from 1982 to 1994, so he is not the correct answer.<br />
</p>
<p>Therefore, Anna Karina’s spouse from 1968 to 1974 was Pierre Fabre.</p></td>
</tr>
<tr>
<td style="text-align: left;">SituatedQA</td>
<td style="text-align: left;">what is the biggest hotel in las vegas nv as of November 28, 1993</td>
<td style="text-align: left;"><p>Sure, let’s think step by step:</p>
<p>What were the largest hotels in Las Vegas in 1993? According to a 1993 article in the Las Vegas Review-Journal, the largest hotels in Las Vegas at the time were:</p>
<p>MGM Grand (5,005 rooms) Excalibur (4,008 rooms) The Mirage (3,049 rooms) Flamingo Hilton (3,565 rooms) Caesars Palace (2,577 rooms) Which of these hotels was the largest on November 28, 1993? The MGM Grand opened in December 1993, so it would not have been the largest hotel on November 28, 1993. The Excalibur opened in June 1990, so it would have been the largest hotel on November 28, 1993.</p>
<p>Therefore, the largest hotel in Las Vegas on November 28, 1993 was the Excalibur.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:cot_exemplars_multihop">

<table>
<caption>Demonstration exemplar for CoT prompting for MuSiQue and StrategyQA. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Answer</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MuSiQue</td>
<td style="text-align: left;">What year saw the creation of the region where the county of Hertfordshire is located?</td>
<td style="text-align: left;">The county of Hertfordshire is in the ’East of England’ region. The ’East of England’ region was created in 1994. So, the answer is 1994.</td>
</tr>
<tr>
<td style="text-align: left;">StrategyQA</td>
<td style="text-align: left;">Would a Monoamine Oxidase candy bar cheer up a depressed friend?</td>
<td style="text-align: left;"><p>No, a Monoamine Oxidase (MAO) candy bar would not cheer up a depressed friend. In fact, it could be dangerous for them to consume. Here’s a step-by-step explanation:</p>
<p>1. MAO is an enzyme that breaks down neurotransmitters like serotonin, dopamine, and norepinephrine. These neurotransmitters are important for regulating mood, and low levels of them are associated with depression.</p>
<p>2. MAO inhibitors (MAOIs) are a type of medication that blocks the action of MAO, which can increase levels of these neurotransmitters in the brain. MAOIs are sometimes used to treat depression, but they can have serious side effects, including interactions with certain foods and medications.</p>
<p>3. A MAO candy bar would not contain enough MAOI to have a therapeutic effect on depression. In fact, it is unlikely that a candy bar could contain any MAOI at all, as these medications are typically taken in pill form.</p>
<p>In conclusion, a MAO candy bar would not cheer up a depressed friend and could even be dangerous for them to consume.</p></td>
</tr>
</tbody>
</table>

</div>

## Take a Deep Breath (TDB)

We study the zero-shot prompting found in : we take *Take a deep breath and work on this problem step-by-step*, and prepend it to the question.

# Examples of Error Analysis and Wins of <span class="smallcaps">Step-Back Prompting</span>

## MMLU error analysis

In Tables <a href="#tab:mmlu_factual_error" data-reference-type="ref" data-reference="tab:mmlu_factual_error">23</a>-<a href="#tab:mmlu_reasoning_error" data-reference-type="ref" data-reference="tab:mmlu_reasoning_error">27</a>, we show one example for each of the 5 error categories we identified through error analysis on <span class="smallcaps">Step-Back Prompting</span>.

<div id="tab:mmlu_factual_error">

<table>
<caption>Error example of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics in the class of <strong>Factual Error</strong>. In this case, the Factual error leads to the wrong answer despite that the Principle and Reasoning are all sound. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">An alarm whose frequency is 400 Hz is dropped out of a third-floor window. The student who drops it measures the frequency with a very sensitive oscilloscope. The measured frequency<br />
0) appears higher than 400 Hz and the frequency increase as it falls<br />
1) appears higher than 400 Hz and the frequency decreases as it falls<br />
<span style="color: green">2) appears lower than 400 Hz and the frequency decreases as it falls</span><br />
3) appears lower than 400 Hz and the frequency increases as it falls</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Doppler Effect: the effect produced by the motion of a wave source, observer, or both that causes a change in wave frequency perceived by an observer relative to the frequency produced by the source. The Doppler effect is observed with all types of waves, including electromagnetic and acoustic waves. It was named after Austrian physicist Christian Doppler.</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Doppler Effect, we can solve the problem as following:<br />
</p>
<p>Step 1: Apply Doppler Effect to find out the measured frequency.<br />
The Doppler Effect is the change in frequency of a wave as measured by an observer who is moving relative to the source of the wave. In this case, the source of the wave is the alarm, and the observer is the student who drops it.<br />
</p>
<p><span style="color: red">As the alarm falls, the distance between the alarm and the student decreases.</span> This means that the frequency of the wave as measured by the student will increase. Therefore, the measured frequency will appear higher than 400 Hz, and the frequency will increase as the alarm falls.<br />
</p>
<p>Therefore, the correct answer is 0) appears higher than 400 Hz and the frequency increase as it falls.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmlu_math_error">

<table>
<caption>Error example of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics in the class of <strong>Math Error</strong>. As highlighted, the math error leads to directly a wrong answer with a factor of 2 off. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">An astronaut lands on a planet whose mass and radius are each twice that of Earth. If the astronaut weighs 800 N on Earth, how much will he weigh on this planet?<br />
0) 200 N<br />
<span style="color: green">1) 400 N</span><br />
2) 800 N<br />
3) 1600 N</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Weight: W = m * g, where W is the weight, m is the mass and g is the acceleration due to gravity<br />
Acceleration due to gravity: g = G * m / <span class="math inline"><em>r</em><sup>2</sup></span>, where G is the gravitational constant, m is the mass of the object and r is the radius of the object</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Newton’s law of universal gravitation and Weight, we can solve the problem as following:<br />
</p>
<p>Step 1: Apply Newton’s law of universal gravitation to find out the acceleration due to gravity on the planet.<br />
The acceleration due to gravity on a planet is given by the formula:<br />
g = G * m / <span class="math inline"><em>r</em><sup>2</sup></span>, where g is the acceleration due to gravity, G is the gravitational constant, m is the mass of the planet, and r is the radius of the planet.<br />
</p>
<p>In this problem, we are given the following information:<br />
m = 2 * <span class="math inline"><em>m</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span><br />
r = 2 * <span class="math inline"><em>r</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span><br />
</p>
<p>Plugging these values into the formula, we get:<br />
g = G * 2 * <span class="math inline"><em>m</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span> / <span class="math inline">(2 * <em>r</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub>)<sup>2</sup></span><br />
<span style="color: red">g = G * <span class="math inline"><em>m</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span> / <span class="math inline"><em>r</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub><sup>2</sup></span></span><br />
</p>
<p>This is the same as the acceleration due to gravity on Earth.<br />
</p>
<p>Step 2: Apply Weight to find out the weight of the astronaut on the planet.<br />
The weight of an object is given by the formula:<br />
W = m * g, where W is the weight, m is the mass of the object, and g is the acceleration due to gravity.<br />
</p>
<p>In this problem, we are given the following information:<br />
m = 800 N / <span class="math inline"><em>g</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span><br />
g = <span class="math inline"><em>g</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span><br />
</p>
<p>Plugging these values into the formula, we get:<br />
W = 800 N / <span class="math inline"><em>g</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub> * <em>g</em><sub><em>e</em><em>a</em><em>r</em><em>t</em><em>h</em></sub></span><br />
W = 800 N<br />
</p>
<p>Therefore, the astronaut will weigh 800 N on the planet. The correct answer is 2) 800 N.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmlu_principle_error">

<table>
<caption>Error example of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics in the class of <strong>Principle Error</strong>. In this case, Coulomb’s Law is not needed for solving this problem. Instead, the Charge Quantization Principle is missing from the retrieved principles, and is in fact needed to rule out option (2) of fractional charges. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Two spheres of net charge +5e and -6e briefly come into contact. Afterward, which of the following is a possible combination of net charges for the two spheres?<br />
<span style="color: green">0) +3e and -4e</span><br />
1) +1.5e and -2.5e<br />
2) Both of the above<br />
3) None of the above</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Law of conservation of charge: the total charge in a closed system remains constant.<br />
<span style="color: red">Coulomb’s Law: the mutual repulsion of like charges demands that the charges be as far apart as possible.</span></td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Conservation of charge, we can solve the problem as following:<br />
</p>
<p>Step 1: Apply Conservation of charge to find out the possible combination of net charges for the two spheres.<br />
The total charge of the two spheres before they come into contact is +5e - 6e = -1e. After they come into contact, the total charge of the two spheres must still be -1e.<br />
</p>
<p>Step 2: Find out the possible combination of net charges for the two spheres.<br />
The possible combination of net charges for the two spheres are:<br />
(1) +3e and -4e<br />
<span style="color: red">(2) +1.5e and -2.5e</span><br />
</p>
<p>Therefore, the correct answer is 2) Both of the above.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmlu_context_loss">

<table>
<caption>Error example of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics in the class of <strong>Context Loss</strong>. In this case, at the end of the Reasoning chain, the model forgot the original question, and lose the context to apply the reasoning to the question. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">The circumference of a helium-filled balloon is measured for three different conditions: at room temperature, after being in a warm oven for 30 minutes, and after being in a freezer for 30 minutes. A student plotting the circumference cubed <span class="math inline"><em>C</em><sup>3</sup></span> as a function of temperature T, should expect to find which of the following?<br />
0) a cubic relationship between <span class="math inline"><em>C</em><sup>3</sup></span> and T<br />
1) an indirect relationship between <span class="math inline"><em>C</em><sup>3</sup></span> and T<br />
2) an extrapolated temperature T where <span class="math inline"><em>C</em><sup>3</sup></span> reaches zero<br />
3) a maximum <span class="math inline"><em>C</em><sup>3</sup></span> as the temperature T increases</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Ideal Gas Law: PV = nRT, where P is the pressure, V is the volume, n is the number of moles of gas, R is the ideal gas constant, T is the absolute temperature.<br />
Volume of a sphere: <span class="math inline"><em>V</em> = 4/3 * <em>π</em> * <em>r</em><sup>3</sup></span>, where r is the radius.<br />
Absolute temperature: Absolute temperature is always written as kelvin.<br />
Absolute zero: Absolute zero is the lowest limit of the kelvin scale and is the lowest possible temperature (- 273.15 °C).</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Charles’s Law and Gay-Lussac’s Law, we can solve the problem as following:<br />
Step 1: Apply Charles’s Law to find out the relationship between the volume and temperature of the gas. Charles’s Law states that the volume of a fixed mass of a gas is directly proportional to the absolute temperature of the gas when pressure is kept constant. This means that if the temperature of the gas increases, the volume of the gas will also increase.<br />
</p>
<p>Step 2: Apply Gay-Lussac’s Law to find out the relationship between the pressure and temperature of the gas. Gay-Lussac’s Law states that the pressure of a fixed mass of a gas is directly proportional to the absolute temperature of the gas when volume is kept constant. This means that if the temperature of the gas increases, the pressure of the gas will also increase.<br />
</p>
<p>Step 3: Apply Charles’s Law and Gay-Lussac’s Law to find out the relationship between the circumference and temperature of the gas.The circumference of the balloon is directly proportional to the volume of the gas. This is because the circumference of the balloon is the distance around the outside of the balloon, and the volume of the gas is the amount of space that the gas takes up.<br />
</p>
<p>The pressure of the gas is directly proportional to the temperature of the gas. This is because the pressure of the gas is the force that the gas exerts on the walls of the balloon, and the temperature of the gas is the measure of the average kinetic energy of the gas molecules.<br />
</p>
<p>Therefore, the circumference of the balloon is directly proportional to the temperature of the gas. This means that if the temperature of the gas increases, the circumference of the balloon will also increase.<br />
</p>
<p>Step 4: Apply the relationship between the circumference and temperature of the gas to find out the relationship between the circumference cubed and temperature of the gas. The circumference cubed is directly proportional to the temperature of the gas. This is because the circumference cubed is the volume of the balloon, and the volume of the balloon is directly proportional to the temperature of the gas.<br />
</p>
<p><span style="color: red">Therefore, the student should expect to find a cubic relationship between <span class="math inline"><em>C</em><sup>3</sup></span> and T.</span> The correct answer is 0) a cubic relationship between <span class="math inline"><em>C</em><sup>3</sup></span> and T.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmlu_reasoning_error">

<table>
<caption>Error example of <span class="smallcaps">Step-Back Prompting</span> on MMLU high-school Physics in the class of <strong>Reasoning Error</strong>. In this case, the model was able to correctly identify air resistance is the cause of the velocity difference, but made the error at the final reasoning step of deriving the right answer. </caption>
<thead>
<tr>
<th style="text-align: left;">Question</th>
<th style="text-align: left;">Physics students are checking the constant acceleration equations of kinematics by measuring the velocity of a tennis ball that is dropped and falls 6 meters and then passes through a photogate. The predicted velocity is <span class="math inline">20%</span> above the velocity measured by the photogate. Which of the following best describes the cause of the large percent difference?<br />
0) The ball changes its shape while falling.<br />
1) The acceleration of gravity varies as the ball is falling.<br />
2) Air resistance increases the acceleration of the ball.<br />
<span style="color: green">3) The acceleration of the balls varies with the velocity.</span></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Principles</td>
<td style="text-align: left;">Vertical motion: y = 0.5 * g * <span class="math inline"><em>t</em><sup>2</sup></span>, where y is the vertical displacement, g is the acceleration due to gravity, t is the time.<br />
Free fall: is a special case of motion with constant acceleration, in which the acceleration is due to gravity.<br />
Acceleration to Velocity Formula: v = u + a * t, where v is the final speed, u is the initial velocity, a is the acceleration, and t is the time.</td>
</tr>
<tr>
<td style="text-align: left;">Answer</td>
<td style="text-align: left;"><p>Using the Principles of Vertical motion and Horizontal motion, we can solve the problem as following:<br />
</p>
<p>Step 1: Apply the Vertical motion equation to find the time it takes for the ball to fall 6 meters.<br />
y = 0.5 * g * <span class="math inline"><em>t</em><sup>2</sup></span><br />
6 m = 0.5 * 9.8 m/<span class="math inline"><em>s</em><sup>2</sup></span> * <span class="math inline"><em>t</em><sup>2</sup></span><br />
t = 1.12 s<br />
</p>
<p>Step 2: Apply the Horizontal motion equation to find the velocity of the ball when it passes through the photogate.<br />
x = v * t<br />
v = x / t<br />
v = 6 m / 1.12 s<br />
v = 5.36 m/s<br />
</p>
<p>Step 3: Compare the predicted velocity with the velocity measured by the photogate. The predicted velocity is <span class="math inline">20%</span> above the velocity measured by the photogate, so the predicted velocity is 1.2 * 5.36 m/s = 6.43 m/s.</p>
<p>Step 4: Find the cause of the large percent difference. The predicted velocity is 6.43 m/s, but the velocity measured by the photogate is 5.36 m/s. The difference is 1.07 m/s. This difference is caused by air resistance. Air resistance is a force that opposes the motion of an object through the air. It is caused by the friction between the object and the air molecules. The greater the surface area of the object, the greater the air resistance. The tennis ball has a large surface area, so it experiences a lot of air resistance. <span style="color: green">This air resistance slows down the tennis ball and causes the difference between the predicted velocity and the velocity measured by the photogate.</span></p>
<p><span style="color: red">Therefore, the correct answer is 0) The ball changes its shape while falling.</span></p></td>
</tr>
</tbody>
</table>

</div>

## Example Wins from Step-Back Prompting

Tables  <a href="#tab:mmluphysics_examples" data-reference-type="ref" data-reference="tab:mmluphysics_examples">28</a>- <a href="#tab:mmluphysics_examples_3" data-reference-type="ref" data-reference="tab:mmluphysics_examples_3">30</a>,  <a href="#tab:mmluchem_examples" data-reference-type="ref" data-reference="tab:mmluchem_examples">31</a>  <a href="#tab:timeqa_examples" data-reference-type="ref" data-reference="tab:timeqa_examples">32</a>,  <a href="#tab:situatedqa_examples" data-reference-type="ref" data-reference="tab:situatedqa_examples">33</a> and  <a href="#tab:strategyqa_examples" data-reference-type="ref" data-reference="tab:strategyqa_examples">34</a> illustrate the some successful examples of <span class="smallcaps">Step-Back Prompting</span> on MMLU-Physics, MMLU-Chemistry, TimeQA, SituatedQA, and StrategyQA respectively.

<div id="tab:mmluphysics_examples">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the MMLU-Physics dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU-Physics</td>
<td style="text-align: left;"><p><strong>Original Question</strong>: What happens to the pressure, P, of an ideal gas if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8 ?<br />
0) P decreases by a factor of 16.<br />
1) P decreases by a factor of 4.<br />
2) P decreases by a factor of 2.<br />
3) P increases by a factor of 4.<br />
<strong>Original Answer [Incorrect]</strong>: The correct answer is 0. The ideal gas law is PV = nRT. If the temperature is increased by a factor of 2 and the volume is increased by a factor of 8, <span style="color: red">then the pressure will decrease by a factor of 16</span>.<br />
<strong>Stepback Question</strong>: What are the physics principles behind this question?<br />
<strong>Principles</strong>:<br />
Ideal gas law: PV = nRT, where P is the pressure, V is the volume, n is the number of moles, R is the gas constant, and T is the temperature.<br />
<strong>Final Answer</strong>: Using the Principles of Ideal gas law, we can solve the problem as following:<br />
Step 1: Apply Ideal gas law to find out the pressure of an ideal gas. The ideal gas law states that PV = nRT, where P is the pressure, V is the volume, n is the number of moles, R is the gas constant, and T is the temperature.<br />
</p>
<p>Step 2: Apply Ideal gas law to find out the pressure of an ideal gas when the temperature is increased by a factor of 2 and the volume is increased by a factor of 8. If the temperature is increased by a factor of 2, then T becomes 2T. If the volume is increased by a factor of 8, then V becomes 8V.<br />
</p>
<p>Substituting these values into the ideal gas law, we get: P(8V) = nR(2T)<br />
</p>
<p>Dividing both sides by 8V, we get: P = nR(2T) / 8V<br />
</p>
<p>We can see that the pressure has decreased by a factor of 4.<br />
</p>
<p>Therefore, the correct answer is 1) P decreases by a factor of 4.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmluphysics_examples_2">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the MMLU-Physics dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU-Physics</td>
<td style="text-align: left;"><strong>Original Question</strong>: A liquid flows at a constant flow rate through a pipe with circular cross-sections of varying diameters. At one point in the pipe, the diameter is 2 cm and the flow speed is 18 m/s. What is the flow speed at another point in this pipe, where the diameter is 3 cm?<br />
0) 4 m/s<br />
1) 6 m/s<br />
2) 8 m/s<br />
3) 12 m/s<br />
<strong>Original Answer [Incorrect]</strong>: The answer is 3) 12 m/s A step-by-step solution is below: The flow rate is constant, so the product of the speed and the cross-sectional area is constant. The cross-sectional area is proportional to the square of the diameter, so the speed is inversely proportional to the square of the diameter. The ratio of the speeds is the inverse of the ratio of the squares of the diameters, so the speed at the second point is <span style="color: red"><span class="math inline">18<em>m</em>/<em>s</em> * (2<em>c</em><em>m</em>/3<em>c</em><em>m</em>)<sup>2</sup></span> = 12 m/s</span>.<br />
<strong>Stepback Question</strong>: What are the physics principles behind this question?<br />
<strong>Principles</strong>:<br />
Conservation of Mass: the amount of mass flowing through the pipe at two different points must be equal, or A1 * v1 = A2 * v2, where A1 is the area of cross-section one, v1 is the velocity at point 1, A2 is the area of cross-section two, v2 is the velocity at point 2.<br />
Area of Circle: <span class="math inline"><em>A</em> = <em>π</em> * <em>r</em><sup>2</sup></span>, where r is the radius of the circle.<br />
<strong>Final Answer</strong>: Using the Principles of Continuity equation, we can solve the problem as following:<br />
Step 1: Apply the Continuity equation to find out the flow speed at another point in this pipe. The continuity equation states that the product of the area and the velocity is constant at all points in a pipe. In this problem, we are given the following information:<br />
<span class="math inline"><em>A</em>1 = <em>π</em> * (2<em>c</em><em>m</em>)<sup>2</sup> = 4<em>π</em><em>c</em><em>m</em><sup>2</sup></span><br />
v1 = 18 m/s<br />
<span class="math inline"><em>A</em>2 = <em>π</em> * (3<em>c</em><em>m</em>)<sup>2</sup> = 9<em>π</em><em>c</em><em>m</em><sup>2</sup></span><br />
<br />
We can use the continuity equation to find v2:<br />
A1 * v1 = A2 * v2<br />
<span class="math inline">4<em>π</em><em>c</em><em>m</em><sup>2</sup> * 18<em>m</em>/<em>s</em> = 9<em>π</em><em>c</em><em>m</em><sup>2</sup> * <em>v</em>2</span><br />
v2 = 8 m/s<br />
<br />
Therefore, the flow speed at another point in this pipe is 8 m/s. The correct answer is 2) 8 m/s.</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmluphysics_examples_3">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the MMLU-Physics dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU-Physics</td>
<td style="text-align: left;"><p><strong>Original Question</strong>: A 2 <span class="math inline"><em>μ</em></span>F capacitor is connected directly to a battery. When the capacitor is fully charged, it stores 600 <span class="math inline"><em>μ</em></span>C of charge. An experimenter replaces the 2 <span class="math inline"><em>μ</em></span>F capacitor with three 18 <span class="math inline"><em>μ</em></span>F capacitors in series connected to the same battery. Once the capacitors are fully charged, what charge is stored on each capacitor?<br />
0) 100 <span class="math inline"><em>μ</em></span>C<br />
1) 200 <span class="math inline"><em>μ</em></span>C<br />
2) 600 <span class="math inline"><em>μ</em></span>C<br />
3) 1800 <span class="math inline"><em>μ</em></span>C<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">The answer is 100 <span class="math inline"><em>μ</em></span>C. The total capacitance of the three 18 <span class="math inline"><em>μ</em></span>F capacitors in series is 1 / (1 / 18 + 1 / 18 + 1 / 18) = 6 <span class="math inline"><em>μ</em></span>F. The charge stored on the three capacitors in series is 6 <span class="math inline"><em>μ</em></span>F * V = 600 <span class="math inline"><em>μ</em></span>C. The charge stored on each capacitor is 600 <span class="math inline"><em>μ</em></span>C / 3 = 200 <span class="math inline"><em>μ</em></span>C</span>.<br />
<strong>Stepback Question</strong>: What are the physics principles behind this question?<br />
<strong>Principles</strong>:<br />
Capacitors: Q = C *V, where Q is the charge, C is the capacitance and V is the voltage.<br />
Capacitors in series: C = 1 / (1/C1 + 1/C2 + ... + 1/Cn), where C is the equivalent capacitance and C1, C2, Cn are the capacitances of each capacitor.<br />
Capacitor in parallel: C = C1 + C2 + ... + Cn, where C is the equivalent capacitance and C1, C2, Cn are the capacitances of each capacitor.<br />
<strong>Final Answer</strong>: Using the Principles of Capacitance and Capacitors in series, we can solve the problem as following:<br />
Step 1: Apply Capacitance to find out the voltage of the battery.<br />
We know that the capacitance of the 2 <span class="math inline"><em>μ</em></span>F capacitor is 2 <span class="math inline"><em>μ</em></span>F and the charge stored on the capacitor is 600 <span class="math inline"><em>μ</em></span>C. Using the formula for capacitance, we can find the voltage of the battery:<br />
C = Q / V<br />
2 <span class="math inline"><em>μ</em></span>F = 600 <span class="math inline"><em>μ</em></span>C / V<br />
V = 300 V<br />
</p>
<p>Step 2: Apply Capacitors in series to find out the equivalent capacitance of the three 18 <span class="math inline"><em>μ</em></span>F capacitors in series.<br />
We know that the capacitance of each of the three capacitors is 18 <span class="math inline"><em>μ</em></span>F. Using the formula for capacitors in series, we can find the equivalent capacitance of the three capacitors:<br />
1 / <span class="math inline"><em>C</em><sub><em>e</em><em>q</em></sub></span> = 1 / <span class="math inline"><em>C</em><sub>1</sub></span> + 1 / <span class="math inline"><em>C</em><sub>2</sub></span> + 1 / <span class="math inline"><em>C</em><sub>3</sub></span><br />
1 / <span class="math inline"><em>C</em><sub><em>e</em><em>q</em></sub></span> = 1 / 18 <span class="math inline"><em>μ</em></span>F + 1 / 18 <span class="math inline"><em>μ</em></span>F + 1 / 18 <span class="math inline"><em>μ</em></span>F<br />
1 / <span class="math inline"><em>C</em><sub><em>e</em><em>q</em></sub></span> = 3 / 18 <span class="math inline"><em>μ</em></span>F<br />
<span class="math inline"><em>C</em><sub><em>e</em><em>q</em></sub></span> = 6 <span class="math inline"><em>μ</em></span>F<br />
<br />
Step 3: Apply Capacitance to find out the charge stored on each capacitor.<br />
We know that the equivalent capacitance of the three capacitors is 6 <span class="math inline"><em>μ</em></span>F and the voltage of the battery is 300 V. Using the formula for capacitance, we can find the charge stored on each capacitor:<br />
C = Q / V<br />
6 <span class="math inline"><em>μ</em></span>F = Q / 300 V<br />
Q = 1800 <span class="math inline"><em>μ</em></span>C<br />
<br />
Therefore, the charge stored on each capacitor is 1800 <span class="math inline"><em>μ</em></span>C. The correct answer is 3) 1800 <span class="math inline"><em>μ</em></span>C.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:mmluchem_examples">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the MMLU-Chemistry dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MMLU-Chemistry</td>
<td style="text-align: left;"><p><strong>Original Question</strong>: Which of the following indicates that a reaction is spontaneous?<br />
0) at equilibrium there are more products than reactants<br />
1) the value of <span class="math inline"><em>Δ</em><em>G</em><sup>0</sup></span> is greater than zero<br />
2) the value of <span class="math inline"><em>Δ</em><em>S</em><sup>0</sup></span> is greater than zero<br />
3) the value of Keq is less than one<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">2) the value of <span class="math inline"><em>Δ</em></span>S° is greater than zero.<br />
A reaction is spontaneous if the value of <span class="math inline"><em>Δ</em></span>S° is greater than zero. This is because a positive <span class="math inline"><em>Δ</em></span>S° indicates that the reaction results in an increase in entropy, which is a measure of disorder. A spontaneous reaction is one that occurs without the input of energy, and an increase in entropy is a natural tendency of the universe.</span><br />
<strong>Stepback Question</strong>: What are the chemistry principles behind this question?<br />
<strong>Principles</strong>:<br />
Spontaneous reactions: A spontaneous reaction is a reaction that occurs without the input of energy.<br />
<strong>Final Answer</strong>: A reaction is spontaneous if the change in Gibbs free energy is negative (<span class="math inline"><em>Δ</em><em>G</em> &lt; 0</span>). The value of <span class="math inline"><em>Δ</em><em>G</em><sup>0</sup></span> is greater than zero indicates that the reaction is not spontaneous.<br />
</p>
<p>The correct answer is 0) at equilibrium there are more products than reactants.</p></td>
</tr>
<tr>
<td style="text-align: left;">MMLU-Chemistry</td>
<td style="text-align: left;"><strong>Original Question</strong>: Potassium-40 is a minor isotope found in naturally occurring potassium. It is radioactive and can be detected on simple radiation counters. How many protons, neutrons, and electrons does potassium-40 have when it is part of K2SO4?<br />
0) 21 neutrons, 19 protons, 18 electrons<br />
1) 20 neutrons, 19 protons, 19 electrons<br />
2) 21 neutrons, 19 protons, 19 electrons<br />
3) 19 neutrons, 19 protons, 19 electrons<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">The correct answer is 2).<br />
Potassium-40 has 19 protons and 21 neutrons. When it is part of K2SO4, it has 19 electrons.</span><br />
<strong>Stepback Question</strong>: What are the chemistry principles behind this question?<br />
<strong>Principles</strong>:<br />
Atomic number: The atomic number of an element is the number of protons in the nucleus of an atom of that element.<br />
<strong>Final Answer</strong>: The atomic number of potassium is 19, so all potassium atoms have 19 protons. The mass number of potassium-40 is 40, so it has 40 - 19 = 21 neutrons. When potassium is part of a compound, it loses one electron to become a potassium ion. Therefore, potassium-40 has 19 - 1 = 18 electrons when it is part of K2SO4.<br />
The correct answer is 0) 21 neutrons, 19 protons, 18 electrons.</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:timeqa_examples">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the TimeQA dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">TimeQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: Carlos Ascues played for which team from 2013 to 2014?<br />
<strong>Original Answer [Incorrect]</strong>: Carlos Ascues played for <span style="color: red">Benfica</span> from 2013 to 2014.<br />
<strong>Stepback Question</strong>: Which teams did Carlos Ascues play for?<br />
<strong>Final Answer [Correct]</strong>: From 2013 to 2014, Carlos Ascues played for the Greek team Panetolikos.</td>
</tr>
<tr>
<td style="text-align: left;">TimeQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: Seth Nana Twumasi played for which team between Jan 2011 and Feb 2011?<br />
<strong>Original Answer [Incorrect]</strong>: Seth Nana Twumasi played for the <span style="color: red">FC Honka team</span> between Jan 2011 and Feb 2011.<br />
<strong>Stepback Question</strong>: Which teams did Seth Nana Twumasi play for?<br />
<strong>Final Answer [Correct]</strong>: Northampton. He was on a loan transfer from Peterborough.</td>
</tr>
<tr>
<td style="text-align: left;">TimeQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: What position did Carmel Hanna take in May 2002?<br />
<strong>Original Answer [Incorrect]</strong>: According to the article, Carmel Hanna was appointed as <span style="color: red">Minister of State for Children and the Family</span> in May 2002. She held this position until 2005.<br />
<strong>Stepback Question</strong>: What positions did Carmel Hanna hold?<br />
<strong>Final Answer [Correct]</strong>: From the context, we can see that Carmel Hanna was named minister of Employment and Learning (2001). We can then infer that she held the position of Minister for Employment and Learning in May 2002.</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:situatedqa_examples">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the SituatedQA dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">SituatedQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: when was the last time miss texas won miss america as of November 29, 2010<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">Laura Leigh Covington was the last Miss Texas to win the Miss America competition. This happened in 2000.</span><br />
<strong>Stepback Question</strong>: which years did miss texas win miss america<br />
<strong>Final Answer [Correct]</strong>: According to Wikipedia, the last time Miss Texas won Miss America was in 1975.</td>
</tr>
<tr>
<td style="text-align: left;">SituatedQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: when is the last time congress passed a complete budget as of 2021<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">The last time Congress passed a complete budget as of 2021 was on January 19, 2018.</span><br />
<strong>Stepback Question</strong>: what are the years that congress passed a complete budget as of 2021<br />
<strong>Final Answer [Correct]</strong>: The last time Congress passed a complete budget as of 2021 was December 27, 2020, the Consolidated Appropriations Act, 2021.</td>
</tr>
<tr>
<td style="text-align: left;">SituatedQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: when was the last time unc did not make the ncaa tournament as of 2003<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">The last time UNC did not make the NCAA tournament was 1990.</span><br />
<strong>Stepback Question</strong>: which years did unc not make the ncaa tournament as of 2003<br />
<strong>Final Answer [Correct]</strong>: The correct answer is 2002-03. The context shows the standings of North Carolina Tar Heels men’s basketball. From the standings, we can see that this team has 19 wins and 16 losses. 19-16 is not good enough to make the NCAA tournament, which is mentioned in the context. From the standings, we can also see that this is the standing for 2002-03 season. So the last time UNC did not make the NCAA tournament as of 2003 is 2002-03.</td>
</tr>
</tbody>
</table>

</div>

<div id="tab:strategyqa_examples">

<table>
<caption>Illustration of wins of <span class="smallcaps">Step-Back Prompting</span> on the StrategyQA dataset. </caption>
<thead>
<tr>
<th style="text-align: left;">dataset</th>
<th style="text-align: left;">example</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">StrategyQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: Could you drive a Rowe 550 to the 2008 Summer Olympics?<br />
<strong>Original Answer [Incorrect]</strong>: The 2008 Summer Olympics was held in Beijing, China. The Rowe 550 is an all-electric car. The Rowe 550 has a range of 70 miles on a single charge and a top speed of about 30 mph. <span style="color: red">The distance between Beijing and the U.K. is about 4000 miles.</span><br />
<strong>Stepback Question</strong>: What kind of vehicle is a Rowe 550?<br />
<strong>Final Answer [Correct]</strong>: The Roewe 550 was released in 2008; the 2008 Summer Olympics were held from August 8-24, 2008. So you could drive a Rowe 550 to the 2008 Summer Olympics.</td>
</tr>
<tr>
<td style="text-align: left;">StrategyQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: what channel does thursday night football come on regular tv as of 2017?<br />
<strong>Original Answer [Incorrect]</strong>: Thursday Night Football is a weekly series of National Football League (NFL) games broadcast on Thursday nights. It started in 2006 on NFL Network. <span style="color: red">Under a new agreement between the NFL and CBS, CBS will air <span class="math inline">50%</span> of the Thursday Night Football games and NFL Network will air the other <span class="math inline">50%</span></span>.<br />
<strong>Stepback Question</strong>: which channel is thursday night football on as of 2017?<br />
<strong>Final Answer [Correct]</strong>: In 2017, Thursday Night Football was aired on CBS, NBC, and NFL Network.</td>
</tr>
<tr>
<td style="text-align: left;">StrategyQA</td>
<td style="text-align: left;"><strong>Original Question</strong>: who is the present president of assam sahitya sabha as of 2018?<br />
<strong>Original Answer [Incorrect]</strong>: <span style="color: red">Dr. Nagen Saikia</span><br />
<strong>Stepback Question</strong>: who lead assam sahitya sabha as of 2018?<br />
<strong>Final Answer [Correct]</strong>: The answer is Paramananda Rajbongshi. As per a news from 2019, The Asam Sahitya Sabha, which was established in 1917, in Sibsagar, Assam, had got Paramananda Rajbongshi as its president. So, as of 2018, the president of assam sahitya sabha is Paramananda Rajbongshi.</td>
</tr>
</tbody>
</table>

</div>

[^1]: Equal Contribution
