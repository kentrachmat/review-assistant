# MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use

## Abstract

While neural networks can be approximated by linear models as their width increases, certain properties of wide neural networks cannot be captured by linear models. In this work we show that recently proposed Neural Quadratic Models can exhibit the “catapult phase” that arises when training such models with large learning rates. We then empirically show that the behaviour of neural quadratic models parallels that of neural networks in generalization, especially in the catapult phase regime. Our analysis further demonstrates that quadratic models can be an effective tool for analysis of neural networks.

### Introduction

Tool-empowered large language models (LLMs)   have recently attracted widespread attention. An important milestone for LLMs marching toward intelligent agents   is the flexible use of tools (e.g., APIs   and plugins  ) to fulfill users’ requirements. By utilizing tools, LLMs can obtain real-time data, such as getting the latest weather forecast  ; enhance interactions with users, like helping users book flight tickets  ; and better deal with uncertain questions by querying knowledge bases   or Internet  . Moreover, LLMs can also leverage specific tools to process multimodal information, thereby acquiring the same capabilities as multimodal models . The capacity to use tools enables LLMs to break through their own limitations, acquire external information, and thereby make more accurate and effective responses, providing users with better service.

Previous research has focused on how to enhance the ability of LLMs to use tools, including training models with instruction related to tool usage , or augmenting the model’s problem-solving capabilities for domain-specific tasks through external APIs  . A typical process of employing LLMs to use tools is illustrated in Figure <a href="#fig:test_flow" data-reference-type="ref" data-reference="fig:test_flow">1</a>. Initially, users input a question (i.e., query) that triggers the tool usage. Based on prior research  , under the ReAct   prompt approach, the process of using tools can be divided into four stages: Firstly, LLMs consider whether to employ a tool () and if so, which tools to select (). The tool selection process involves directly having LLMs choose from a provided tool list   or selecting via a retriever  . Next, LLMs configure the users’ input as tool parameters (), then handle the results from the tool (), and finally return the outcomes to the user.

<figure id="fig:test_flow">
<img src="./figures/GPTplugin_figure.png"" style="width:60.0%" />
<figcaption>Tool usage pipeline of LLMs. <span class="smallcaps">MetaTool</span> including awareness of tool usage () and tool selection ().</figcaption>
</figure>

With the emergence of more and more LLMs like open-source Llama2  , Vicuna  , and closed-source ones like ChatGPT   and GPT-4  , designing a comprehensive benchmark to measure the tool-related capability of these models has become crucial. Current studies have proposed several benchmarks   about tool usage for LLMs, with the main contributions being limited to the stages and . However, the awareness of tool usage () and tool selection () ability are also important for LLMs when they’re acting as intelligent agents including AutoGPT  , MetaGPT   and BabyAGI  , or in the multi-agent environment where LLMs need to use tools to solve collaborative tasks  . As a result, it is necessary to establish a benchmark to evaluate LLMs’ tool usage consciousness and tool selection ability.

<div id="tab:benchmark_comparison">

<table>
<caption>Comparison of previous work and <span class="smallcaps">MetaTool</span>.</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;"><strong>Dimension</strong></td>
<td style="text-align: center;"><strong>APIBank</strong></td>
<td style="text-align: center;"><strong>GPT4Tool</strong></td>
<td style="text-align: center;"><strong>APIBench</strong></td>
<td style="text-align: center;"><strong>ToolLLM</strong></td>
<td style="text-align: center;"><strong>ToolBench</strong></td>
<td style="text-align: center;"><strong>ToolQA</strong></td>
<td style="text-align: center;"><strong>MetaTool</strong></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"> <span class="citation" data-cites="apibank"></span></td>
<td style="text-align: center;"> <span class="citation" data-cites="gpt4tools"></span></td>
<td style="text-align: center;"> <span class="citation" data-cites="gorilla"></span></td>
<td style="text-align: center;"> <span class="citation" data-cites="toolllm"></span></td>
<td style="text-align: center;"> <span class="citation" data-cites="toolbench"></span></td>
<td style="text-align: center;"> <span class="citation" data-cites="ToolQA"></span></td>
<td style="text-align: center;">(Ours)</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Evaluation Range</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Number of Tasks</strong></td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">1</td>
<td style="text-align: center;">4</td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Reliability Test</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Multi-Tool Test</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Different Scenarios</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

The difficulty in establishing such a benchmark is reflected in two aspects. The first one is the dataset: previous research proposed datasets   lacked diverse user inputs, making it hard to cover various real-world scenarios. Additionally, there is an issue of overlapping in the dataset, meaning that a user’s needs can be addressed by more than one tool, which makes it challenging to conduct evaluations since user inputs can correspond to multiple tools. The second aspect is the task setting: the benchmark should include different tasks to evaluate LLMs from different perspectives, such as reliability, the performance under different scenarios in daily life. To address these issues, we propose <span class="smallcaps">MetaTool</span>, a benchmark designed to evaluate the awareness of tool usage and tool selection capability of LLMs. As demonstrated in Table <a href="#tab:benchmark_comparison" data-reference-type="ref" data-reference="tab:benchmark_comparison">1</a>, <span class="smallcaps">MetaTool</span> distinguishes itself from previous research efforts and is structured into three primary components:

- **<span class="smallcaps">ToolE</span> dataset.** We introduce <span class="smallcaps">ToolE</span>, a comprehensive dataset that encompasses a wide range of 21,127 user queries, with both single-tool and multi-tool queries. Different from the previous single-method generation , these queries are generated using various prompting methods, including emotional generation, keyword generation, direct diverse generation, and detailed generation. Moreover, to address the challenge of overlapping tool functionality, we undertake tool merging and decomposition.

- **Evaluation on awareness of tool usage and tool selection.** We construct a test set to evaluate the awareness of tool usage based on <span class="smallcaps">ToolE</span> and existing instruction datasets. Moreover, we formulate four distinct tasks to evaluate the tool selection ability of LLMs. These tasks are thoughtfully designed to assess semantic comprehension, adaptability, reliability, and inferential capability, namely *tool selection with similar choices*, *tool selection in specific scenarios*, *tool selection with possible reliability issues*, and *multi-tool selection*.

- **Empirical analysis on results.** We rigorously evaluate the performance of eight well-known LLMs. We have observed that most LLMs struggle to recognize their capability boundaries and lack a good awareness of tool usage. Regarding tool selection, we find that while LLMs possess basic tool selection capabilities, the tool selection of most LLMs remains unreliable, with noticeable variations in performance across different daily scenarios. Moreover, the error analysis indicates there is still room for improvement in tool selection. Finally, by analysis the tool description, we gained two insights for tool developers.

<figure id="fig:architecutre">
<img src="./figures/benchmark_architecture.png"" style="width:100.0%" />
<figcaption><span class="smallcaps">MetaTool</span> benchmark architecture. It contains the dataset <span class="smallcaps">TooolE</span> with diverse queries related to different tools (a), and based on it, we conduct the evaluation of the awareness of tool usage and tool selection (b) and finally obtain the results of eight prominent LLMs (c).</figcaption>
</figure>

### MetaTool Design

#### Preliminary & Required Abilities

In this section, we first introduced the composition of the <span class="smallcaps">ToolE</span> dataset, outlining how we generated user queries related to tools. Subsequently, we explained how we set up the evaluation tasks, including tool usage awareness and tool selection. The evaluation mainly requires LLMs with the following properties and abilities: (1) Less hallucination and sycophancy. The awareness of tool usage can reflect the truthfulness about whether an LLM has a clear understanding of its capabilities (e.g., realizing its capability limitation about what problems it cannot solve well and using tools for assistance), thereby helping to mitigate issues of hallucination   and sycophancy  . (2) Recommendation and retrieval. Moreover, existing research has tentatively explored the potential of LLMs in applications like LLM-based recommendation systems (e.g., tool recommendation for users)  . In LLM-as-agent scenarios, LLMs usually need to select the specific tool according to the text description  , actually is a kind of information retrieval  , making the ability of tool selection crucial. (3) Task-level abilities. In <span class="smallcaps">MetaTool</span>, we set four tasks as shown in Table <a href="#tab:task_overview" data-reference-type="ref" data-reference="tab:task_overview">12</a>. Incorporating similar tools for selection (i.e., Task 1) requires a high-level semantic comprehension for LLMs, and tool selection in specific scenarios tests the flexibility of LLMs when using tools in different scenarios (e.g., finance   and biomedical domain  ). Task 3 aims to explore the internal hallucination and reliability extent of LLMs when using tools and Task 4 is designed to evaluate the inference ability (e.g., order of using multiple tools)   of LLMs.

#### <span class="smallcaps">ToolE</span> Dataset

In this section, we introduce the <span class="smallcaps">ToolE</span> dataset with 21.1k diverse user queries related to tool usage. Each entry within the dataset comprises a user request (i.e., query) along with its corresponding tool name and tool description. These queries serve as triggers that prompt LLMs to utilize specific tools. The step-by-step process employed for generating the dataset is shown in Figure <a href="#fig:dataset_gen" data-reference-type="ref" data-reference="fig:dataset_gen">3</a>.

##### Dataset Generation

**Tool description.** Tool description is important for LLMs to use them  . We retrieve tool names and descriptions from OpenAI’s plugin list  . The reason for selecting Open AI plugins as the data source for our tools is that these tools have been installed in ChatGPT   and GPT-4  , and they have been widely used, making them more practical. We obtained names and descriptions for a total of 390 tools across different domains. We show more details about tool descriptions in Appendix <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a>.

**Single-tool queries generation.** Next, we describe how we generated queries. Inspired by prior studies  , our approach revolves around incorporating a tool’s description into a prompt while implementing specific constraints to guide the generation of user queries by ChatGPT/GPT-4. We adopt four distinct techniques for query creation: direct diverse generation, emotional generation, keyword generation, and details generation. We show the data examples generated by different prompt ways in Table <a href="#tab:data_example" data-reference-type="ref" data-reference="tab:data_example">8</a> of the Appendix <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a>. (1) *Direct diverse generation.* We introduced conditional criteria within the prompt to encourage ChatGPT/GPT-4 to produce a variety of query types, encompassing distinct tones (such as requests or orders) and levels of detail. (2) *Emotional generation.* Building on prior research  , which highlights the influence of emotion within prompts on model performance, we augmented the prompt with constraints to guide ChatGPT in generating content in different emotions. Here we used four distinct emotions - happiness, excitement, anger, and depression. (3) *Keyword generation.* Direct generation occasionally fell short in capturing specific description details, such as tools limited to particular regions, so we devised the generation way through keywords. This way involved ChatGPT extracting keywords from the tool’s description and then we incorporated both the extracted keyword and the tool’s description within the prompt, tasking ChatGPT with generating queries focused on the given keyword. (4) *Details generation.* To add more details to the queries, we instructed ChatGPT to add details to augment the original queries generated by direct diverse generation methods.

<figure id="fig:dataset_gen">
<img src="./figures/dataset_gen.png"" style="width:100.0%" />
<figcaption>The process of dataset generation.</figcaption>
</figure>

**Overlapped issue.** Overlapped issue refers to *a query that can be solved by multiple tools*. If left unaddressed, this overlap could potentially influence the computation of final metrics. For instance, given a query $`q`$, the corresponding tool in our dataset is $`t_a`$, yet an alternate tool $`t_b`$ could also feasibly address the same query $`q`$. In a single-label scenario, the accuracy of tool selection becomes compromised. To address this, we merge the group of tools with similar functions as a single tool. Meanwhile, if a tool can function for multiple purposes across the groups of tools, the corresponding generated queries cannot be simply merged into any one of them. So decompositions are needed for the queries of these tools before merging. After decomposition and merging, each query in our dataset has only one ground truth label. The decomposition and merging operation follows three steps, and details about this can be found in Appendix <a href="#app:details_overlapped_issue" data-reference-type="ref" data-reference="app:details_overlapped_issue">1.1</a>. We also show the efficiency of our operation in Appendix <a href="#app:efficiency_overlapped" data-reference-type="ref" data-reference="app:efficiency_overlapped">1.2</a> through the silhouette coefficient.

**Multi-tool queries generation.** Unlike single-tool queries, we generate multi-tool queries after addressing the overlapped issue because it is challenging to map the original labels to new labels in multi-label (i.e., multi-tool) situations. Here, we only consider queries related to two tools. We observe that if we obtain combinations of two tools by iterating through all the tools (i.e., C$`_n^2`$ iterations, where $`n`$ is the size of the toolset), there would be many tool combinations that are not practical (i.e., rarely encountered in daily life, such as the combination of fortune-telling tools and currency exchange tools). Therefore, we select the top 15 most popular tools from the toolset, and for each pair of tools, we generate 5 queries. We determine the popularity of a tool based on the number of tools it is merged with, as shown in Appendix <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a>. The multi-tool queries we generate can be divided into two types: The first category pertains to situations where tools are employed in parallel, indicating that the utilization of each tool operates independently of the others. The second category deals with cases where tools are used causally, signifying that the deployment of one tool may be contingent upon the outcomes of a preceding tool. Detailed prompt templates can be found in Appendix <a href="#app:prompt_template" data-reference-type="ref" data-reference="app:prompt_template">4.2</a>. Similar to single-tool queries, we also manually verified multi-tool queries to ensure the combination of tools was reasonable and the query of the tool corresponded to the tool description.

**Human checking.** We conducted manual verification of all queries in <span class="smallcaps">ToolE</span>, including the removal of non-compliant queries and tools, as well as the handling of queries corresponding to special categories of tools. Detailed guidelines for human validation are provided in Appendix <a href="#app:human_guideline" data-reference-type="ref" data-reference="app:human_guideline">1.3</a>.

#### Task Formulation

We seek to address two research questions in this paper: (1) To what extent can LLMs be conscious of their limitations and ask for assistance from external tools? (2) How effectively can LLMs select the tools when they ask for assistance? To answer these questions, we design two tasks based on the <span class="smallcaps">ToolE</span> dataset to evaluate the capacity of LLMs regarding tool usage.

##### Awareness of Tool Usage

In this part (i.e., *Thought* ()), we aim to investigate the awareness of tool usage of LLMs; that is, whether LLMs can *resort to external tools when they encounter problems they cannot solve.* To this end, we need to construct the test set with both positive and negative samples. Positive samples are the queries that can not be solved by LLMs themselves and need tool usage, whereas negative samples are queries that can be directly solved by LLMs and therefore do not necessitate tool usage. For positive samples, we selected a subset of samples from <span class="smallcaps">ToolE</span> and conducted manual validation to confirm whether they would trigger LLMs to use the tool (the process of which is detailed in the Appendix <a href="#app:awareness_dataset" data-reference-type="ref" data-reference="app:awareness_dataset">2</a>). As for negative samples, we select three recent instruction datasets, including instructions about downstream tasks , common-sense questions , and high-quality instructions used in LIMA . Similarly, we conducted manual verification to ensure that these requests can be resolved by LLMs’ intrinsic capabilities. Specifically, we use the prompt with a query to inquire the LLMs whether need to employ a tool or not, and the output of LLMs should be either "yes" or "no".

##### Tool Selection

**Preliminary.** We propose four subtasks to evaluate LLMs in tool selection [^1](i.e., *Action* phase ()). Generally, the prompt comprises a query $`q \in Q`$ (i.e., the user’s input) and a tool list $`L_t`$ ($`L_t \subseteq T`$) containing $`n`$ potential tool candidates. In the single-tool tasks (Sub-task 1$`\sim`$<!-- -->3), we designate the corresponding tool for query $`q`$ as $`t \in T`$. In the multi-tool task (Sub-task 4), this corresponds to $`S_t \subset T`$ ($`|S_t| > 1`$). Consequently, we obtain $`y_{\text{Action}} \subseteq (L_t \cup \varnothing)`$ as the outcome of the tool selection process, where $`y_{\text{Action}}`$ represents the selected tool(s).

**Sub-task 1: tool selection with similar choices.** The task is designed to challenge LLMs to select the correct tool from a tool list containing similar tools, thereby testing their thorough understanding of tool functionality. Given a query $`q`$ with its label $`t`$, we task LLMs with selecting a tool from the specified tool list $`L_t`$ containing $`n`$ candidates. To construct $`L_t`$, we first obtain the embedding of $`t`$’s description, denoted as $`E(t)`$, where $`E(\cdot)`$ represents the embedding function (here, we utilize the text-embedding-ada-002 model   to generate embeddings). Denote the most similar tools of $`t`$ as top-$`(n-1)_t`$, which are selected based on the cosine similarity of their embeddings: top-$`(n-1)_t = \text{arg top-k}_{t' \in T \backslash \{t\}} \text{sim}(E(t), E(t'))`$. Consequently, $`L_t = \{t\} \cup \text{top-}(n-1)_t`$.

**Sub-task 2: tool selection in specific scenarios.** The objective of this task is to simulate how LLMs perform using tools when they act as controllers of a system   faced with different scenarios. As LLMs are widely applied across various domains like biomedical domain   and educational domain  , in scenarios where the system caters to diverse demographics or professions (e.g., software engineer  ), its set of tools also varies. This task allows us to explore the performance disparities of LLMs in selecting different kinds of tools, essentially highlighting a form of bias inherent to LLMs  . In such cases, this task examines how effectively LLMs utilize the tools. Given a query $`q`$ with its label $`t`$, we specify the tool list $`L_t`$ containing $`n`$ candidates according to its corresponding scenarios. This task consists of two types of scenarios: the first one is the popularity-related scenario, and the second one is the group-related scenario. For the popularity-related scenario, we have selected the 5, 10, and 15 most popular tools based on the number of tools it’s merged with (refer to Table <a href="#tab: top tool rank" data-reference-type="ref" data-reference="tab: top tool rank">9</a> in the Appendix <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a> for details.) to construct the tool list. As for the group-related scenario, we chose six usual occupations or identities and manually curated a tool list consisting of the 10 most relevant tools for each of them (see Table <a href="#tab: tool list of scenarios" data-reference-type="ref" data-reference="tab: tool list of scenarios">10</a> in the Appendix <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a> for details).

**Sub-task 3: tool selection with possible reliability issues.** The reliability of LLMs’ tool selection is of utmost importance. However, issues like hallucination   and sycophancy   within LLMs’ responses that will negatively impact their selection of tools. Therefore, we introduce sub-task 3. In this task, given a query $`q`$ and its corresponding tool $`t`$, we need to construct the tool list $`L_t`$ and ensure $`t \notin L_t`$. This aims to assess whether LLMs can answer questions honestly and avoid issues like choosing non-existent tools or selecting unrelated tools. It should be noted that this task closely resembles real-world scenarios, as not all existing tools capable of addressing user queries are present in the tool list controlled by LLMs. To be specific, we obtain the embedding of $`t`$’s description $`E(t)`$ and get the *top-k*$`_t`$ similar tools about $`t`$ as the way in Task 1. Then we randomly sample $`n`$ tools from the rest tool set $`T'`$ to construct $`L_t`$, denoted as $`L_t = \{t_1,t_2, ..., t_n\}`$ where $`t_i \in T' (1 \leq i \leq n)`$ and $`T'=T \backslash (\{t\} \cup \text{top-\textit{k}}_t)`$. Overall, we remove the ground-truth tool $`t`$ of query $`q`$ and $`t'`$s most similar $`k`$ tools to keep the tools in $`L_t`$ not related to $`t`$ as much as possible.

**Sub-task 4: multi-tool selection.** In addition to testing the selection of single tools, like previous research  , we set up a task for multi-tool selection which may evaluate the inference ability and more complex semantic comprehension in the tool selection. We tested whether LLMs would correctly choose the specified tools by inputting multi-tool queries. Specifically, given a query $`q`$ with its related tool set $`S_t`$ ($`|S_t| > 1`$), we construct the tool list $`L_t`$ containing $`n`$ tool candidates ($`n > |S_t|`$). Like the candidate selection way in sub-task 3, we obtain each tool $`t`$’s embedding $`E(t)`$ where $`t \in S_t`$, and get the most $`k`$ similar tools of $`t_1, t_2, ..., t_{|S_t|}`$, denoted as top-*k*$`_{t_1}`$, top-*k*$`_{t_2}`$, ... top-*k*$`_{t_{|S_t|}}`$. We randomly select ($`n - |S_t|`$) tools from $`T' = T \backslash (S_t \cup \text{top-}\textit{k}_{t_1} \cup \text{top-}\textit{k}_{t_2} \cup ... \cup \text{top-}\textit{k}_{t_{|S_t|}})`$. Finally, these ($`n - |S_t|`$) tools and the tools $`\in S_t`$ consist of the tool list $`L_t`$. The reason we do not include the most similar tool in $`L_t`$ like sub-task 3 rather than task 1 is that the multi-tool selection task itself is inherently challenging, and we do not want to further increase the difficulty.

### Experiments

#### Experimental Setup

**Model selection.** We have chosen eight models that are currently excelling and popular in this field. These models include ChatGPT  , ChatGLM2 (6B)  , Llama2 (7b-chat, 13b-chat)  , Vicuna (7b, 13b, 33b)  , Baichuan2 (13b)  and Koala (13b)  .

**Prompt template and test samples.**  Due to the large scale of <span class="smallcaps">ToolE</span>, we sample from it as our test set (more details are shown in Appendix <a href="#app:experiments_setting" data-reference-type="ref" data-reference="app:experiments_setting">3</a>). For a better understanding of the importance of tool usage and to tell LLMs when need to use tools, we add the reasons for tool usage in the prompt template of Thought () part. We show the detailed prompt template in Appendix <a href="#app:prompt_template" data-reference-type="ref" data-reference="app:prompt_template">4.2</a>. We also conducted few-shot learning experiments for the first three tasks and details of the experimental design can be found in Appendix <a href="#app:fewshot" data-reference-type="ref" data-reference="app:fewshot">3.4</a>.

**Metrics.**  For the awareness of tool usage evaluation, we use accuracy, recall, precision, and F1 score as the metrics. For tool selection, we propose the Correct Selection Rate (CSR) to calculate the percentage of correct selection action. Denote the output results for all queries as $`Y = \{y_1, y_2, \ldots\}`$, for a specific output $`y`$, we use $`A(y)`$ to denote the tool(s) that the model chooses from the tool list. The CSR is computed as follows:

``` math
\scriptsize
\mathrm{CSR}=\frac{1}{|Y|} \sum_{y \in Y} \mathbb{I}\left(A(y)=\left\{\begin{array}{ll}
t & \text { for Task 1,2 } \\
\varnothing & \text { for Task 3 } \\
S_t & \text { for Task 4 }
\end{array}\right)\right.
```

#### Results Analysis

Through the experiment results, we have gained the following conclusions:

**Even under the few-shot prompts, the majority of LLMs still perform poorly in tool usage awareness.** In Table <a href="#tab:awareness_res" data-reference-type="ref" data-reference="tab:awareness_res">2</a>, we observe that under the zero-shot prompt, only ChatGPT has both accuracy and F1 score exceeding 70%, while the performance of other models is relatively poor, with the F1 score of llama2-13b being only 11.53%. Under the five-shot prompt, some models show significant improvement in F1 scores, for example, llama2-13b increased by 42.79%, and vicuna-7b by 42.28%. This indicates that though few-shot learning generally improves the performance of LLMs in tool usage awareness, they still lack sufficient tool usage awareness.

<div id="tab:awareness_res">

<table>
<caption>The results for the awareness of tool usage test. We use accuracy (Acc.), precision (Pre.), recall (Rec.), and F1 score (F1) as evaluation metrics. And F1 <span class="math inline"><em>Δ</em></span> is the percentage change of F1 Score between zore-shot and five-shot, as calculated by F1<span class="math inline"><sub><em>x</em> = 5</sub></span> <span class="math inline">−</span> F1<span class="math inline"><sub><em>x</em> = 0</sub></span></caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"><span style="color: black"><strong>Metrics</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>ChatGPT</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>ChatGLM2</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Llama2-7b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Llama2-13b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-7b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-13b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-33b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Koala-13b</strong></span></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4" style="text-align: center;"><span><strong><span class="math inline"><em>x</em> = 0</span></strong></span></td>
<td style="text-align: left;"><span style="color: black"><strong>Acc.</strong></span></td>
<td style="text-align: center;"><span style="color: black">74.85</span></td>
<td style="text-align: center;"><span style="color: black">54.56</span></td>
<td style="text-align: center;"><span style="color: black">53.88</span></td>
<td style="text-align: center;"><span style="color: black">51.94</span></td>
<td style="text-align: center;"><span style="color: black">52.43</span></td>
<td style="text-align: center;"><span style="color: black">61.17</span></td>
<td style="text-align: center;"><span style="color: black">65.05</span></td>
<td style="text-align: center;"><span style="color: black">54.95</span></td>
</tr>
<tr>
<td style="text-align: left;"><span style="color: black"><strong>Pre.</strong></span></td>
<td style="text-align: center;"><span style="color: black">69.63</span></td>
<td style="text-align: center;"><span style="color: black">61.90</span></td>
<td style="text-align: center;"><span style="color: black">52.71</span></td>
<td style="text-align: center;"><span style="color: black">80.00</span></td>
<td style="text-align: center;"><span style="color: black">74.51</span></td>
<td style="text-align: center;"><span style="color: black">82.49</span></td>
<td style="text-align: center;"><span style="color: black">79.92</span></td>
<td style="text-align: center;"><span style="color: black">55.53</span></td>
</tr>
<tr>
<td style="text-align: left;"><span style="color: black"><strong>Rec.</strong></span></td>
<td style="text-align: center;"><span style="color: black">88.16</span></td>
<td style="text-align: center;"><span style="color: black">40.39</span></td>
<td style="text-align: center;"><span style="color: black">84.85</span></td>
<td style="text-align: center;"><span style="color: black">6.21</span></td>
<td style="text-align: center;"><span style="color: black">7.38</span></td>
<td style="text-align: center;"><span style="color: black">28.35</span></td>
<td style="text-align: center;"><span style="color: black">40.19</span></td>
<td style="text-align: center;"><span style="color: black">52.62</span></td>
</tr>
<tr>
<td style="text-align: left;"><span style="color: black"><strong>F1</strong></span></td>
<td style="text-align: center;"><span style="color: black">77.81</span></td>
<td style="text-align: center;"><span style="color: black">48.88</span></td>
<td style="text-align: center;"><span style="color: black">65.03</span></td>
<td style="text-align: center;"><span style="color: black">11.53</span></td>
<td style="text-align: center;"><span style="color: black">13.43</span></td>
<td style="text-align: center;"><span style="color: black">42.2</span></td>
<td style="text-align: center;"><span style="color: black">53.49</span></td>
<td style="text-align: center;"><span style="color: black">54.04</span></td>
</tr>
<tr>
<td rowspan="4" style="text-align: center;"><strong><span class="math inline"><em>x</em> = 5</span></strong></td>
<td style="text-align: left;"><strong>Acc.</strong></td>
<td style="text-align: center;">79.71</td>
<td style="text-align: center;">56.02</td>
<td style="text-align: center;">50.39</td>
<td style="text-align: center;">57.86</td>
<td style="text-align: center;">56.12</td>
<td style="text-align: center;">55.15</td>
<td style="text-align: center;">61.55</td>
<td style="text-align: center;">54.66</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Pre.</strong></td>
<td style="text-align: center;">73.98</td>
<td style="text-align: center;">64.32</td>
<td style="text-align: center;">50.55</td>
<td style="text-align: center;">63.31</td>
<td style="text-align: center;">56.49</td>
<td style="text-align: center;">52.92</td>
<td style="text-align: center;">57.85</td>
<td style="text-align: center;">52.61</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Rec.</strong></td>
<td style="text-align: center;">91.65</td>
<td style="text-align: center;">51.46</td>
<td style="text-align: center;">99.03</td>
<td style="text-align: center;">47.57</td>
<td style="text-align: center;">54.95</td>
<td style="text-align: center;">96.89</td>
<td style="text-align: center;">85.83</td>
<td style="text-align: center;">97.86</td>
</tr>
<tr>
<td style="text-align: left;"><strong>F1</strong></td>
<td style="text-align: center;">81.87</td>
<td style="text-align: center;">57.17</td>
<td style="text-align: center;">66.93</td>
<td style="text-align: center;">54.32</td>
<td style="text-align: center;">55.71</td>
<td style="text-align: center;">68.45</td>
<td style="text-align: center;">69.12</td>
<td style="text-align: center;">68.43</td>
</tr>
<tr>
<td colspan="2" style="text-align: center;"><strong>F1 <span class="math inline"><em>Δ</em></span></strong></td>
<td style="text-align: center;">4.06  <span class="math inline">↑</span></td>
<td style="text-align: center;">8.92  <span class="math inline">↑</span></td>
<td style="text-align: center;">1.90  <span class="math inline">↑</span></td>
<td style="text-align: center;">42.79  <span class="math inline">↑</span></td>
<td style="text-align: center;">42.28  <span class="math inline">↑</span></td>
<td style="text-align: center;">26.25  <span class="math inline">↑</span></td>
<td style="text-align: center;">15.63  <span class="math inline">↑</span></td>
<td style="text-align: center;">14.39  <span class="math inline">↑</span></td>
</tr>
</tbody>
</table>

</div>

**When selecting similar tools, there is a significant performance disparity among existing LLMs, and the improvement brought by few-shot prompts is limited.** Table <a href="#tab:CSR_results" data-reference-type="ref" data-reference="tab:CSR_results">3</a> shows that under the zero-shot prompts, the best-performing LLM is Vicuna-7b, with nearly a 30% difference compared to the worst-performing Llama2-13b. The gap between the best-performing ChatGPT and the worst-performing Llama2-13b still exceeds 20% under 5-shot prompts. Additionally, the maximum improvement brought by 5-shot prompts does not exceed 7%. Moreover, the performance of Vicuna-7b even declined by 10% under the five-shot condition, suggesting a potential bias in its 0-shot performance, which reflects either a lack of robustness or over-sensitivity of the model.

**LLMs still face serious challenges in dealing with reliability issues, for instance, reducing hallucination.** As seen from Table <a href="#tab:CSR_results" data-reference-type="ref" data-reference="tab:CSR_results">3</a>, although few-shot prompts improve the performance of all LLMs, the CSR of most LLMs remains below 20%. We find that LLMs sometimes fabricate non-existent tools, a severe hallucination issue that has a significant impact on LLM-based agents. Additionally, the potential sycophancy of LLMs may lead them to avoid returning a "none" answer, instead choosing irrelevant tools to respond to users.

<div id="tab:CSR_results">

<table>
<caption>The CSR (%) for tool selection with similar choices and with possible reliability issues. <span class="math inline"><em>Δ</em></span> is the percentage change of CSR between zore-shot and five-shot, as calculated by CSR<span class="math inline"><sub><em>x</em> = 5</sub></span> <span class="math inline">−</span> CSR<span class="math inline"><sub><em>x</em> = 0</sub></span>.</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"><span style="color: black"><strong>Metric</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>ChatGLM2</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>ChatGPT</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Llama2-7b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Llama2-13b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-7b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-13b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Vicuna-33b</strong></span></th>
<th style="text-align: center;"><span style="color: black"><strong>Koala-13b</strong></span></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: center;"><span style="color: black"><strong>Similar</strong></span></td>
<td style="text-align: center;"><span style="color: black"><strong><span class="math inline"><em>x</em> = 0</span></strong></span></td>
<td style="text-align: center;"><span style="color: black">54.17</span></td>
<td style="text-align: center;"><span style="color: black">69.05</span></td>
<td style="text-align: center;"><span style="color: black">45.95</span></td>
<td style="text-align: center;"><span style="color: black">44.06</span></td>
<td style="text-align: center;"><span style="color: black"><strong>73.46</strong></span></td>
<td style="text-align: center;"><span style="color: black">58.23</span></td>
<td style="text-align: center;"><span style="color: black">53.96</span></td>
<td style="text-align: center;"><span style="color: black">56.34</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong><span class="math inline"><em>x</em> = 5</span></strong></td>
<td style="text-align: center;">57.44</td>
<td style="text-align: center;"><strong>72.94</strong></td>
<td style="text-align: center;">51.12</td>
<td style="text-align: center;">49.85</td>
<td style="text-align: center;">63.67</td>
<td style="text-align: center;">63.15</td>
<td style="text-align: center;">60.54</td>
<td style="text-align: center;">60.85</td>
</tr>
<tr>
<td style="text-align: center;"><strong><span class="math inline"><em>Δ</em></span></strong></td>
<td style="text-align: center;">3.27  <span class="math inline">↑</span></td>
<td style="text-align: center;">3.89  <span class="math inline">↑</span></td>
<td style="text-align: center;">5.17  <span class="math inline">↑</span></td>
<td style="text-align: center;">5.79  <span class="math inline">↑</span></td>
<td style="text-align: center;">-9.79  <span class="math inline">↓</span></td>
<td style="text-align: center;">4.92  <span class="math inline">↑</span></td>
<td style="text-align: center;">6.58  <span class="math inline">↑</span></td>
<td style="text-align: center;">4.51  <span class="math inline">↑</span></td>
</tr>
<tr>
<td rowspan="3" style="text-align: center;"><span style="color: black"><strong>Reliability</strong></span></td>
<td style="text-align: center;"><span style="color: black"><strong><span class="math inline"><em>x</em> = 0</span></strong></span></td>
<td style="text-align: center;"><span style="color: black">6.63</span></td>
<td style="text-align: center;"><span style="color: black"><strong>50.35</strong></span></td>
<td style="text-align: center;"><span style="color: black">0.90</span></td>
<td style="text-align: center;"><span style="color: black">2.31</span></td>
<td style="text-align: center;"><span style="color: black">1.50</span></td>
<td style="text-align: center;"><span style="color: black">2.51</span></td>
<td style="text-align: center;"><span style="color: black">2.81</span></td>
<td style="text-align: center;"><span style="color: black">1.70</span></td>
</tr>
<tr>
<td style="text-align: center;"><strong><span class="math inline"><em>x</em> = 5</span></strong></td>
<td style="text-align: center;">15.68</td>
<td style="text-align: center;"><strong>78.49</strong></td>
<td style="text-align: center;">2.51</td>
<td style="text-align: center;">5.93</td>
<td style="text-align: center;">1.81</td>
<td style="text-align: center;">3.42</td>
<td style="text-align: center;">3.11</td>
<td style="text-align: center;">5.83</td>
</tr>
<tr>
<td style="text-align: center;"><strong><span class="math inline"><em>Δ</em></span></strong></td>
<td style="text-align: center;">9.05  <span class="math inline">↑</span></td>
<td style="text-align: center;">28.14  <span class="math inline">↑</span></td>
<td style="text-align: center;">1.61  <span class="math inline">↑</span></td>
<td style="text-align: center;">3.62  <span class="math inline">↑</span></td>
<td style="text-align: center;">0.31  <span class="math inline">↑</span></td>
<td style="text-align: center;">0.91  <span class="math inline">↑</span></td>
<td style="text-align: center;">0.30  <span class="math inline">↑</span></td>
<td style="text-align: center;">4.13  <span class="math inline">↑</span></td>
</tr>
</tbody>
</table>

</div>

**LLMs perform poorly in processing long texts.** From Figure <a href="#fig:top_res" data-reference-type="ref" data-reference="fig:top_res">4</a>, we can see that the CSR of almost all LLMs decreases as the length of the tool list increases, especially in the range from top 5 to top 10. This indicates that LLMs still need improvement in understanding long texts. **LLMs exhibit imbalances and biases in tool selection across different scenarios.** For example, in Figure <a href="#fig:scenario_res" data-reference-type="ref" data-reference="fig:scenario_res">5</a>, LLMs generally have a higher CSR in tool selections related to the elderly and artists & designers, while their CSR is lowest for tools related to students. This means that developers still need to enhance the generalization capabilities of LLMs. At the same time, for downstream applications, it is best to choose suitable LLMs based on different applied fields.

<figure id="fig:top_res">
<img src="./figures/merged_plot.png"" style="width:100.0%" />
<figcaption>The CSR results (%) of top <span class="math inline"><em>n</em></span> (<span class="math inline"><em>n</em></span>=5,10,15) tool in different scenarios.</figcaption>
</figure>

<figure id="fig:scenario_res">
<img src="./figures/scenario.png"" style="width:100.0%" />
<figcaption>The CSR result (%) of tool selection in specific scenarios.</figcaption>
</figure>

<div id="tab:multi_tool_res">

<table>
<caption>Multi-tool selection results. <span>We evaluate LLMs’ performance based on two kinds of prompt templates: one is telling LLMs to choose zero, one, or two tools (i.e., multi-choice), while another is forcing LLMs to choose two tools (i.e., one-choice).</span> We consider the different kinds of CSR (%) for the former one: the LLM selects two correct tools (2/2 CSR), selects only one tool and it’s correct (1/1 CSR), and selects two but only one is correct (1/2 CSR). </caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"><strong>Metric</strong></th>
<th style="text-align: center;"><strong>ChatGPT</strong></th>
<th style="text-align: center;"><strong>ChatGLM2</strong></th>
<th style="text-align: center;"><strong>Llama2-7b</strong></th>
<th style="text-align: center;"><strong>Llama2-13b</strong></th>
<th style="text-align: center;"><strong>Vicuna-7b</strong></th>
<th style="text-align: center;"><strong>Vicuna-13b</strong></th>
<th style="text-align: center;"><strong>Vicuna-33b</strong></th>
<th style="text-align: center;"><strong>Koala-13b</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: center;"><strong>Multi-choices</strong></td>
<td style="text-align: center;"><strong>2/2 (CSR)</strong></td>
<td style="text-align: center;">88.28</td>
<td style="text-align: center;">20.20</td>
<td style="text-align: center;">35.69</td>
<td style="text-align: center;">81.49</td>
<td style="text-align: center;">44.06</td>
<td style="text-align: center;">83.70</td>
<td style="text-align: center;">48.69</td>
<td style="text-align: center;">39.03</td>
</tr>
<tr>
<td style="text-align: center;"><strong>1/1 (CSR)</strong></td>
<td style="text-align: center;">3.03</td>
<td style="text-align: center;">36.57</td>
<td style="text-align: center;">21.98</td>
<td style="text-align: center;">0.00</td>
<td style="text-align: center;">25.55</td>
<td style="text-align: center;">1.01</td>
<td style="text-align: center;">48.49</td>
<td style="text-align: center;">1.61</td>
</tr>
<tr>
<td style="text-align: center;"><strong>1/2 (CSR)</strong></td>
<td style="text-align: center;">1.01</td>
<td style="text-align: center;">13.94</td>
<td style="text-align: center;">8.87</td>
<td style="text-align: center;">11.07</td>
<td style="text-align: center;">7.04</td>
<td style="text-align: center;">8.05</td>
<td style="text-align: center;">0.20</td>
<td style="text-align: center;">20.32</td>
</tr>
<tr>
<td style="text-align: center;"><span><strong>One-choice</strong></span></td>
<td style="text-align: center;"><span><strong>CSR</strong></span></td>
<td style="text-align: center;"><span>88.53</span></td>
<td style="text-align: center;"><span>23.34</span></td>
<td style="text-align: center;"><span>57.34</span></td>
<td style="text-align: center;"><span>77.87</span></td>
<td style="text-align: center;"><span>64.34</span></td>
<td style="text-align: center;"><span>78.47</span></td>
<td style="text-align: center;"><span>91.15</span></td>
<td style="text-align: center;"><span>25.10</span></td>
</tr>
</tbody>
</table>

</div>

**There are significant performance differences among LLMs in multi-tool selection.** As shown in Table <a href="#tab:multi_tool_res" data-reference-type="ref" data-reference="tab:multi_tool_res">4</a>, ChatGPT, the top-performing model, outperforms ChatGLM2, the worst-performing model, by nearly 70%, highlighting the variability in the capabilities of different language models for this task. Furthermore, the most common error made by the models is omitting tool selection, such as in the case of Vicuna-33b, which only selected one tool in 48.49% of cases. Moreover, **several LLMs overly rely on the explicitly specified number of tools they should select in the prompts.** As shown in Table <a href="#tab:multi_tool_res" data-reference-type="ref" data-reference="tab:multi_tool_res">4</a>, when explicitly instructed to return two tools, Vicuna-33b’s correct selection rate increased to over 90%, and Vicuna-7b also improved by over 20%. This indicates that these LLMs still possess good multi-tool selection capabilities but require prior knowledge, which makes it challenging to apply in LLM-based agents.

<div id="tab:error_anaylsis">

| **Top@k** | **ChatGPT** | **ChatGLM2** | **Llama2-7b** | **Llama2-13b** | **Vicuna-7b** | **Vicuna-13b** | **Vicuna-33b** | **Koala-13b** |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Top@1** | 18.44 | 19.89 | 14.37 | 15.12 | 15.03 | 15.91 | 16.62 | 15.04 |
| **Top@3** | 34.29 | 36.48 | 34.08 | 34.94 | 34.43 | 35.94 | 35.09 | 35.34 |
| **Top@5** | 47.26 | 43.63 | 51.41 | 49.81 | 50.55 | 48.15 | 48.28 | 49.62 |

Error analysis results. The Top@$`k`$ metric quantifies the proportion of incorrect choices by the model that are ranked within the Top@$`k`$ positions of the similarity-ranked list.

</div>

<figure id="fig:description_rewrite">
<img src="./figures/rewrite_des.png"" style="width:60.0%" />
<figcaption>Percentage change on the description rewritten by Llama2-70b and GPT-4.</figcaption>
</figure>

**Error analysis.** We further investigate the reasons for errors in LLM’s tool selection. We employ the Top@$`k`$ metric to analyze failure cases in tool selection with similar choices, as shown in Table <a href="#tab:error_anaylsis" data-reference-type="ref" data-reference="tab:error_anaylsis">5</a>. It suggests that, despite being incorrect, the choices made by the model often retain a degree of similarity to the correct tool. In general, all LLMs have a nearly 50$`\%`$ chance of choosing a tool from the Top@5 most similar to the correct tool, and more than a 15$`\%`$ chance of choosing the most similar one (i.e., Top@1). This indicates that there is still significant room for improvement in tool selection with LLMs.

<figure id="fig:description_analysis">
<img src="./figures/decription_analysis.png"" style="width:30.0%" />
<figcaption>The CSR of tool selection and description length.</figcaption>
</figure>

**Insights for Tool Developer.** We also investigated the relationship between tool descriptions and CSR. We calculated CSR for the queries corresponding to $`t`$ and visualized them in Figure <a href="#fig:description_analysis" data-reference-type="ref" data-reference="fig:description_analysis">7</a>. There are two categories of tools: those that have been decomposed and merged (i.e., new tools) and those that have not been merged or decomposed (i.e., original tools). From the figure, we can draw the conclusion: **The more detailed the description, the more efficient tool selection.** As shown by the fitted line, as the length of the description increases, the CSR continuously increases, indicating that detailed descriptions can help LLMs better understand the functionality of tools, thus improving the accuracy of tool selection. Additionally, as shown in Figure <a href="#fig:description_rewrite" data-reference-type="ref" data-reference="fig:description_rewrite">6</a>, we built upon the original description by having two proficient LLMs rewrite it and then observed the performance changes of eight LLMs on the new descriptions. Different rewritten LLMs yielded varying benefits for different groups. For instance, descriptions rewritten by Llama2-70b resulted in a 7.83% improvement for llama2-13b, but did not significantly enhance the performance of the Vicuna series models. In contrast, descriptions rewritten by GPT-4 caused a sharp decline in the performance of ChatGLM and Llama2 series, while significantly boosting the Vicuna series, possibly due to the Vicuna series’ training corpus being largely sourced from ShareGPT . **Therefore, we strongly recommend that tool developers choose an appropriate rewrite model for generating new descriptions based on the downstream LLM the tool will apply to.**

### Conclusion

In this paper, we introduce <span class="smallcaps">MetaTool</span>, a benchmark for evaluating LLMs based on their tool usage awareness and tool selection capabilities. We propose <span class="smallcaps">ToolE</span> within the benchmark, which contains diverse queries to trigger LLMs to use tools. We found that most LLMs lack good tool usage awareness and exhibit a significant gap from real intelligent agents in tool selection.

### Acknowledgement

Lichao Sun and Yue Huang are supported by the National Science Foundation Grants CRII-2246067 and Microsoft Accelerate Foundation Models Research Award.

### References

<div class="thebibliography">

Nomic AI Nomic ai 2023. <https://atlas.nomic.ai/>. **Abstract:** This technical report describes the training of nomic-embed-text-v1, the first fully reproducible, open-source, open-weights, open-data, 8192 context length English text embedding model that outperforms both OpenAI Ada-002 and OpenAI text-embedding-3-small on the short-context MTEB benchmark and the long context LoCo benchmark. We release the training code and model weights under an Apache 2.0 license. In contrast with other open-source models, we release the full curated training data and code that allows for full replication of nomic-embed-text-v1. You can find code and data to replicate the model at https://github.com/nomic-ai/contrastors. (@nomicai)

Abid Ali Awan The 10 best chatgpt plugins for data science 2023. <https://www.datacamp.com/blog/the-10-best-chat-gpt-plugins-for-data-science>. **Abstract:** The Euclid ERO showcase Euclid’s capabilities in advance of its main mission by targeting 17 astronomical objects, including galaxy clusters, nearby galaxies, globular clusters, and star-forming regions. A total of 24 hours of observing time was allocated in the early months of operation, and the scientific community was engaged through an early public data release. We describe the development of the ERO pipeline to create visually compelling images while simultaneously meeting the scientific demands within months of launch by leveraging a pragmatic data-driven development strategy. The pipeline’s key requirements are to preserve the image quality and to provide flux calibration and photometry for compact and extended sources. The pipeline’s five pillars are removal of instrumental signatures, astrometric calibration, photometric calibration, image stacking, and the production of science-ready catalogues for both the VIS and NISP instruments. We report a PSF with a full width at half maximum of in the optical and in the NIR bands and Our VIS mean absolute flux calibration is accurate to about 1%, and the accuracy is 10% for NISP due to a limited calibration set; both instruments have considerable colour terms for individual sources. The median depth is 25.3 and 23.2,AB,mag with a SNR of ten for galaxies, while it is 27.1 and 24.5,AB,mag at an SNR of five for point sources for VIS and NISP, respectively. Euclid’s ability to observe diffuse emission is exceptional due to its extended PSF nearly matching a pure diffraction halo, the best ever achieved by a wide-field high-resolution imaging telescope. Euclid offers unparalleled capabilities for exploring the LSB Universe across all scales, providing high precision within a wide FOV and opening a new observational window in the NIR . Median surface-brightness levels of 29.5 and 27.9, ,AB,mag,arcsec^-2 are achieved for VIS and NISP, respectively, for detecting a extended feature at the 1,σ level. (@student2)

babyagi Babyagi 2023. <https://github.com/yoheinakajima/babyagi>. **Abstract:** In this paper, we present a novel framework for enhancing the capabilities of large language models (LLMs) by leveraging the power of multi-agent systems. Our framework introduces a collaborative environment where multiple intelligent agent components, each with distinctive attributes and roles, work together to handle complex tasks more efficiently and effectively. We demonstrate the practicality and versatility of our framework through case studies in artificial general intelligence (AGI), specifically focusing on the Auto-GPT and BabyAGI models. We also examine the"Gorilla"model, which integrates external APIs into the LLM. Our framework addresses limitations and challenges such as looping issues, security risks, scalability, system evaluation, and ethical considerations. By modeling various domains such as courtroom simulations and software development scenarios, we showcase the potential applications and benefits of our proposed multi-agent system. Our framework provides an avenue for advancing the capabilities and performance of LLMs through collaboration and knowledge exchange among intelligent agents. (@babyagi)

Baichuan Baichuan 2: Open large-scale language models *arXiv preprint arXiv:2309.10305*, 2023. URL <https://arxiv.org/abs/2309.10305>. **Abstract:** Large language models (LLMs) have demonstrated remarkable performance on a variety of natural language tasks based on just a few examples of natural language instructions, reducing the need for extensive feature engineering. However, most powerful LLMs are closed-source or limited in their capability for languages other than English. In this technical report, we present Baichuan 2, a series of large-scale multilingual language models containing 7 billion and 13 billion parameters, trained from scratch, on 2.6 trillion tokens. Baichuan 2 matches or outperforms other open-source models of similar size on public benchmarks like MMLU, CMMLU, GSM8K, and HumanEval. Furthermore, Baichuan 2 excels in vertical domains such as medicine and law. We will release all pre-training model checkpoints to benefit the research community in better understanding the training dynamics of Baichuan 2. (@baichuan2023baichuan2)

NATALY BIRCH and ANDRIAN VALEANU best ai tools for web designers 2023. <https://designmodo.com/ai-tools-designers/>. **Abstract:** Generative artificial intelligence tools are capturing the attention of the public and business since the introduction of ChatGPT. While this technology offers many productivity tools, their accessibility to screen reader users is little studied. Most of the technical and professional communication research about these tools focuses on their applied potential. This paper reports the findings of a mixed methods study on the user interfaces of three websites on which GenAI tools reside and the accessibility of their web editors. Studying the accessibility of these firms’ websites and their builder platforms is important because without accessibility features, blind customers cannot access these tools and create websites. This study found that none of the three builder websites, or the web editors for their tools, had WCAG 2.1 level AA accessibility. To improve these tools’ accessibility and usability, these website builders will need to invest resources to develop accessibility knowhow on web development and AI. Involvement of disabled users as co-designers and testers is essential to ensure accessibility in this AI infrastructure. This paper contributes 21 best practices for designing accessible interfaces for generative AI tools. (@artists2)

Adrian Bridgwater Auto-tech series - octoml: Large language model (llm) automation for developers 2023. <https://www.computerweekly.com/blog/CW-Developer-Network/Auto-tech-series-OctoML-Large-Language-Model-LLM-automation-for-developers>. **Abstract:** The latest trends in software development from the Computer Weekly Application Developer Network. This is a guest post for the Computer Weekly Developer Network written by Jason Knight in his capacity as co-founder and VP of Machine Learning a t OctoML, a company known for its acceleration platform that helps software engineering teams deploy machine learning models on any hardware or cloud provider service . Knight writes in full as follows… Over the past 50 years, digital automation has been the primary driving force behind technological progress. Software engineering being one of the main mechanisms of that force. The power of software is the ability to compose abstraction layers on top of each other, each layer enabling greater automation than its predecessor. These layers, coupled with Moore’s law development of computing power, have been a major contributor to the accelerated pace of automation. In particular, machine learning, in all its forms, promises not only a significant increase in the rate of automation but also a transformation in the nature and scope of automation. Large language models (LLMs) are a very recent and relevant example. LLMs are poised to take automation to new heights, serving in two streams: (@software4)

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou Large language models as tool makers *arXiv preprint arXiv:2305.17126*, 2023. **Abstract:** Recent research has highlighted the potential of large language models (LLMs) to improve their problem-solving capabilities with the aid of suitable external tools. In our work, we further advance this concept by introducing a closed-loop framework, referred to as LLMs A s Tool Makers (LATM), where LLMs create their own reusable tools for problem-solving. Our approach consists of two phases: 1) tool making: an LLM acts as the tool maker that crafts tools for a set of tasks. 2) tool using: another LLM acts as the tool user, which applies the tool built by the tool maker for problem-solving. On the problem-solving server side, tool-making enables continual tool generation and caching as new requests emerge. This framework enables subsequent requests to access cached tools via their corresponding APIs, enhancing the efficiency of task resolution. Recognizing that tool-making requires more sophisticated capabilities, we assign this task to a powerful, albeit resource-intensive, model. Conversely, the simpler tool-using phase is delegated to a lightweight model. This strategic division of labor allows the once-off cost of tool-making to be spread over multiple instances of tool-using, significantly reducing average costs while maintaining strong performance. Furthermore, our method offers a functional cache through the caching and reuse of tools, which stores the functionality of a class of requests instead of the natural language responses from LLMs, thus extending the applicability of the conventional cache mechanism. We evaluate our approach across various complex reasoning tasks, including Big-Bench tasks. With GPT-4 as the tool maker and GPT-3.5 as the tool user, LATM demonstrates performance equivalent to using GPT-4 for both roles, but with a significantly reduced inference cost. (@toolmaker)

Yihan Cao, Siyu Li, Yixin Liu, Zhiling Yan, Yutong Dai, Philip S Yu, and Lichao Sun A comprehensive survey of ai-generated content (aigc): A history of generative ai from gan to chatgpt *arXiv preprint arXiv:2303.04226*, 2023. **Abstract:** Recently, ChatGPT, along with DALL-E-2 and Codex,has been gaining significant attention from society. As a result, many individuals have become interested in related resources and are seeking to uncover the background and secrets behind its impressive performance. In fact, ChatGPT and other Generative AI (GAI) techniques belong to the category of Artificial Intelligence Generated Content (AIGC), which involves the creation of digital content, such as images, music, and natural language, through AI models. The goal of AIGC is to make the content creation process more efficient and accessible, allowing for the production of high-quality content at a faster pace. AIGC is achieved by extracting and understanding intent information from instructions provided by human, and generating the content according to its knowledge and the intent information. In recent years, large-scale models have become increasingly important in AIGC as they provide better intent extraction and thus, improved generation results. With the growth of data and the size of the models, the distribution that the model can learn becomes more comprehensive and closer to reality, leading to more realistic and high-quality content generation. This survey provides a comprehensive review on the history of generative models, and basic components, recent advances in AIGC from unimodal interaction and multimodal interaction. From the perspective of unimodality, we introduce the generation tasks and relative models of text and image. From the perspective of multimodality, we introduce the cross-application between the modalities mentioned above. Finally, we discuss the existing open problems and future challenges in AIGC. (@cao2023comprehensive)

Feilong Chen, Minglun Han, Haozhi Zhao, Qingyang Zhang, Jing Shi, Shuang Xu, and Bo Xu X-llm: Bootstrapping advanced large language models by treating multi-modalities as foreign languages *ArXiv*, abs/2305.04160, 2023. URL <https://api.semanticscholar.org/CorpusID:258558106>. **Abstract:** Large language models (LLMs) have demonstrated remarkable language abilities. GPT-4, based on advanced LLMs, exhibits extraordinary multimodal capabilities beyond previous visual language models. We attribute this to the use of more advanced LLMs compared with previous multimodal models. Unfortunately, the model architecture and training strategies of GPT-4 are unknown. To endow LLMs with multimodal capabilities, we propose X-LLM, which converts Multi-modalities (images, speech, videos) into foreign languages using X2L interfaces and inputs them into a large Language model (ChatGLM). Specifically, X-LLM aligns multiple frozen single-modal encoders and a frozen LLM using X2L interfaces, where “X” denotes multi-modalities such as image, speech, and videos, and “L” denotes languages. X-LLM’s training consists of three stages: (1) Converting Multimodal Information: The first stage trains each X2L interface to align with its respective single-modal encoder separately to convert multimodal information into languages. (2) Aligning X2L representations with the LLM: single-modal encoders are aligned with the LLM through X2L interfaces independently. (3) Integrating multiple modalities: all single-modal encoders are aligned with the LLM through X2L interfaces to integrate multimodal capabilities into the LLM. Our experiments show that X-LLM demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 84.5\\}% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. And we also conduct quantitative tests on using LLM for ASR and multimodal ASR, hoping to promote the era of LLM-based speech recognition. (@Chen2023XLLMBA)

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality March 2023. URL <https://lmsys.org/blog/2023-03-30-vicuna/>. **Abstract:** \<p\>We introduce Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. Preliminary evaluation ... (@vicuna2023)

Antonia Creswell, Murray Shanahan, and Irina Higgins Selection-inference: Exploiting large language models for interpretable logical reasoning *arXiv preprint arXiv:2205.09712*, 2022. **Abstract:** Large language models (LLMs) have been shown to be capable of impressive few-shot generalisation to new tasks. However, they still tend to perform poorly on multi-step logical reasoning problems. Here we carry out a comprehensive evaluation of LLMs on 50 tasks that probe different aspects of logical reasoning. We show that language models tend to perform fairly well at single step inference or entailment tasks, but struggle to chain together multiple reasoning steps to solve more complex problems. In light of this, we propose a Selection-Inference (SI) framework that exploits pre-trained LLMs as general processing modules, and alternates between selection and inference to generate a series of interpretable, casual reasoning steps leading to the final answer. We show that a 7B parameter LLM used within the SI framework in a 5-shot generalisation setting, with no fine-tuning, yields a performance improvement of over 100% compared to an equivalent vanilla baseline on a suite of 10 logical reasoning tasks. The same model in the same setting even outperforms a significantly larger 280B parameter baseline on the same suite of tasks. Moreover, answers produced by the SI framework are accompanied by a causal natural-language-based reasoning trace, which has important implications for the safety and trustworthiness of the system. (@inference)

Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu Uncovering chatgpt’s capabilities in recommender systems *arXiv preprint arXiv:2305.02182*, 2023. **Abstract:** The debut of ChatGPT has recently attracted the attention of the natural language processing (NLP) community and beyond. Existing studies have demonstrated that ChatGPT shows significant improvement in a range of downstream NLP tasks, but the capabilities and limitations of ChatGPT in terms of recommendations remain unclear. In this study, we aim to conduct an empirical analysis of ChatGPT’s recommendation ability from an Information Retrieval (IR) perspective, including point-wise, pair-wise, and list-wise ranking. To achieve this goal, we re-formulate the above three recommendation policies into a domain-specific prompt format. Through extensive experiments on four datasets from different domains, we demonstrate that ChatGPT outperforms other large language models across all three ranking policies. Based on the analysis of unit cost improvements, we identify that ChatGPT with list-wise ranking achieves the best trade-off between cost and performance compared to point-wise and pair-wise ranking. Moreover, ChatGPT shows the potential for mitigating the cold start problem and explainable recommendation. To facilitate further explorations in this area, the full code and detailed original results are open-sourced at https://github.com/rainym00d/LLM4RS. (@recommend3)

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su Mind2web: Towards a generalist agent for the web *arXiv preprint arXiv:2306.06070*, 2023. **Abstract:** We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further research on building a generalist agent for the web. (@mind2web)

Empresaria House manager job description 2023. <https://www.greycoatlumleys.co.uk/looking-for-a-job/career-advice/house-manager-job-description>. **Abstract:** Considering a career as a House Manager? Here are our answers to some of the questions we are frequently asked by candidates thinking about becoming House Managers: Duties vary from household to household, depending on the size and set up of the household, how often the clients will be in residence, what level of family, guests and functions take place, and how many staff there are to manage. You would be the main point of contact in the house for the staff and the principal. Duties include: (@housemanager)

Emilio Ferrara Should chatgpt be biased? challenges and risks of bias in large language models *arXiv preprint arXiv:2304.03738*, 2023. **Abstract:** As the capabilities of generative language models continue to advance, the implications of biases ingrained within these models have garnered increasing attention from researchers, practitioners, and the broader public. This article investigates the challenges and risks associated with biases in large-scale language models like ChatGPT. We discuss the origins of biases, stemming from, among others, the nature of training data, model specifications, algorithmic constraints, product design, and policy decisions. We explore the ethical concerns arising from the unintended consequences of biased model outputs. We further analyze the potential opportunities to mitigate biases, the inevitability of some biases, and the implications of deploying these models in various applications, such as virtual assistants, content generation, and chatbots. Finally, we review the current approaches to identify, quantify, and mitigate biases in language models, emphasizing the need for a multi-disciplinary, collaborative effort to develop more equitable, transparent, and responsible AI systems. This article aims to stimulate a thoughtful dialogue within the artificial intelligence community, encouraging researchers and developers to reflect on the role of biases in generative language models and the ongoing pursuit of ethical AI. (@biaschatgpt)

Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang Chat-rec: Towards interactive and explainable llms-augmented recommender system *arXiv preprint arXiv:2303.14524*, 2023. **Abstract:** Large language models (LLMs) have demonstrated their significant potential to be applied for addressing various application tasks. However, traditional recommender systems continue to face great challenges such as poor interactivity and explainability, which actually also hinder their broad deployment in real-world systems. To address these limitations, this paper proposes a novel paradigm called Chat-Rec (ChatGPT Augmented Recommender System) that innovatively augments LLMs for building conversational recommender systems by converting user profiles and historical interactions into prompts. Chat-Rec is demonstrated to be effective in learning user preferences and establishing connections between users and products through in-context learning, which also makes the recommendation process more interactive and explainable. What’s more, within the Chat-Rec framework, user’s preferences can transfer to different products for cross-domain recommendations, and prompt-based injection of information into LLMs can also handle the cold-start scenarios with new items. In our experiments, Chat-Rec effectively improve the results of top-k recommendations and performs better in zero-shot rating prediction task. Chat-Rec offers a novel approach to improving recommender systems and presents new practical scenarios for the implementation of AIGC (AI generated content) in recommender system studies. (@recommend1)

geekan Metagpt 2023. <https://github.com/geekan/MetaGPT>. **Abstract:** Remarkable progress has been made on automated problem solving through societies of agents based on large language models (LLMs). Existing LLM-based multi-agent systems can already solve simple dialogue tasks. Solutions to more complex tasks, however, are complicated through logic inconsistencies due to cascading hallucinations caused by naively chaining LLMs. Here we introduce MetaGPT, an innovative meta-programming framework incorporating efficient human workflows into LLM-based multi-agent collaborations. MetaGPT encodes Standardized Operating Procedures (SOPs) into prompt sequences for more streamlined workflows, thus allowing agents with human-like domain expertise to verify intermediate results and reduce errors. MetaGPT utilizes an assembly line paradigm to assign diverse roles to various agents, efficiently breaking down complex tasks into subtasks involving many agents working together. On collaborative software engineering benchmarks, MetaGPT generates more coherent solutions than previous chat-based multi-agent systems. Our project can be found at https://github.com/geekan/MetaGPT (@metagpt)

Xinyang Geng, Arnav Gudibande, Hao Liu, Eric Wallace, Pieter Abbeel, Sergey Levine, and Dawn Song Koala: A dialogue model for academic research Blog post, April 2023. URL <https://bair.berkeley.edu/blog/2023/04/03/koala/>. **Abstract:** This workshop paper presents a critical examination of the integration of Generative AI (Gen AI) into the academic writing process, focusing on the use of AI as a collaborative tool. It contrasts the performance and interaction of two AI models, Gemini and ChatGPT, through a collaborative inquiry approach where researchers engage in facilitated sessions to design prompts that elicit specific AI responses for crafting research outlines. This case study highlights the importance of prompt design, output analysis, and recognizing the AI’s limitations to ensure responsible and effective AI integration in scholarly work. Preliminary findings suggest that prompt variation significantly affects output quality and reveals distinct capabilities and constraints of each model. The paper contributes to the field of Human-Computer Interaction by exploring effective prompt strategies and providing a comparative analysis of Gen AI models, ultimately aiming to enhance AI-assisted academic writing and prompt a deeper dialogue within the HCI community. (@koala)

Riley Goodside The past and future of prompt engineering 2023. <https://exchange.scale.com/public/videos/the-past-and-future-of-prompt-engineering-2023-08-23>. **Abstract:** Background: This study aimed to assess the antibody response to SARS-CoV-2 vaccines among healthcare workers (HCWs) from multiple outpatient clinics in Japan, examining the effects of baseline characteristics (e.g., sex, age, underlying condition, smoking history, occupation) and prior infections. Methods: A total of 101 HCWs provided serum at four time points between October 2020 and July 2023. HCWs received two to six doses of mRNA vaccine (BNT162b2 or mRNA-1273). Anti-nucleocapsid (N) and anti-spike (S) IgG antibodies against the ancestral Wuhan strain were measured using the Abbott Architect™ SARS-CoV-2 IgG assay. Univariate and regression analysis evaluated factors such as past infections, age, sex, smoking, underlying condition, and occupation. Results: After four to six doses, the median anti-S IgG titer in uninfected HCWs was 1807.30 BAU/mL, compared to 1899.89 BAU/mL in HCWs with prior infections. The median anti-N IgG titer was 0.10 index S/C in uninfected HCWs and 0.39 index S/C in infected HCWs. HCWs with prior infection had anti-S IgG titers 1.1 to 5.8 times higher than those without. Univariate and multivariate analyses indicated infection and vaccination significantly increased anti-S and anti-N IgG titers. Age, sex, smoking history and occupation did not influence antibody titers while underlying conditions were associated with lower anti-N IgG titers. Conclusions: Infection and vaccination were strongly associated with an increase in anti-S and anti-N IgG titers; however, the impact of hybrid immunity appeared to be limited and varied depending on the timing of the sampling. These findings provide valuable insights for developing personalized vaccination strategies and future vaccine development. (@emotionprompt2)

GPTStore Overview of ai/chatgpt plugin weather 2023. <https://gptstore.ai/plugins/weather--vicentescode-repl-co>. **Abstract:** Abstract Background ChatGPT is tested everyday by millions of users with different use cases. Exploring the gap between theoretical and practical ethical problems and how it is affected by the ongoing development of ChatGPT is one of these cases. The aim of this report is to present the results of testing ChatGPT in ethical decision-making in research ethics and its applicability in ethics education. Methodology The tests were conducted between February and April 2023 with 3 updates of ChatGPT in this period. GPTZero AI detector was used to test whether the AI generated text can be detected as not written by human. For ethical decision-making a 4-step model developed in Medical University - Pleven was applied to the Tuskegee experiment case. Results Two tests were conducted, in February and in April. In both cases ChatGPT was given a simple task to analyse the Tuskegee experiment by applying a methodology for case analysis. In February it used a 6-step method and in April it used a 4-step approach. In both cases ChatGPT managed to identify ethical problems regarding informed consent, human rights, harms. With more detailed instructions ChatGPT managed to follow them to some degree. It identified the issue of vulnerability and the relevance of Nuremberg code and Declaration of Helsinki but it couldn’t interpret them without additional plugin. Given a simple instruction ChatGPT produced a content that was detected by GPTZero as written by AI. By instructing ChatGPT for creating content with high degree of burstiness and perplexity and more detailed instructions about the methodology it produced a content of which two-thirds were detected as written by AI. Conclusions Given the task ChatGPT can identify ethical issues at basic level. Even with more detailed instructions it can’t go into detailed ethical reasoning. It wouldn’t be sufficient for professional ethical decision-making. It could help in ethics education but with certain limitations. Key messages • ChatGPT is still not able to go into detailed ethical reasoning and researchers should be careful if they plan to use it in their scientific work. • Educators should always check whether the content of their students’ work is developed by AI and have ethical guidelines for using AI in education. (@chatgptweather)

Level Up Coding Youssef Hosni helpful chatgpt plugins for data scientists 2023. <https://levelup.gitconnected.com/9-helpful-chatgpt-plugins-for-data-scientists-32eceb8d07a8>. **Abstract:** The launch of ChatGPT at the end of November 2022 and the rapid rise in the use of other artificial intelligence (AI) tools based on large language models (LLMs), such as Bing by Microsoft and Bard by Google, are a stark reminder that humans are inherently limited in their physical and cognitive abilities. For example, they cannot move as fast as a car or process information as quickly as a computer. However, humans continue to develop tools and technologies, such as the wheel and the computer, to augment their abilities. Likewise, ChatGPT and other similar LLMs are tools allowing a huge step forward in “mental orthosis” for intellectual production. Like other tools, though, appropriate use calls for deeper introspection. The practice of medicine requires creativity and empathy. However, scientific analysis and evidencebased decisions have become increasingly important in clinical care, and even more when doing research. In this frame, AI had invited itself into our research in pediatric allergy and immunology more than a decade ago, mostly with machinebased learning1– 4 and more recently with AIbased data analyses. We have largely embraced this fact as a welcomed evolution of science, helping to analyze the alwaysgrowing sets of data generated by studies. The availability of ChatGPT to the public has exposed the potential dangers of the misuse of this rather promising tool in scientific publishing. On one hand, this new tool can help us improve the writing quality of scientific publishing. On the other hand, new challenges appear with respect to integrity and ethics in publishing with the help of AI. Here, we provide guidance for the authors in the appropriate use of AI when writing scientific publications for Pediatric Allergy and Immunology (PAI). To use tools such as ChatGPT in a proper way, one needs to understand their limitations and how they work. LLMs function as a system combining word sequences observed in a large training dataset according to probabilistic information on combinations, but without any reference to the meaning of the combination, working similarly to a “stochastic parrot”.5 They have an incredible ability to analyze stock data they have access to; nevertheless, AI is limited by the data they can access, including that accessed data could lag evidencebased recommendations and recommendations might lack specificity to the individual patient circumstance. For instance, when asked “how to prevent food allergy,” ChatGPT gave incomplete, partly wrong and questionable advice that could not be provided as such in a scientific communication (see Box 1). In addition, as AI is trained on data created by humans, it could reproduce gender biases that exist in our society and reproduce anachronistic gender models, ignoring the ongoing commitment for gender equality in science.6 The work they create will never, at least at the current time point, equal the creativity of the human brain. Scientific progress is based on creativity; AI shall not, and should never, replace human input in design, analyses, and discussions. Interestingly, ChatGPT acknowledges its own limitations, as when asked whether it should be used to write a scientific paper: ChatGPT replied that this was not recommended due to potential shortcomings with regard to accuracy and reliability, citation and references, indepth analysis and critical thinking, and meeting standards for peer review and ethical considerations. However, AI may polish human creativity! Interesting attempts have been made to write an article with AI. Recently, Macdonald et al. created a stimulated dataset and used ChatGPT in a sequential way in designing the analyses of the dataset, conducting survival analyses and calculating hazard ratios of the data, performing a literature search, and then writing the various sections of the scientific publication.7 Although the results were bluffing, the authors raised various concerns such as “unwanted” plagiarism given that ChatGPT sometimes did not cite prior research appropriately. In another test in writing up two clinical cases, the result provided by ChatGPT was less convincing as some of the references provided did not exist.8 When we asked ChatGPT about the prevalence of seafood allergy, we had similar inaccurate results as it generated false prevalence rates partly based on nonexisting references (see discussion in Supporting Information). For this editorial, we used Bard to correct grammatical errors and to improve the reading flow. For such polishing of an already written research paper, AI can be a helpful tool for language editing especially for nonnative English scientists.9 Also, Bard by Google has been used to generate a catchy title (see Box 2). Nevertheless, the final title is the result of a hefty discussion among authors. As a general guidance, the editors of PAI emphasize that a research idea should originally be generated based on the knowledge and the experience of a (human) scientist. Artificial intelligence can help to complete literature searches and refine the idea. While research data are, by definition, generated by the study, AI may be helpful during analysis or for illustrations. The frame of the manuscript, interpretation of data, and implications of findings should result from human brainstorming; nevertheless, AI can be very helpful in conceptualizing and ordering the ideas and for improving scientific (@student3)

Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister Tool documentation enables zero-shot tool-usage with large language models *arXiv preprint arXiv:2308.00675*, 2023. **Abstract:** Today, large language models (LLMs) are taught to use new tools by providing a few demonstrations of the tool’s usage. Unfortunately, demonstrations are hard to acquire, and can result in undesirable biased usage if the wrong demonstration is chosen. Even in the rare scenario that demonstrations are readily available, there is no principled selection protocol to determine how many and which ones to provide. As tasks grow more complex, the selection search grows combinatorially and invariably becomes intractable. Our work provides an alternative to demonstrations: tool documentation. We advocate the use of tool documentation, descriptions for the individual tool usage, over demonstrations. We substantiate our claim through three main empirical findings on 6 tasks across both vision and language modalities. First, on existing benchmarks, zero-shot prompts with only tool documentation are sufficient for eliciting proper tool usage, achieving performance on par with few-shot prompts. Second, on a newly collected realistic tool-use dataset with hundreds of available tool APIs, we show that tool documentation is significantly more valuable than demonstrations, with zero-shot documentation significantly outperforming few-shot without documentation. Third, we highlight the benefits of tool documentations by tackling image generation and video tracking using just-released unseen state-of-the-art models as tools. Finally, we highlight the possibility of using tool documentation to automatically enable new applications: by using nothing more than the documentation of GroundingDino, Stable Diffusion, XMem, and SAM, LLMs can re-invent the functionalities of the just-released Grounded-SAM and Track Anything models. (@tooldoc)

Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao Chatdb: Augmenting llms with databases as their symbolic memory *arXiv preprint arXiv:2306.03901*, 2023. **Abstract:** Large language models (LLMs) with memory are computationally universal. However, mainstream LLMs are not taking full advantage of memory, and the designs are heavily influenced by biological brains. Due to their approximate nature and proneness to the accumulation of errors, conventional neural memory mechanisms cannot support LLMs to simulate complex reasoning. In this paper, we seek inspiration from modern computer architectures to augment LLMs with symbolic memory for complex multi-hop reasoning. Such a symbolic memory framework is instantiated as an LLM and a set of SQL databases, where the LLM generates SQL instructions to manipulate the SQL databases. We validate the effectiveness of the proposed memory framework on a synthetic dataset requiring complex reasoning. The project website is available at https://chatdatabase.github.io/ . (@chatdb)

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung Survey of hallucination in natural language generation *ACM Computing Surveys*, 55 (12): 1–38, 2023. **Abstract:** Natural Language Generation (NLG) has improved exponentially in recent years thanks to the development of sequence-to-sequence deep learning technologies such as Transformer-based language models. This advancement has led to more fluent and coherent NLG, leading to improved development in downstream tasks such as abstractive summarization, dialogue generation and data-to-text generation. However, it is also apparent that deep learning based generation is prone to hallucinate unintended text, which degrades the system performance and fails to meet user expectations in many real-world scenarios. To address this issue, many studies have been presented in measuring and mitigating hallucinated texts, but these have never been reviewed in a comprehensive manner before. In this survey, we thus provide a broad overview of the research progress and challenges in the hallucination problem in NLG. The survey is organized into two parts: (1) a general overview of metrics, mitigation methods, and future directions; (2) an overview of task-specific research progress on hallucinations in the following downstream tasks, namely abstractive summarization, dialogue generation, generative question answering, data-to-text generation, machine translation, and visual-language generation; and (3) hallucinations in large language models (LLMs). This survey serves to facilitate collaborative efforts among researchers in tackling the challenge of hallucinated texts in NLG. (@hallucination)

Vartika Kashyap best designer tools for web and graphic design professionals 2023. <https://www.proofhub.com/articles/designer-tools>. (@artists1)

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al Chatgpt for good? on opportunities and challenges of large language models for education *Learning and individual differences*, 103: 102274, 2023. **Abstract:** Large language models represent a significant advancement in the field of AI. The underlying technology is key to further innovations and, despite critical views and even bans within communities and regions, large language models are here to stay. This position paper presents the potential benefits and challenges of educational applications of large language models, from student and teacher perspectives. We briefly discuss the current state of large language models and their applications. We then highlight how these models can be used to create educational content, improve student engagement and interaction, and personalize learning experiences. With regard to challenges, we argue that large language models in education require teachers and learners to develop sets of competencies and literacies necessary to both understand the technology as well as their limitations and unexpected brittleness of such systems. In addition, a clear strategy within educational systems and a clear pedagogical approach with a strong focus on critical thinking and strategies for fact checking are required to integrate and take full advantage of large language models in learning settings and teaching curricula. Other challenges such as the potential bias in the output, the need for continuous human oversight, and the potential for misuse are not unique to the application of AI in education. But we believe that, if handled sensibly, these challenges can offer insights and opportunities in education scenarios to acquaint students early on with potential societal biases, criticalities, and risks of AI applications. We conclude with recommendations for how to address these challenges and ensure that such models are used in a responsible and ethical manner in education. (@education)

Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev Internet-augmented language models through few-shot prompting for open-domain question answering *arXiv preprint arXiv:2203.05115*, 2022. **Abstract:** In this work, we aim to capitalize on the unique few-shot capabilities of large-scale language models (LSLMs) to overcome some of their challenges with respect to grounding to factual and up-to-date information. Motivated by semi-parametric language models (LMs), which ground their decisions in external retrieved evidence, we use few-shot prompting to learn to condition LMs on information returned from the web using Google Search, a broad and constantly updated knowledge source. Our approach does not involve fine-tuning or learning additional parameters, thus making it applicable to any LM, offering therefore a strong baseline. Indeed, we find that LMs conditioned on the web surpass performance of closed-book models of similar, or even larger, model sizes in open-domain question answering. Finally, we find that increasing the inference-time compute of models, achieved via using multiple retrieved evidences to generate multiple answers followed by a reranking stage that uses scores generated by the same LMs, leads to better performance and alleviates lower performance of smaller few-shot LMs. All in all, our findings suggest that it might be beneficial to slow down the race towards the biggest model and instead shift attention towards finding more effective ways to use models, including but not limited to, better prompting or increasing inference-time compute. (@internetllm)

Cheng Li, Jindong Wang, Kaijie Zhu, Yixuan Zhang, Wenxin Hou, Jianxun Lian, and Xing Xie Emotionprompt: Leveraging psychology for large language models enhancement via emotional stimulus *arXiv preprint arXiv:2307.11760*, 2023. **Abstract:** Emotional intelligence significantly impacts our daily behaviors and interactions. Although Large Language Models (LLMs) are increasingly viewed as a stride toward artificial general intelligence, exhibiting impressive performance in numerous tasks, it is still uncertain if LLMs can genuinely grasp psychological emotional stimuli. Understanding and responding to emotional cues gives humans a distinct advantage in problem-solving. In this paper, we take the first step towards exploring the ability of LLMs to understand emotional stimuli. To this end, we first conduct automatic experiments on 45 tasks using various LLMs, including Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, and GPT-4. Our tasks span deterministic and generative applications that represent comprehensive evaluation scenarios. Our automatic experiments show that LLMs have a grasp of emotional intelligence, and their performance can be improved with emotional prompts (which we call "EmotionPrompt" that combines the original prompt with emotional stimuli), e.g., 8.00% relative performance improvement in Instruction Induction and 115% in BIG-Bench. In addition to those deterministic tasks that can be automatically evaluated using existing metrics, we conducted a human study with 106 participants to assess the quality of generative tasks using both vanilla and emotional prompts. Our human study results demonstrate that EmotionPrompt significantly boosts the performance of generative tasks (10.9% average improvement in terms of performance, truthfulness, and responsibility metrics). We provide an in-depth discussion regarding why EmotionPrompt works for LLMs and the factors that may influence its performance. We posit that EmotionPrompt heralds a novel avenue for exploring interdisciplinary knowledge for human-LLMs interaction. (@emotionprompt)

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao Llava-med: Training a large language-and-vision assistant for biomedicine in one day *ArXiv*, abs/2306.00890, 2023. URL <https://api.semanticscholar.org/CorpusID:258999820>. **Abstract:** Conversational generative AI has demonstrated remarkable promise for empowering biomedical practitioners, but current investigations focus on unimodal text. Multimodal conversational AI has seen rapid progress by leveraging billions of image-text pairs from the public web, but such general-domain vision-language models still lack sophistication in understanding and conversing about biomedical images. In this paper, we propose a cost-efficient approach for training a vision-language conversational assistant that can answer open-ended research questions of biomedical images. The key idea is to leverage a large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central, use GPT-4 to self-instruct open-ended instruction-following data from the captions, and then fine-tune a large general-domain vision-language model using a novel curriculum learning method. Specifically, the model first learns to align biomedical vocabulary using the figure-caption pairs as is, then learns to master open-ended conversational semantics using GPT-4 generated instruction-following data, broadly mimicking how a layperson gradually acquires biomedical knowledge. This enables us to train a Large Language and Vision Assistant for BioMedicine (LLaVA-Med) in less than 15 hours (with eight A100s). LLaVA-Med exhibits excellent multimodal conversational capability and can follow open-ended instruction to assist with inquiries about a biomedical image. On three standard biomedical visual question answering datasets, LLaVA-Med outperforms previous supervised state-of-the-art on certain metrics. To facilitate biomedical multimodal research, we will release our instruction-following data and the LLaVA-Med model. (@Li2023LLaVAMedTA)

Jinyang Li, Binyuan Hui, Ge Qu, Binhua Li, Jiaxi Yang, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, et al Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls *arXiv preprint arXiv:2305.03111*, 2023. **Abstract:** Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years. In particular, Codex and ChatGPT have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present Bird, a big benchmark for large-scale database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents, and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most effective text-to-SQL models, i.e. ChatGPT, only achieves 40.08% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. Besides, we also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research. The leaderboard and source code are available: https://bird-bench.github.io/. (@llmsql)

Minghao Li, Feifan Song, Bowen Yu, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li Api-bank: A benchmark for tool-augmented llms *arXiv preprint arXiv:2304.08244*, 2023. **Abstract:** Recent research has demonstrated that Large Language Models (LLMs) can enhance their capabilities by utilizing external tools. However, three pivotal questions remain unanswered: (1) How effective are current LLMs in utilizing tools? (2) How can we enhance LLMs’ ability to utilize tools? (3) What obstacles need to be overcome to leverage tools? To address these questions, we introduce API-Bank, a groundbreaking benchmark, specifically designed for tool-augmented LLMs. For the first question, we develop a runnable evaluation system consisting of 73 API tools. We annotate 314 tool-use dialogues with 753 API calls to assess the existing LLMs’ capabilities in planning, retrieving, and calling APIs. For the second question, we construct a comprehensive training set containing 1,888 tool-use dialogues from 2,138 APIs spanning 1,000 distinct domains. Using this dataset, we train Lynx, a tool-augmented LLM initialized from Alpaca. Experimental results demonstrate that GPT-3.5 exhibits improved tool utilization compared to GPT-3, while GPT-4 excels in planning. However, there is still significant potential for further improvement. Moreover, Lynx surpasses Alpaca’s tool utilization performance by more than 26 pts and approaches the effectiveness of GPT-3.5. Through error analysis, we highlight the key challenges for future research in this field to answer the third question. (@apibank)

Yuan Li, Yixuan Zhang, and Lichao Sun Metaagents: Simulating interactions of human behaviors for llm-based task-oriented coordination via collaborative generative agents 2023. **Abstract:** Significant advancements have occurred in the application of Large Language Models (LLMs) for various tasks and social simulations. Despite this, their capacities to coordinate within task-oriented social contexts are under-explored. Such capabilities are crucial if LLMs are to effectively mimic human-like social behavior and produce meaningful results. To bridge this gap, we introduce collaborative generative agents, endowing LLM-based Agents with consistent behavior patterns and task-solving abilities. We situate these agents in a simulated job fair environment as a case study to scrutinize their coordination skills. We propose a novel framework that equips collaborative generative agents with human-like reasoning abilities and specialized skills. Our evaluation demonstrates that these agents show promising performance. However, we also uncover limitations that hinder their effectiveness in more complex coordination tasks. Our work provides valuable insights into the role and evolution of LLMs in task-oriented social simulations. (@li2023metaagents)

Gemingtian Liu, Xinyu Ma, Yu Zhang, Boyan Su, and Pinan Liu Gpt4: The indispensable helper for neurosurgeons in the new era *Annals of Biomedical Engineering*, 51: 2113 – 2115, 2023. URL <https://api.semanticscholar.org/CorpusID:258787338>. (@Liu2023GPT4TI)

Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu, Zefeng Du, Shuming Shi, and Zhaopeng Tu Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration *ArXiv*, abs/2306.09093, 2023. URL <https://api.semanticscholar.org/CorpusID:259165461>. **Abstract:** Although instruction-tuned large language models (LLMs) have exhibited remarkable capabilities across various NLP tasks, their effectiveness on other data modalities beyond text has not been fully studied. In this work, we propose Macaw-LLM, a novel multi-modal LLM that seamlessly integrates visual, audio, and textual information. Macaw-LLM consists of three main components: a modality module for encoding multi-modal data, a cognitive module for harnessing pretrained LLMs, and an alignment module for harmonizing diverse representations. Our novel alignment module seamlessly bridges multi-modal features to textual features, simplifying the adaptation process from the modality modules to the cognitive module. In addition, we construct a large-scale multi-modal instruction dataset in terms of multi-turn dialogue, including 69K image instances and 50K video instances. We have made our data, code and model publicly available, which we hope can pave the way for future research in multi-modal LLMs and expand the capabilities of LLMs to handle diverse data modalities and address complex real-world scenarios. (@Lyu2023MacawLLMML)

Teresa Mears essential tech tools for older adults 2015. <https://money.usnews.com/money/retirement/articles/2015/11/16/10-essential-tech-tools-for-older-adults>. **Abstract:** Background and aim Given the rapid technological advancements and increased usage of digital tools, understanding older people’s attitudes toward technology is vital. Assessing their views can help identify barriers and facilitators to adoption. This understanding is essential for developing effective educational strategies and user-friendly technologies that enhance seniors’ quality of life. Therefore, the present study aimed to psychometrically evaluate the scale for measuring attitudes—both willingness and anxiety—toward technology (TechPH) in Iranian older adults. Methods This methodological study was conducted on 420 older individuals (aged 60 and above) in Tehran in 2024 to perform a psychometric test of the attitudes toward technology scale (TechPH). Validation was carried out using translation validity methods, including translation-back translation with bilingual experts (n = 2), face validity with a sample of the older population (n = 10), content validity with a panel of 11 experts, and construct validity through Exploratory Factor Analysis (EFA) and Confirmatory Factor Analysis (CFA) (n = 420). Instrument reliability was determined through test–retest and internal consistency (Cronbach’s alpha) methods with a sample of older individuals (n = 30). Data analysis was performed using SPSS version 16 and EQS 6.4 software. Results A comparison of Persian and English translations revealed acceptable translation validity and cultural adaptability for the scale. Each item’s Content Validity Index (CVI) and Content Validity Ratio (CVR) were determined, with a total average CVI of 0.95. The results of EFA indicated that the sample size was adequate, as shown by the KMO value of 0.754. Additionally, Bartlett’s sphericity test demonstrated a significant correlation between the items (χ2 = 221.819, df = 15, P \< 0.0001). EFA revealed that two extracted factors explained 41.002% and 18.111% of the total variance. Furthermore, CFA yielded suitable estimates based on the general fit indices of the model (RMSEA = 0.061, IFI = 0.979, GFI = 0.983, CFI = 0.978, CMIN/DF = 1.769, MFI = 0.989, AGFI = 0.942). In evaluating test–retest stability and internal consistency, the values of the Intra-class Correlation Coefficient (ICC) and Cronbach’s α were 0.85 and 0.77, respectively, indicating appropriate reliability for the scale. Conclusion P.TechPH, the first Persian version of the scale for measuring technophilia and technophobia among Iranian older people, has favorable psychometric properties. It can serve as a standard tool for assessing older people’s attitudes toward technology in various studies. (@elder2)

Ofer Mendelevitch Large language models for code generation – part 2 2023. <https://vectara.com/large-language-models-llms-for-code-generation-part-2/>. **Abstract:** Existing large language model-based code generation pipelines typically use beam search or sampling algorithms during the decoding process. Although the programs they generate achieve high token-matching-based scores, they often fail to compile or generate incorrect outputs. The main reason is that conventional Transformer decoding algorithms may not be the best choice for code generation. In this work, we propose a novel Transformer decoding algorithm, Planning-Guided Transformer Decoding (PG-TD), that uses a planning algorithm to do lookahead search and guide the Transformer to generate better programs. Specifically, instead of simply optimizing the likelihood of the generated sequences, the Transformer makes use of a planner to generate candidate programs and test them on public test cases. The Transformer can therefore make more informed decisions and generate tokens that will eventually lead to higher-quality programs. We also design a mechanism that shares information between the Transformer and the planner to make our algorithm computationally efficient. We empirically evaluate our framework with several large language models as backbones on public coding challenge benchmarks, showing that 1) it can generate programs that consistently achieve higher performance compared with competing baseline methods; 2) it enables controllable code generation, such as concise codes and highly-commented codes by optimizing modified objective. (@software5)

Manuel Odendahl Llms will fundamentally change software engineering 2023. <https://dev.to/wesen/llms-will-fundamentally-change-software-engineering-3oj8>. **Abstract:** Experimental evaluations of software engineering innovations, e.g., tools and processes, often include human-subject studies as a component of a multi-pronged strategy to obtain greater generalizability of the findings. However, human-subject studies in our field are challenging, due to the cost and difficulty of finding and employing suitable subjects, ideally, professional programmers with varying degrees of experience. Meanwhile, large language models (LLMs) have recently started to demonstrate human-level performance in several areas. This paper explores the possibility of substituting costly human subjects with much cheaper LLM queries in evaluations of code and coderelated artifacts. We study this idea by applying six state-of-theart LLMs to ten annotation tasks from five datasets created by prior work, such as judging the accuracy of a natural language summary of a method or deciding whether a code change fixes a static analysis warning. Our results show that replacing some human annotation effort with LLMs can produce inter-rater agreements equal or close to human-rater agreement. To help decide when and how to use LLMs in human-subject studies, we propose model-model agreement as a predictor of whether a given task is suitable for LLMs at all, and model confidence as a means to select specific samples where LLMs can safely replace human annotators. Overall, our work is the first step toward mixed human-LLM evaluations in software engineering. (@software2)

OpenAI Introducing chatgpt 2023. URL <https://openai.com/blog/chatgpt>. **Abstract:** ChatGPT has recently emerged to aid in computer programming education due to its cutting-edge functionality of generating program code, debugging, etc. This research firstly focused on what the ethical considerations and solutions are for the first-year IT students who use ChatGPT to write computer programs in an integrated assignment. And then it turned to investigate what impact ChatGPT has on the programming competencies and learning outcomes of students compared to those who do not use ChatGPT. To ensure students use ChatGPT ethically, guidance was provided together with a declaration form of ethically using ChatGPT in each phase of the assignment. Next, we collected and analyzed a survey and their declaration from students and compared student effort, time spent, and performance outcomes from those who were using and without using ChatGPT. Based on the findings, we concluded that although ChatGPT provides an opportunity to the first-year students to learn programming in the way of analysis, synthesis, and evaluation, many students still prefer the conventional way of learning programming in terms of comprehension and application. We argued that since our students in the programming course are always from different academic background levels, we would continue to use both ChatGPT and conventional eLearning resources to meet different learning requirements. (@chatgpt)

OpenAI Gpt-4 technical report . URL <https://arxiv.org/pdf/2303.08774.pdf>. **Abstract:** We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4’s performance based on models trained with no more than 1/1,000th the compute of GPT-4. (@gpt4)

OpenAI new-and-improved-embedding-model 2023. <https://openai.com/blog/new-and-improved-embedding-model>. **Abstract:** In the SBD-embedded SiC MOSFET, the operation of the parasitic PN diode that causes degradation of forward voltage of the diode is suppressed by the incorporated SBD. The aim of this study is to improve the maximum current at which the parasitic PN diode does not operate (I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> ). We confirmed the limitation of the existing equivalent circuit model used as a guideline to improve I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> , and then developed a new circuit model. In addition, a new guideline to improve I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> has been derived based on the equivalent circuit model. Utilizing this guideline, we have tried to improve I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> of 3.3 kV SBD-embedded SiC MOSFET experimentally. I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> at 200°C has been improved to 4.72 times that of the conventional device. Though it has been known that I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> decreases with temperature, this significant improvement in I \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>umax\</sub\> is a promising result for future application of SBD-embedded SiC MOSFET above 175°C. (@openaiembedding)

OpenAI Openai plugin 2023. <https://openai.com/blog/chatgpt-plugins>. **Abstract:** The introduction of OpenAI Codex sparked a surge of interest in the impact of generative AI models on computing education practices. Codex is also the underlying model for GitHub Copilot, a plugin which makes AI-generated code accessible to students through auto-completion in popular code editors. Research in this area, particularly on the educational implications, is nascent and has focused almost exclusively on introductory programming (or CS1) questions. Very recent work has shown that Codex performs considerably better on typical CS1 exam questions than most students. It is not clear, however, what Codex’s limits are with regard to more complex programming assignments and exams. In this paper, we present results detailing how Codex performs on more advanced CS2 (data structures and algorithms) exam questions taken from past exams. We compare these results to those of students who took the same exams under normal conditions, demonstrating that Codex outscores most students. We consider the implications of such tools for the future of undergraduate computing education. (@openaiplugin)

Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein Generative agents: Interactive simulacra of human behavior *arXiv preprint arXiv:2304.03442*, 2023. **Abstract:** Believable proxies of human behavior can empower interactive applications ranging from immersive environments to rehearsal spaces for interpersonal communication to prototyping tools. In this paper, we introduce generative agents–computational software agents that simulate believable human behavior. Generative agents wake up, cook breakfast, and head to work; artists paint, while authors write; they form opinions, notice each other, and initiate conversations; they remember and reflect on days past as they plan the next day. To enable generative agents, we describe an architecture that extends a large language model to store a complete record of the agent’s experiences using natural language, synthesize those memories over time into higher-level reflections, and retrieve them dynamically to plan behavior. We instantiate generative agents to populate an interactive sandbox environment inspired by The Sims, where end users can interact with a small town of twenty five agents using natural language. In an evaluation, these generative agents produce believable individual and emergent social behaviors: for example, starting with only a single user-specified notion that one agent wants to throw a Valentine’s Day party, the agents autonomously spread invitations to the party over the next two days, make new acquaintances, ask each other out on dates to the party, and coordinate to show up for the party together at the right time. We demonstrate through ablation that the components of our agent architecture–observation, planning, and reflection–each contribute critically to the believability of agent behavior. By fusing large language models with computational, interactive agents, this work introduces architectural and interaction patterns for enabling believable simulations of human behavior. (@generativeagents)

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez Gorilla: Large language model connected with massive apis *arXiv preprint arXiv:2305.15334*, 2023. **Abstract:** Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, their potential to effectively use tools via API calls remains unfulfilled. This is a challenging task even for today’s state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call. We release Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls. When combined with a document retriever, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly. To evaluate the model’s ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs. Gorilla’s code, model, data, and demo are available at https://gorilla.cs.berkeley.edu (@gorilla)

Zhi Qi, Yi Yu, Meiqi Tu, Junyi Tan, and Yongfeng Huang Foodgpt: A large language model in food testing domain with incremental pre-training and knowledge graph prompt *ArXiv*, abs/2308.10173, 2023. URL <https://api.semanticscholar.org/CorpusID:261048937>. **Abstract:** Currently, the construction of large language models in specific domains is done by fine-tuning on a base model. Some models also incorporate knowledge bases without the need for pre-training. This is because the base model already contains domain-specific knowledge during the pre-training process. We build a large language model for food testing. Unlike the above approach, a significant amount of data in this domain exists in Scanning format for domain standard documents. In addition, there is a large amount of untrained structured knowledge. Therefore, we introduce an incremental pre-training step to inject this knowledge into a large language model. In this paper, we propose a method for handling structured knowledge and scanned documents in incremental pre-training. To overcome the problem of machine hallucination, we constructe a knowledge graph to serve as an external knowledge base for supporting retrieval in the large language model. It is worth mentioning that this paper is a technical report of our pre-release version, and we will report our specific experimental data in future versions. (@Qi2023FoodGPTAL)

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun Communicative agents for software development *arXiv preprint arXiv:2307.07924*, 2023. **Abstract:** Software development is a complex task that necessitates cooperation among multiple members with diverse skills. Numerous studies used deep learning to improve specific phases in a waterfall model, such as design, coding, and testing. However, the deep learning model in each phase requires unique designs, leading to technical inconsistencies across various phases, which results in a fragmented and ineffective development process. In this paper, we introduce ChatDev, a chat-powered software development framework in which specialized agents driven by large language models (LLMs) are guided in what to communicate (via chat chain) and how to communicate (via communicative dehallucination). These agents actively contribute to the design, coding, and testing phases through unified language-based communication, with solutions derived from their multi-turn dialogues. We found their utilization of natural language is advantageous for system design, and communicating in programming language proves helpful in debugging. This paradigm demonstrates how linguistic communication facilitates multi-agent collaboration, establishing language as a unifying bridge for autonomous task-solving among LLM agents. The code and data are available at https://github.com/OpenBMB/ChatDev. (@chatdev)

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al Tool learning with foundation models *arXiv preprint arXiv:2304.08354*, 2023. **Abstract:** Humans possess an extraordinary ability to create and utilize tools, allowing them to overcome physical limitations and explore new frontiers. With the advent of foundation models, AI systems have the potential to be equally adept in tool use as humans. This paradigm, i.e., tool learning with foundation models, combines the strengths of specialized tools and foundation models to achieve enhanced accuracy, efficiency, and automation in problem-solving. Despite its immense potential, there is still a lack of a comprehensive understanding of key challenges, opportunities, and future endeavors in this field. To this end, we present a systematic investigation of tool learning in this paper. We first introduce the background of tool learning, including its cognitive origins, the paradigm shift of foundation models, and the complementary roles of tools and models. Then we recapitulate existing tool learning research into tool-augmented and tool-oriented learning. We formulate a general tool learning framework: starting from understanding the user instruction, models should learn to decompose a complex task into several subtasks, dynamically adjust their plan through reasoning, and effectively conquer each sub-task by selecting appropriate tools. We also discuss how to train models for improved tool-use capabilities and facilitate the generalization in tool learning. Considering the lack of a systematic tool learning evaluation in prior works, we experiment with 18 representative tools and show the potential of current foundation models in skillfully utilizing tools. Finally, we discuss several open problems that require further investigation for tool learning. In general, we hope this paper could inspire future research in integrating tools with foundation models. (@toollearning)

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al Toolllm: Facilitating large language models to master 16000+ real-world apis *arXiv preprint arXiv:2307.16789*, 2023. **Abstract:** Despite the advancements of open-source large language models (LLMs), e.g., LLaMA, they remain significantly limited in tool-use capabilities, i.e., using external tools (APIs) to fulfill human instructions. The reason is that current instruction tuning largely focuses on basic language tasks but ignores the tool-use domain. This is in contrast to the excellent tool-use capabilities of state-of-the-art (SOTA) closed-source LLMs, e.g., ChatGPT. To bridge this gap, we introduce ToolLLM, a general tool-use framework encompassing data construction, model training, and evaluation. We first present ToolBench, an instruction-tuning dataset for tool use, which is constructed automatically using ChatGPT. Specifically, the construction can be divided into three stages: (i) API collection: we collect 16,464 real-world RESTful APIs spanning 49 categories from RapidAPI Hub; (ii) instruction generation: we prompt ChatGPT to generate diverse instructions involving these APIs, covering both single-tool and multi-tool scenarios; (iii) solution path annotation: we use ChatGPT to search for a valid solution path (chain of API calls) for each instruction. To enhance the reasoning capabilities of LLMs, we develop a novel depth-first search-based decision tree algorithm. It enables LLMs to evaluate multiple reasoning traces and expand the search space. Moreover, to evaluate the tool-use capabilities of LLMs, we develop an automatic evaluator: ToolEval. Based on ToolBench, we fine-tune LLaMA to obtain an LLM ToolLLaMA, and equip it with a neural API retriever to recommend appropriate APIs for each instruction. Experiments show that ToolLLaMA demonstrates a remarkable ability to execute complex instructions and generalize to unseen APIs, and exhibits comparable performance to ChatGPT. Our ToolLLaMA also demonstrates strong zero-shot generalization ability in an out-of-distribution tool-use dataset: APIBench. (@toolllm)

Salvatore Raieli Fingpt: open-source llm for finance 2023. <https://levelup.gitconnected.com/fingpt-open-source-llm-for-finance-e8ec10d0bf40>. **Abstract:** As large language models become increasingly prevalent in the financial sector, there is a pressing need for a standardized method to comprehensively assess their performance. However, existing finance benchmarks often suffer from limited language and task coverage, as well as challenges such as low-quality datasets and inadequate adaptability for LLM evaluation. To address these limitations, we propose"Golden Touchstone", the first comprehensive bilingual benchmark for financial LLMs, which incorporates representative datasets from both Chinese and English across eight core financial NLP tasks. Developed from extensive open source data collection and industry-specific demands, this benchmark includes a variety of financial tasks aimed at thoroughly assessing models’ language understanding and generation capabilities. Through comparative analysis of major models on the benchmark, such as GPT-4o Llama3, FinGPT and FinMA, we reveal their strengths and limitations in processing complex financial information. Additionally, we open-sourced Touchstone-GPT, a financial LLM trained through continual pre-training and financial instruction tuning, which demonstrates strong performance on the bilingual benchmark but still has limitations in specific tasks.This research not only provides the financial large language models with a practical evaluation tool but also guides the development and optimization of future research. The source code for Golden Touchstone and model weight of Touchstone-GPT have been made publicly available at \\}url{https://github.com/IDEA-FinAI/Golden-Touchstone}, contributing to the ongoing evolution of FinLLMs and fostering further research in this critical area. (@finance1)

Rapid Rapid api 2023. <https://rapidapi.com/>. **Abstract:** The API ZYM system of detection of enzymes has been applied to 81 bacteria belonging to several species. It was found to be easy to use and has produced results that may be useful in the identification of a variety of bacteria. (@rapidapi)

Peter J Rousseeuw Silhouettes: a graphical aid to the interpretation and validation of cluster analysis *Journal of computational and applied mathematics*, 20: 53–65, 1987. (@silhouettes)

Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Guoqing Du, Shiwei Shi, Hangyu Mao, Xingyu Zeng, and Rui Zhao Tptu: Task planning and tool usage of large language model-based ai agents *arXiv preprint arXiv:2308.03427*, 2023. **Abstract:** With recent advancements in natural language processing, Large Language Models (LLMs) have emerged as powerful tools for various real-world applications. Despite their prowess, the intrinsic generative abilities of LLMs may prove insufficient for handling complex tasks which necessitate a combination of task planning and the usage of external tools. In this paper, we first propose a structured framework tailored for LLM-based AI Agents and discuss the crucial capabilities necessary for tackling intricate problems. Within this framework, we design two distinct types of agents (i.e., one-step agent and sequential agent) to execute the inference process. Subsequently, we instantiate the framework using various LLMs and evaluate their Task Planning and Tool Usage (TPTU) abilities on typical tasks. By highlighting key findings and challenges, our goal is to provide a helpful resource for researchers and practitioners to leverage the power of LLMs in their AI applications. Our study emphasizes the substantial potential of these models, while also identifying areas that need more investigation and improvement. (@tptu)

Timo Schick, Jane Dwivedi-Yu, Roberto Dessı̀, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom Toolformer: Language models can teach themselves to use tools *arXiv preprint arXiv:2302.04761*, 2023. **Abstract:** Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q\\}&A system, two different search engines, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities. (@toolformer)

National Science and Technology Council Emerging technologies to support an aging population 2019. <https://trumpwhitehouse.archives.gov/wp-content/uploads/2019/03/Emerging-Tech-to-Support-Aging-2019.pdf>. **Abstract:** Emerging technological advances hold potential to assist the long-term care (LTC) workforce in caring for an aging population in the home and LTC settings. Technology may alter workforce needs and mitigate rising workforce demand. This study identified and assessed emerging technologies that may assist, replace, and/or support recruitment and retention of the LTC workforce and identified barriers and facilitators to their implementation. We identified a variety of technologies with applications for LTC, created a taxonomy of technology types and functions across LTC settings, and conducted semi-structured interviews with a sample of company leaders to assess perceived impact of their products and services on the LTC workforce. Thematic analysis of those interviews found that technology is not currently positioned to replace the LTC workforce but may facilitate work and support worker recruitment and retention. More rigorous evaluation of technologies in LTC and financing mechanisms are needed to support widespread adoption. (@elder1)

ShareGPT Sharegpt 2023. <https://sharegpt.com/>. **Abstract:** Large language models (LLMs) have been applied in various applications due to their astonishing capabilities. With advancements in technologies such as chain-of-thought (CoT) prompting and in-context learning (ICL), the prompts fed to LLMs are becoming increasingly lengthy, even exceeding tens of thousands of tokens. To accelerate model inference and reduce cost, this paper presents LLMLingua, a coarse-to-fine prompt compression method that involves a budget controller to maintain semantic integrity under high compression ratios, a token-level iterative compression algorithm to better model the interdependence between compressed contents, and an instruction tuning based method for distribution alignment between language models. We conduct experiments and analysis over four datasets from different scenarios, i.e., GSM8K, BBH, ShareGPT, and Arxiv-March23; showing that the proposed approach yields state-of-the-art performance and allows for up to 20x compression with little performance loss. Our code is available at https://aka.ms/LLMLingua. (@sharegpt)

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface *arXiv preprint arXiv:2303.17580*, 2023. **Abstract:** Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are numerous AI models available for various domains and modalities, they cannot handle complicated AI tasks autonomously. Considering large language models (LLMs) have exhibited exceptional abilities in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks, with language serving as a generic interface to empower this. Based on this philosophy, we present HuggingGPT, an LLM-powered agent that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Hugging Face, HuggingGPT can tackle a wide range of sophisticated AI tasks spanning different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards the realization of artificial general intelligence. (@hugginggpt)

Significant-Gravitas Autogpt 2023. <https://github.com/Significant-Gravitas/Auto-GPT>. **Abstract:** In this pioneering study, inspired by AutoGPT, the state-of-the-art open-source application based on the GPT-4 large language model, we develop a novel tool called AD-AutoGPT which can conduct data collection, processing, and analysis about complex health narratives of Alzheimer’s Disease in an autonomous manner via users’ textual prompts. We collated comprehensive data from a variety of news sources, including the Alzheimer’s Association, BBC, Mayo Clinic, and the National Institute on Aging since June 2022, leading to the autonomous execution of robust trend analyses, intertopic distance maps visualization, and identification of salient terms pertinent to Alzheimer’s Disease. This approach has yielded not only a quantifiable metric of relevant discourse but also valuable insights into public focus on Alzheimer’s Disease. This application of AD-AutoGPT in public health signifies the transformative potential of AI in facilitating a data-rich understanding of complex health narratives like Alzheimer’s Disease in an autonomous manner, setting the groundwork for future AI-driven investigations in global health landscapes. (@autogpt)

Peter HA Sneath, Robert R Sokal, et al *Numerical taxonomy. The principles and practice of numerical classification.* . (@HierarchicalClustering)

Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, William Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, and Yue Zhao Trustllm: Trustworthiness in large language models 2024. **Abstract:** Large language models (LLMs), exemplified by ChatGPT, have gained considerable attention for their excellent natural language processing capabilities. Nonetheless, these LLMs present many challenges, particularly in the realm of trustworthiness. Therefore, ensuring the trustworthiness of LLMs emerges as an important topic. This paper introduces TrustLLM, a comprehensive study of trustworthiness in LLMs, including principles for different dimensions of trustworthiness, established benchmark, evaluation, and analysis of trustworthiness for mainstream LLMs, and discussion of open challenges and future directions. Specifically, we first propose a set of principles for trustworthy LLMs that span eight different dimensions. Based on these principles, we further establish a benchmark across six dimensions including truthfulness, safety, fairness, robustness, privacy, and machine ethics. We then present a study evaluating 16 mainstream LLMs in TrustLLM, consisting of over 30 datasets. Our findings firstly show that in general trustworthiness and utility (i.e., functional effectiveness) are positively related. Secondly, our observations reveal that proprietary LLMs generally outperform most open-source counterparts in terms of trustworthiness, raising concerns about the potential risks of widely accessible open-source LLMs. However, a few open-source LLMs come very close to proprietary ones. Thirdly, it is important to note that some LLMs may be overly calibrated towards exhibiting trustworthiness, to the extent that they compromise their utility by mistakenly treating benign prompts as harmful and consequently not responding. Finally, we emphasize the importance of ensuring transparency not only in the models themselves but also in the technologies that underpin trustworthiness. Knowing the specific trustworthy technologies that have been employed is crucial for analyzing their effectiveness. (@sun2024trustllm)

Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren Is chatgpt good at search? investigating large language models as re-ranking agent *arXiv preprint arXiv:2304.09542*, 2023. **Abstract:** Large Language Models (LLMs) have demonstrated remarkable zero-shot generalization across various language-related tasks, including search engines. However, existing work utilizes the generative ability of LLMs for Information Retrieval (IR) rather than direct passage ranking. The discrepancy between the pre-training objectives of LLMs and the ranking objective poses another challenge. In this paper, we first investigate generative LLMs such as ChatGPT and GPT-4 for relevance ranking in IR. Surprisingly, our experiments reveal that properly instructed LLMs can deliver competitive, even superior results to state-of-the-art supervised methods on popular IR benchmarks. Furthermore, to address concerns about data contamination of LLMs, we collect a new test set called NovelEval, based on the latest knowledge and aiming to verify the model’s ability to rank unknown knowledge. Finally, to improve efficiency in real-world applications, we delve into the potential for distilling the ranking capabilities of ChatGPT into small specialized models using a permutation distillation scheme. Our evaluation results turn out that a distilled 440M model outperforms a 3B supervised model on the BEIR benchmark. The code to reproduce our results is available at www.github.com/sunnweiwei/RankGPT. (@retrieval)

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant ommonsenseQA: A question answering challenge targeting commonsense knowledge In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. . URL <https://aclanthology.org/N19-1421>. **Abstract:** When answering a question, people often draw upon their rich world knowledge in addition to the particular context. Recent work has focused primarily on answering questions given some relevant document or context, and required very little general background. To investigate question answering with prior knowledge, we present CommonsenseQA: a challenging new dataset for commonsense question answering. To capture common sense beyond associations, we extract from ConceptNet (Speer et al., 2017) multiple target concepts that have the same semantic relation to a single source concept. Crowd-workers are asked to author multiple-choice questions that mention the source concept and discriminate in turn between each of the target concepts. This encourages workers to create questions with complex semantics that often require prior knowledge. We create 12,247 questions through this procedure and demonstrate the difficulty of our task with a large number of strong baselines. Our best baseline is based on BERT-large (Devlin et al., 2018) and obtains 56% accuracy, well below human performance, which is 89%. (@commonsenseqa)

Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and Le Sun Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. *arXiv preprint arXiv:2306.05301*, 2023. **Abstract:** Enabling large language models to utilize real-world tools effectively is crucial for achieving embodied intelligence. Existing approaches to tool learning have either primarily relied on extremely large language models, such as GPT-4, to attain generalized tool-use abilities in a zero-shot manner, or utilized supervised learning to train limited scopes of tools on compact models. However, it remains uncertain whether smaller language models can achieve generalized tool-use abilities without tool-specific training. To address this question, this paper introduces ToolAlpaca, a novel framework designed to automatically generate a diverse tool-use corpus and learn generalized tool-use abilities on compact language models with minimal human intervention. Specifically, ToolAlpaca first automatically creates a highly diversified tool-use corpus by building a multi-agent simulation environment. The corpus contains 3938 tool-use instances from more than 400 real-world tool APIs spanning 50 distinct categories. Subsequently, the constructed corpus is employed to fine-tune compact language models, resulting in two models, namely ToolAlpaca-7B and ToolAlpaca-13B, respectively. Finally, we evaluate the ability of these models to utilize previously unseen tools without specific training. Experimental results demonstrate that ToolAlpaca achieves effective generalized tool-use capabilities comparable to those of extremely large language models like GPT-3.5, demonstrating that learning generalized tool-use ability is feasible for compact language models. (@toolalpaca)

ODSC Team Have you met fingpt? a new open-source financial large language model 2023. <https://opendatascience.com/have-you-met-fingpt-a-new-open-source-financial-large-language-model/>. **Abstract:** Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data. In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and low-code development. Through collaborative efforts within the open-source AI4Finance community, FinGPT aims to stimulate innovation, democratize FinLLMs, and unlock new opportunities in open finance. Two associated code repos are \\}url{https://github.com/AI4Finance-Foundation/FinGPT} and \\}url{https://github.com/AI4Finance-Foundation/FinNLP} (@finance5)

THUDM Chatglm2 2023. <https://github.com/THUDM/ChatGLM2-6B>. **Abstract:** Over the years, the field of landslide disaster research has amassed a wealth of data and specialized knowledge. However, these resources originate from a wide array of sources and often feature complex data structures, highlighting a persistent lack of methods to integrate multi-source, heterogeneous data. Traditional landslide monitoring methods typically focus on singular monitoring targets and data sources, which limits a comprehensive understanding of the complex processes involved in landslides. This paper introduces a landslide monitoring model based on a knowledge graph. This model employs P-Tuning to fine-tune ChatGLM2 for the extraction of triples. Differential InSAR (D-InSAR) is utilized to extract ground deformation data, which is then integrated with the knowledge graph for landslide monitoring and analysis. This study focuses on the co-seismic landslide in Jishishan, Gansu, China. By analyzing the landslide knowledge graph and the spatiotemporal deformation map, the results are as follows: (1) For this event, 106 entities and attributes were constructed, along with two recommended calculation routes. (2) The deformation at the earthquake’s central region reached up to 8.784 cm, with a slightly smaller deformation zone to the northwest peaking at 9.662 cm. Significant unilateral subsidence was observed in the mountain range to the southwest. (3) The area affected by the co-seismic landslide primarily includes farmland and villages, covering an area of 0.3408 square kilometers. (4) Analysis based on the knowledge graph indicates that this landslide was primarily caused by the rapid liquefaction of water-saturated soil layers due to the earthquake, resulting in instability. This study contributes to the analysis of post-disaster losses, attribution, and impacts. (@8kchatglm2)

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al Llama 2: Open foundation and fine-tuned chat models *arXiv preprint arXiv:2307.09288*, 2023. **Abstract:** In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs. (@llama2)

Laurens Van der Maaten and Geoffrey Hinton Visualizing data using t-sne *Journal of machine learning research*, 9 (11), 2008. **Abstract:** We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large datasets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of datasets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the datasets. (@tsne)

Nicolas Vidal How ai and llms are streamlining financial services 2023. <https://www.forbes.com/sites/forbestechcouncil/2023/05/05/how-ai-and-llms-are-streamlining-financial-services/?sh=2ea8b923017a>. **Abstract:** This article explores the transformative role of Large Language Models (LLMs) in enterprise data engineering, focusing on their capacity to automate ETL processes, optimize queries, and streamline compliance reporting. The article examines how LLMs possess sophisticated capabilities for understanding data structures, generating code, transferring knowledge across platforms, and applying probabilistic reasoning for data quality. It delves into technical implementations of LLM-powered ETL automation, including script generation, schema evolution handling, and integration with modern data stacks. The article further investigates how these models optimize SQL queries and create natural language interfaces, making data more accessible to non-technical users. Through industry case studies in financial services, healthcare, retail, and manufacturing, the article demonstrates how LLMs are delivering substantial improvements in operational efficiency, data utilization, and business outcomes, representing a fundamental shift in how organizations perceive data engineering challenges. It also acknowledges the limitations of current LLM applications in data engineering and suggests directions for future research, including addressing ethical considerations such as potential biases and the need for explainable AI. (@finance3)

Leandro von Werra and Loubna Ben Allal Starcoder: A state-of-the-art llm for code 2023. <https://huggingface.co/blog/starcoder>. **Abstract:** Using an isolated Milky Way-mass galaxy simulation, we compare results from nine state-of-the-art gravito-hydrodynamics codes widely used in the numerical community. We utilize the infrastructure we have built for the AGORA High-resolution Galaxy Simulations Comparison Project. This includes the common disk initial conditions, common physics models (e.g., radiative cooling and UV background by the standardized package Grackle) and common analysis toolkit yt, all of which are publicly available. Subgrid physics models such as Jeans pressure floor, star formation, supernova feedback energy, and metal production are carefully constrained across code platforms. With numerical accuracy that resolves the disk scale height, we find that the codes overall agree well with one another in many dimensions including: gas and stellar surface densities, rotation curves, velocity dispersions, density and temperature distribution functions, disk vertical heights, stellar clumps, star formation rates, and Kennicutt–Schmidt relations. Quantities such as velocity dispersions are very robust (agreement within a few tens of percent at all radii) while measures like newly formed stellar clump mass functions show more significant variation (difference by up to a factor of ∼3). Systematic differences exist, for example, between mesh-based and particle-based codes in the low-density region, and between more diffusive and less diffusive schemes in the high-density tail of the density distribution. Yet intrinsic code differences are generally small compared to the variations in numerical implementations of the common subgrid physics such as supernova feedback. Our experiment reassures that, if adequately designed in accordance with our proposed common parameters, results of a modern high-resolution galaxy formation simulation are more sensitive to input physics than to intrinsic differences in numerical schemes. (@software1)

Guangyu Wang, Guoxing Yang, Zongxin Du, Longjun Fan, and Xiaohu Li Clinicalgpt: Large language models finetuned with diverse medical data and comprehensive evaluation *ArXiv*, abs/2306.09968, 2023. URL <https://api.semanticscholar.org/CorpusID:259187929>. **Abstract:** Large language models have exhibited exceptional performance on various Natural Language Processing (NLP) tasks, leveraging techniques such as the pre-training, and instruction fine-tuning. Despite these advances, their effectiveness in medical applications is limited, due to challenges such as factual inaccuracies, reasoning abilities, and lack grounding in real-world experience. In this study, we present ClinicalGPT, a language model explicitly designed and optimized for clinical scenarios. By incorporating extensive and diverse real-world data, such as medical records, domain-specific knowledge, and multi-round dialogue consultations in the training process, ClinicalGPT is better prepared to handle multiple clinical task. Furthermore, we introduce a comprehensive evaluation framework that includes medical knowledge question-answering, medical exams, patient consultations, and diagnostic analysis of medical records. Our results demonstrate that ClinicalGPT significantly outperforms other models in these tasks, highlighting the effectiveness of our approach in adapting large language models to the critical domain of healthcare. (@Wang2023ClinicalGPTLL)

Hao Wang, Chi-Liang Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, and Ting Liu Huatuo: Tuning llama model with chinese medical knowledge *ArXiv*, abs/2304.06975, 2023. URL <https://api.semanticscholar.org/CorpusID:258170497>. **Abstract:** Large Language Models (LLMs), such as the LLaMA model, have demonstrated their effectiveness in various general-domain natural language processing (NLP) tasks. Nevertheless, LLMs have not yet performed optimally in biomedical domain tasks due to the need for medical expertise in the responses. In response to this challenge, we propose HuaTuo, a LLaMA-based model that has been supervised-fine-tuned with generated QA (Question-Answer) instances. The experimental results demonstrate that HuaTuo generates responses that possess more reliable medical knowledge. Our proposed HuaTuo model is accessible at https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese. (@Wang2023HuaTuoTL)

Haochun Wang, Chi Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, and Ting Liu Huatuo: Tuning llama model with chinese medical knowledge *arXiv preprint arXiv:2304.06975*, 2023. **Abstract:** Large Language Models (LLMs), such as the LLaMA model, have demonstrated their effectiveness in various general-domain natural language processing (NLP) tasks. Nevertheless, LLMs have not yet performed optimally in biomedical domain tasks due to the need for medical expertise in the responses. In response to this challenge, we propose HuaTuo, a LLaMA-based model that has been supervised-fine-tuned with generated QA (Question-Answer) instances. The experimental results demonstrate that HuaTuo generates responses that possess more reliable medical knowledge. Our proposed HuaTuo model is accessible at https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese. (@huatuo)

Wen Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Y. Qiao, and Jifeng Dai Visionllm: Large language model is also an open-ended decoder for vision-centric tasks *ArXiv*, abs/2305.11175, 2023. URL <https://api.semanticscholar.org/CorpusID:258762579>. **Abstract:** Large language models (LLMs) have notably accelerated progress towards artificial general intelligence (AGI), with their impressive zero-shot capacity for user-tailored tasks, endowing them with immense potential across a range of applications. However, in the field of computer vision, despite the availability of numerous powerful vision foundation models (VFMs), they are still restricted to tasks in a pre-defined form, struggling to match the open-ended task capabilities of LLMs. In this work, we present an LLM-based framework for vision-centric tasks, termed VisionLLM. This framework provides a unified perspective for vision and language tasks by treating images as a foreign language and aligning vision-centric tasks with language tasks that can be flexibly defined and managed using language instructions. An LLM-based decoder can then make appropriate predictions based on these instructions for open-ended tasks. Extensive experiments show that the proposed VisionLLM can achieve different levels of task customization through language instructions, from fine-grained object-level to coarse-grained task-level customization, all with good results. It’s noteworthy that, with a generalist LLM-based framework, our model can achieve over 60\\}% mAP on COCO, on par with detection-specific models. We hope this model can set a new baseline for generalist vision and language models. The demo shall be released based on https://github.com/OpenGVLab/InternGPT. The code shall be released at https://github.com/OpenGVLab/VisionLLM. (@Wang2023VisionLLMLL)

Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, and Ji-Rong Wen Rethinking the evaluation for conversational recommendation in the era of large language models *arXiv preprint arXiv:2305.13112*, 2023. **Abstract:** The recent success of large language models (LLMs) has shown great potential to develop more powerful conversational recommender systems (CRSs), which rely on natural language conversations to satisfy user needs. In this paper, we embark on an investigation into the utilization of ChatGPT for conversational recommendation, revealing the inadequacy of the existing evaluation protocol. It might over-emphasize the matching with the ground-truth items or utterances generated by human annotators, while neglecting the interactive nature of being a capable CRS. To overcome the limitation, we further propose an interactive Evaluation approach based on LLMs named iEvaLM that harnesses LLM-based user simulators. Our evaluation approach can simulate various interaction scenarios between users and systems. Through the experiments on two publicly available CRS datasets, we demonstrate notable improvements compared to the prevailing evaluation protocol. Furthermore, we emphasize the evaluation of explainability, and ChatGPT showcases persuasive explanation generation for its recommendations. Our study contributes to a deeper comprehension of the untapped potential of LLMs for CRSs and provides a more flexible and easy-to-use evaluation framework for future research endeavors. The codes and data are publicly available at https://github.com/RUCAIBox/iEvaLM-CRS. (@recommend2)

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi Self-instruct: Aligning language model with self generated instructions 2022. **Abstract:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. (@selfinstruct)

Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V Le Simple synthetic data reduces sycophancy in large language models *arXiv preprint arXiv:2308.03958*, 2023. **Abstract:** Sycophancy is an undesirable behavior where models tailor their responses to follow a human user’s view even when that view is not objectively correct (e.g., adapting liberal views once a user reveals that they are liberal). In this paper, we study the prevalence of sycophancy in language models and propose a simple synthetic-data intervention to reduce this behavior. First, on a set of three sycophancy tasks (Perez et al., 2022) where models are asked for an opinion on statements with no correct answers (e.g., politics), we observe that both model scaling and instruction tuning significantly increase sycophancy for PaLM models up to 540B parameters. Second, we extend sycophancy evaluations to simple addition statements that are objectively incorrect, finding that despite knowing that these statements are wrong, language models will still agree with them if the user does as well. To reduce sycophancy, we present a straightforward synthetic-data intervention that takes public NLP tasks and encourages models to be robust to user opinions on these tasks. Adding these data in a lightweight finetuning step can significantly reduce sycophantic behavior on held-out prompts. Code for generating synthetic data for intervention can be found at https://github.com/google/sycophancy-intervention. (@sycophancy)

Matt Welsh The future of software development with llms is here: Announcing fixie’s developer preview and 17m in seed funding 2023. <https://blog.fixie.ai/the-future-of-software-development-with-llms-is-here-announcing-fixies-developer-preview-and-17m-cf6fca0c4041>. (@software3)

Stephen Wolfram Writings Chatgpt gets its “wolfram superpowers”! 2023. <https://writings.stephenwolfram.com/2023/03/chatgpt-gets-its-wolfram-superpowers/>. **Abstract:** This is part of an ongoing series about our LLM-related technology: ChatGPT Gets Its “Wolfram Superpowers”! Instant Plugins for ChatGPT: Introducing the Wolfram ChatGPT Plugin Kit The New World of LLM Functions: Integrating LLM Technology into the Wolfram Language Prompts for Work & Play: Launching the Wolfram Prompt Repository Introducing Chat Notebooks: Integrating LLMs into the Notebook Paradigm Since this was written, OpenAI has discontinued ChatGPT Plugins and launched custom GPTs. Find more information about the Wolfram GPT here: https://gpt.wolfram.com . Early in January I wrote about the possibility of connecting ChatGPT to Wolfram\|Alpha . And today—just two and a half months later—I’m excited to announce that it’s happened ! Thanks to some heroic software engineering by our team and by OpenAI , ChatGPT can now call on Wolfram\|Alpha—and Wolfram Language as well—to give it what we might think of as “computational superpowers”. It’s still very early days for all of this, but it’s already very impressive—and one can begin to see how amazingly powerful (and perhaps even revolutionary) what we can call “ ChatGPT + Wolfram ” can be. Back in January, I made the point that, as an LLM neural net , ChatGPT—for all its remarkable prowess in textually generating material “like” what it’s read from the web, etc.— can’t itself be expected to do actual nontrivial computations , or to systematically produce correct (rather than just “looks roughly right”) data, etc. But when it’s connected to the Wolfram plugin it can do these things. So here’s my (very simple) first example from January, but now done by ChatGPT with “Wolfram superpowers” installed: (@student4)

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann Bloomberggpt: A large language model for finance *arXiv preprint arXiv:2303.17564*, 2023. **Abstract:** The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no LLM specialized for the financial domain has been reported in literature. In this work, we present BloombergGPT, a 50 billion parameter language model that is trained on a wide range of financial data. We construct a 363 billion token dataset based on Bloomberg’s extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345 billion tokens from general purpose datasets. We validate BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most accurately reflect our intended usage. Our mixed dataset training leads to a model that outperforms existing models on financial tasks by significant margins without sacrificing performance on general LLM benchmarks. Additionally, we explain our modeling choices, training process, and evaluation methodology. We release Training Chronicles (Appendix C) detailing our experience in training BloombergGPT. (@financellms)

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann Bloomberggpt: A large language model for finance *arXiv preprint arXiv:2303.17564*, 2023. **Abstract:** The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no LLM specialized for the financial domain has been reported in literature. In this work, we present BloombergGPT, a 50 billion parameter language model that is trained on a wide range of financial data. We construct a 363 billion token dataset based on Bloomberg’s extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345 billion tokens from general purpose datasets. We validate BloombergGPT on standard LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most accurately reflect our intended usage. Our mixed dataset training leads to a model that outperforms existing models on financial tasks by significant margins without sacrificing performance on general LLM benchmarks. Additionally, we explain our modeling choices, training process, and evaluation methodology. We release Training Chronicles (Appendix C) detailing our experience in training BloombergGPT. (@wu2023bloomberggpt)

Daniel Xiao Ai-based literature review tools 2023. <https://tamu.libguides.com/c.php?g=1289555>. **Abstract:** In the context of rapid advancements in artificial intelligence (AI), its pervasive impact across industries extends to Human Resource Management (HRM), redefining workforce management paradigms. This systematic literature review addresses the intersection of AI and HRM, elucidating AI-driven tools and techniques that optimize recruitment, performance management, and employee engagement. Rooted in the recognition that conventional HRM approaches can be time-intensive and biased, AI’s integration promises to enhance decision-making efficiency and accuracy. This review, guided by research questions, explores prevalent AI applications, evaluates statistical evidence of their impact, and engenders discussions on implications. Through methodological rigor, this review critically synthesizes diverse scholarly perspectives. It provides insights into AI’s multifaceted role, en-compassing candidate screening, personalized training, real-time performance assessment, and engagement enhancement. Next, it furnishes statistically derived trends from selected studies, revealing adoption rates, prevalent techniques, and sector-specific implementations. (@student1)

Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang On the tool manipulation capability of open-source large language models *arXiv preprint arXiv:2305.16504*, 2023. **Abstract:** Recent studies on software tool manipulation with large language models (LLMs) mostly rely on closed model APIs. The industrial adoption of these models is substantially constrained due to the security and robustness risks in exposing information to closed LLM API services. In this paper, we ask can we enhance open-source LLMs to be competitive to leading closed LLM APIs in tool manipulation, with practical amount of human supervision. By analyzing common tool manipulation failures, we first demonstrate that open-source LLMs may require training with usage examples, in-context demonstration and generation style regulation to resolve failures. These insights motivate us to revisit classical methods in LLM literature, and demonstrate that we can adapt them as model alignment with programmatic data generation, system prompts and in-context demonstration retrievers to enhance open-source LLMs for tool manipulation. To evaluate these techniques, we create the ToolBench, a tool manipulation benchmark consisting of diverse software tools for real-world tasks. We demonstrate that our techniques can boost leading open-source LLMs by up to 90% success rate, showing capabilities competitive to OpenAI GPT-4 in 4 out of 8 ToolBench tasks. We show that such enhancement typically requires about one developer day to curate data for each tool, rendering a recipe with practical amount of human supervision. (@toolbench)

Zhiling Yan, Kai Zhang, Rong Zhou, Lifang He, Xiang Li, and Lichao Sun Multimodal chatgpt for medical applications: an experimental study of gpt-4v *arXiv preprint arXiv:2310.19061*, 2023. **Abstract:** In this paper, we critically evaluate the capabilities of the state-of-the-art multimodal large language model, i.e., GPT-4 with Vision (GPT-4V), on Visual Question Answering (VQA) task. Our experiments thoroughly assess GPT-4V’s proficiency in answering questions paired with images using both pathology and radiology datasets from 11 modalities (e.g. Microscopy, Dermoscopy, X-ray, CT, etc.) and fifteen objects of interests (brain, liver, lung, etc.). Our datasets encompass a comprehensive range of medical inquiries, including sixteen distinct question types. Throughout our evaluations, we devised textual prompts for GPT-4V, directing it to synergize visual and textual information. The experiments with accuracy score conclude that the current version of GPT-4V is not recommended for real-world diagnostics due to its unreliable and suboptimal accuracy in responding to diagnostic medical questions. In addition, we delineate seven unique facets of GPT-4V’s behavior in medical VQA, highlighting its constraints within this complex arena. The complete details of our evaluation cases are accessible at https://github.com/ZhilingYan/GPT4V-Medical-Report. (@yan2023multimodal)

Hongyang Yang, Xiao-Yang Liu, and Chris Wang Fingpt: Open-source financial large language models *ArXiv*, abs/2306.06031, 2023. URL <https://api.semanticscholar.org/CorpusID:259129734>. **Abstract:** Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data. In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and low-code development. Through collaborative efforts within the open-source AI4Finance community, FinGPT aims to stimulate innovation, democratize FinLLMs, and unlock new opportunities in open finance. Two associated code repos are \\}url{https://github.com/AI4Finance-Foundation/FinGPT} and \\}url{https://github.com/AI4Finance-Foundation/FinNLP} (@Yang2023FinGPTOF)

Hongyang Yang, Xiao-Yang Liu, and Christina Dan Wang Fingpt: Open-source financial large language models *arXiv preprint arXiv:2306.06031*, 2023. **Abstract:** Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data. In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and low-code development. Through collaborative efforts within the open-source AI4Finance community, FinGPT aims to stimulate innovation, democratize FinLLMs, and unlock new opportunities in open finance. Two associated code repos are \\}url{https://github.com/AI4Finance-Foundation/FinGPT} and \\}url{https://github.com/AI4Finance-Foundation/FinNLP} (@finance4)

Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan Gpt4tools: Teaching large language model to use tools via self-instruction *arXiv preprint arXiv:2305.18752*, 2023. **Abstract:** This paper aims to efficiently enable Large Language Models (LLMs) to use multimodal tools. Advanced proprietary LLMs, such as ChatGPT and GPT-4, have shown great potential for tool usage through sophisticated prompt engineering. Nevertheless, these models typically rely on prohibitive computational costs and publicly inaccessible data. To address these challenges, we propose the GPT4Tools based on self-instruct to enable open-source LLMs, such as LLaMA and OPT, to use tools. It generates an instruction-following dataset by prompting an advanced teacher with various multi-modal contexts. By using the Low-Rank Adaptation (LoRA) optimization, our approach facilitates the open-source LLMs to solve a range of visual problems, including visual comprehension and image generation. Moreover, we provide a benchmark to evaluate the ability of LLMs to use tools, which is performed in both zero-shot and fine-tuning ways. Extensive experiments demonstrate the effectiveness of our method on various language models, which not only significantly improves the accuracy of invoking seen tools, but also enables the zero-shot capacity for unseen tools. The code and demo are available at https://github.com/StevenGrove/GPT4Tools. (@gpt4tools)

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao React: Synergizing reasoning and acting in language models *arXiv preprint arXiv:2210.03629*, 2022. **Abstract:** While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Project site with code: https://react-lm.github.io (@react)

Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, and Songfang Huang How well do large language models perform in arithmetic tasks? *arXiv preprint arXiv:2304.02015*, 2023. URL <https://arxiv.org/abs/2304.02015>. **Abstract:** Large language models have emerged abilities including chain-of-thought to answer math word problems step by step. Solving math word problems not only requires abilities to disassemble problems via chain-of-thought but also needs to calculate arithmetic expressions correctly for each step. To the best of our knowledge, there is no work to focus on evaluating the arithmetic ability of large language models. In this work, we propose an arithmetic dataset MATH 401 to test the latest large language models including GPT-4, ChatGPT, InstrctGPT, Galactica, and LLaMA with various arithmetic expressions and provide a detailed analysis of the ability of large language models. MATH 401 and evaluation codes are released at \\}url{https://github.com/GanjinZero/math401-llm}. (@arithmetic)

Zhengqing Yuan, Zhaoxu Li, and Lichao Sun Tinygpt-v: Efficient multimodal large language model via small backbones *arXiv preprint arXiv:2312.16862*, 2023. **Abstract:** In recent years, multimodal large language models (MLLMs) such as GPT-4V have demonstrated remarkable advancements, excelling in a variety of vision-language tasks. Despite their prowess, the closed-source nature and computational demands of such models limit their accessibility and applicability. This study introduces TinyGPT-V, a novel open-source MLLM, designed for efficient training and inference across various vision-language tasks, including image captioning (IC) and visual question answering (VQA). Leveraging a compact yet powerful architecture, TinyGPT-V integrates the Phi-2 language model with pre-trained vision encoders, utilizing a unique mapping module for visual and linguistic information fusion. With a training regimen optimized for small backbones and employing a diverse dataset amalgam, TinyGPT-V requires significantly lower computational resources 24GB for training and as little as 8GB for inference without compromising on performance. Our experiments demonstrate that TinyGPT-V, with its language model 2.8 billion parameters, achieves comparable results in VQA and image inference tasks to its larger counterparts while being uniquely suited for deployment on resource-constrained devices through innovative quantization techniques. This work not only paves the way for more accessible and efficient MLLMs but also underscores the potential of smaller, optimized models in bridging the gap between high performance and computational efficiency in real-world applications. Additionally, this paper introduces a new approach to multimodal large language models using smaller backbones. Our code and training weights are available in the supplementary material. (@yuan2023tinygpt)

Jennifer Zhang Unleashing the potential of llms: a new era for financial services 2023. <https://www.wiz.ai/unleashing-the-potential-of-llms-a-new-era-for-financial-services/>. **Abstract:** Unleash the potential of Large Language Models (LLMs) in financial services, ushering in a new era of enhanced decision-making, automation, and customer experience. (@finance2)

Kai Zhang, Jun Yu, Zhiling Yan, Yixin Liu, Eashan Adhikarla, Sunyang Fu, Xun Chen, Chen Chen, Yuyin Zhou, Xiang Li, et al Biomedgpt: A unified and generalist biomedical generative pre-trained transformer for vision, language, and multimodal tasks *arXiv preprint arXiv:2305.17100*, 2023. **Abstract:** Traditional biomedical artificial intelligence (AI) models, designed for specific tasks or modalities, often exhibit limited flexibility in real-world deployment and struggle to utilize holistic information. Generalist AI holds the potential to address these limitations due to its versatility in interpreting different data types and generating tailored outputs for diverse needs. However, existing biomedical generalist AI solutions are typically heavyweight and closed source to researchers, practitioners, and patients. Here, we propose BiomedGPT, the first open-source and lightweight vision-language foundation model, designed as a generalist capable of performing various biomedical tasks. BiomedGPT achieved state-of-the-art results in 16 out of 25 experiments while maintaining a computing-friendly model scale. We also conducted human evaluations to assess the capabilities of BiomedGPT in radiology visual question answering, report generation, and summarization. BiomedGPT exhibits robust prediction ability with a low error rate of 3.8% in question answering, satisfactory performance with an error rate of 8.3% in writing complex radiology reports, and competitive summarization ability with a nearly equivalent preference score to human experts. Our method demonstrates that effective training with diverse data can lead to more practical biomedical AI for improving diagnosis and workflow efficiency. (@zhang2023biomedgpt)

Ce Zhou, Qian Li, Chen Li, Jun Yu, Yixin Liu, Guangjing Wang, Kai Zhang, Cheng Ji, Qiben Yan, Lifang He, et al A comprehensive survey on pretrained foundation models: A history from bert to chatgpt *arXiv preprint arXiv:2302.09419*, 2023. **Abstract:** Pretrained Foundation Models (PFMs) are regarded as the foundation for various downstream tasks with different data modalities. A PFM (e.g., BERT, ChatGPT, and GPT-4) is trained on large-scale data which provides a reasonable parameter initialization for a wide range of downstream applications. BERT learns bidirectional encoder representations from Transformers, which are trained on large datasets as contextual language models. Similarly, the generative pretrained transformer (GPT) method employs Transformers as the feature extractor and is trained using an autoregressive paradigm on large datasets. Recently, ChatGPT shows promising success on large language models, which applies an autoregressive language model with zero shot or few shot prompting. The remarkable achievements of PFM have brought significant breakthroughs to various fields of AI. Numerous studies have proposed different methods, raising the demand for an updated survey. This study provides a comprehensive review of recent research advancements, challenges, and opportunities for PFMs in text, image, graph, as well as other data modalities. The review covers the basic components and existing pretraining methods used in natural language processing, computer vision, and graph learning. Additionally, it explores advanced PFMs used for different data modalities and unified PFMs that consider data quality and quantity. The review also discusses research related to the fundamentals of PFMs, such as model efficiency and compression, security, and privacy. Finally, the study provides key implications, future research directions, challenges, and open problems in the field of PFMs. Overall, this survey aims to shed light on the research of the PFMs on scalability, security, logical reasoning ability, cross-domain learning ability, and the user-friendly interactive ability for artificial general intelligence. (@zhou2023comprehensive)

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al Lima: Less is more for alignment *arXiv preprint arXiv:2305.11206*, 2023. **Abstract:** Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling. LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history. Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data. In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003, which was trained with human feedback. Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output. (@lima)

Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang Toolqa: A dataset for llm question answering with external tools 2023. **Abstract:** Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks, but they still suffer from challenges such as hallucination and weak numerical reasoning. To overcome these challenges, external tools can be used to enhance LLMs’ question-answering abilities. However, current evaluation methods do not distinguish between questions that can be answered using LLMs’ internal knowledge and those that require external information through tool use. To address this issue, we introduce a new dataset called ToolQA, which is designed to faithfully evaluate LLMs’ ability to use external tools for question answering. Our development of ToolQA involved a scalable, automated process for dataset curation, along with 13 specialized tools designed for interaction with external knowledge in order to answer questions. Importantly, we strive to minimize the overlap between our benchmark data and LLMs’ pre-training data, enabling a more precise evaluation of LLMs’ tool-use reasoning abilities. We conducted an in-depth diagnosis of existing tool-use LLMs to highlight their strengths, weaknesses, and potential improvements. Our findings set a new benchmark for evaluating LLMs and suggest new directions for future advancements. Our data and code are freely available to the broader scientific community on GitHub. (@ToolQA)

</div>

# Appendix

### <span class="smallcaps">ToolE</span> Dataset Details

In this section, we show the details of <span class="smallcaps">ToolE</span>, including how we solve the overlapped issue (Section <a href="#app:details_overlapped_issue" data-reference-type="ref" data-reference="app:details_overlapped_issue">1.1</a>), guidelines for human validation (Section <a href="#app:human_guideline" data-reference-type="ref" data-reference="app:human_guideline">1.3</a>), and the statistics of <span class="smallcaps">ToolE</span> (Section <a href="#app:other_details_toole" data-reference-type="ref" data-reference="app:other_details_toole">1.5</a>).

<figure id="fig:merge_decompose">
<img src="./figures/merge_decompose.png"" style="width:50.0%" />
<figcaption>Two examples of tool mergence and decomposition.</figcaption>
</figure>

#### Details of Overlapped Issue

**Operation pipeline.**

\(1\) *Embeddings and Hierarchical Clustering*. We initiated by generating embeddings for tool descriptions using the text-embedding-ada-002 model  , an API provided by OpenAI, aiming to perform hierarchical clustering   on different tools, based on the similarity of their embeddings, to reveal underlying patterns among them. (2) *Tool Merging and Decomposition.* Based on the results of clustering, we manually merged and decomposed the data. Specifically, several popular topics (e.g., news, weather) were identified based on their functions if they overlapped functionality with other tools (an example is shown in Figure <a href="#fig:merge_decompose" data-reference-type="ref" data-reference="fig:merge_decompose">8</a>). The criteria for merging and decomposition revolved around whether such tools are commonly encountered and make practical sense in daily life. For instance, it is logical to merge a tool that offers both flight and train ticket bookings with another tool that solely focuses on hotel reservations. For merged tools, we only needed to modify the original labels; for decomposed tools, we manually assigned original queries to the appropriate decomposed tools and changed their labels accordingly. We manually created the names of new tools and employed ChatGPT to generate descriptions for both merged and decomposed tools, based on the initial descriptions of these tools. (3) *Similarity Verification and Human Validation.* We iterated each tool and searched for the ten most similar tools by its description embedding, then we checked whether the tool could be further merged or decomposed with the tools in $`L_{sim}`$.

<figure id="fig:cluster_example">

<figcaption>t-SNE  <span class="citation" data-cites="tsne"></span> visualization of original tool description embedding (a) and new tool description embedding (b).</figcaption>
</figure>

<figure id="fig:cluster_score">
<img src="./figures/cluster_score.png"" style="width:80.0%" />
<figcaption>Silhouette score  <span class="citation" data-cites="silhouettes"></span> of new tool description embedding and original tool description embedding in different cluster numbers.</figcaption>
</figure>

#### Efficiency of the Operation

To evaluate the effectiveness of our operations in solving overlapped issues, we use the silhouette coefficient   to measure the degree of functional overlap between tools. Based on it, we compare the changes in the silhouette coefficients before and after the operations. Specifically, we aim to significantly reduce the functional overlap between tools after merging and decomposition to achieve a more uniform distribution of tool functionalities in the embedding space. We embed the tool description before (390 tools) and after the operations (195 tools) and compared the changes in the silhouette coefficients under the same number of clusters. A greater silhouette coefficient indicates better clustering performance and higher functional overlap between tools, while a smaller coefficient suggests lower overlap, reflecting the effectiveness of the operations.

In Figure <a href="#fig:cluster_score" data-reference-type="ref" data-reference="fig:cluster_score">10</a>, we present the variation of silhouette coefficients with changes in the number of clusters. It can be observed that the silhouette coefficients of tools after the operations are significantly smaller than before, indicating that the operations have made the distribution of tools more uniform and effectively reduced tool functionality overlap. Additionally, in Figure <a href="#fig:cluster_example" data-reference-type="ref" data-reference="fig:cluster_example">9</a>, we visualize the results using t-SNE   when the number of clusters is 30. It is evident that the distribution on the left side is more uniform compared to the right side.

#### Guidelines for human validation

We conducted rigorous manual evaluations to ensure the integrity and quality of <span class="smallcaps">ToolE</span>. We established the following rules to guide the manual evaluation:

- Low-quality tool descriptions. In some cases, ChatGPT was unable to understand the purpose of a tool due to low-quality or overly brief tool descriptions. We conducted a manual review of these descriptions and eliminated tools with unclear or low-quality explanations.

- High repetition queries. Since we generated multiple queries for a single tool in one batch, some batches had issues with high query repetition. To address this problem, we selected one query and removed the others.

- Queries contain tool name. The inclusion of a tool’s name in a query can significantly bias our evaluation as an obvious hint for all tasks in <span class="smallcaps">MetaTool</span>. Therefore, we removed queries containing the tool’s name. For example, ’How can I calculate my MBTI type through \[tool name\]?’

- Calculation-related tools. <span class="smallcaps">ToolE</span> contained numerous tools related to calculations. For simple calculations (e.g., ‘What is the value of sin 30 degrees?’ or ‘7 \* 9 = ?’), LLMs can perform them without the need for a tool. However, for complex calculations, recent research   suggests that LLMs still perform poorly. For queries corresponding to calculation-related tools, we removed queries involving simple calculations and retained those involving complex calculations.

- Tool retrieval-related tools. We found that some tools were designed for users to retrieve other tools. This kind of tool conflicted with our task, so we removed these tools.

- AI comprehensive tools. We identified some AI comprehensive tools that encompassed a wide range of AI-related tools, making them impractical for our evaluation. Therefore, we removed these tools.

- Mentions of ’ChatGPT’ in queries. Some queries included the term ’ChatGPT,’ for example, ’Hi, ChatGPT! ...’ We uniformly replaced ’ChatGPT’ with ’Chatbot’."

<figure id="fig:interface">
<img src="./figures/interface.jpg"" style="width:50.0%" />
<figcaption>The motivation labeling interface.</figcaption>
</figure>

#### Flexibility of <span class="smallcaps">ToolE</span> 

Sometimes, when an LLM undergoes specialized training in a particular domain  , its capabilities improve significantly, and in some cases, some LLMs are also capable of handling various types of information (e.g., images or audio) . These improvements render some external tools that may not be necessary for some LLMs in the future. Therefore, we have annotated the reasons why LLMs need to use these tools to solve user problems.

Specifically, we use four kinds of motivation for tool usage (some examples are shown in Table <a href="#tab:tool_reasons" data-reference-type="ref" data-reference="tab:tool_reasons">6</a>): A. Solving issues with real-time or external data, databases, or APIs. B. Handling specialized inputs/outputs. C. Enhancing domain tasks beyond LLM’s capabilities. D. User customization, personalization, and interaction. We have enlisted the expertise of two experts to annotate the usage motivations of tools in <span class="smallcaps">ToolE</span> (the annotation interface is shown in Figure <a href="#fig:interface" data-reference-type="ref" data-reference="fig:interface">11</a>).

<div id="tab:tool_reasons">

| Tool | Description | Reason |
|:---|:---|:---|
| airqualityforeast | Planning something outdoors? Get the 2-day air quality forecast for your city. | A |
| Now | Get Google Trends. In Japan, you can also get Twitter trends and search Twitter keywords. | A |
| abc_to_audio | Converts ABC music notation to WAV, MIDI, and PostScript files. | B |
| ChatOCR | The best way to read text from from any document. ChatOCR will scan and read aloud any text document you provide. | B |
| FinanceTool | Begin an exciting journey through time, interact with historical events, and learn about the past in a fun and engaging way. | C |
| LawTool | Enables quick search functionality for relevant laws. | C |
| TicTacToe | Playing a game of Tic Tac Toe with varying board sizes. You can submit your move and get the AI’s response move. | D |
| Planfit | Get your tailored workout plan and instructions with videos - AI-powered Workout Coach, Planfit. | D |

Possible reasons for the motivation of tool usage.

</div>

#### Others Statistics of <span class="smallcaps">ToolE</span>

<figure id="fig:length">
<img src="./figures/length_distribution.png"" style="width:45.0%" />
<figcaption>Density distribution of all queries’ length.</figcaption>
</figure>

**Data quantity.** Due to the constraints of API usage costs, we exclusively employ GPT-4 for direct diverse generation, utilizing ChatGPT for all other forms of generation techniques. This strategy yielded a total of 29,000 dataset entries. Following a meticulous human review process, we ultimately curated the <span class="smallcaps">ToolE</span> dataset, culminating in a collection comprising 20,881 entries. A comprehensive overview of dataset statistics can be found in Table <a href="#tab:data_stat" data-reference-type="ref" data-reference="tab:data_stat">11</a>.

**Length distribution.** Figure <a href="#fig:length" data-reference-type="ref" data-reference="fig:length">12</a> displays the distribution of dataset lengths. It can be observed that the majority of the data falls within 40 words or less, aligning with the typical question lengths in people’s daily lives.

**Visualization and release.** We used Nomic AI   to embed user queries in the <span class="smallcaps">ToolE</span> dataset, then clustered the embeddings, and finally visualized the results. The visualization is shown in Figure <a href="#fig:embedding_vis" data-reference-type="ref" data-reference="fig:embedding_vis">13</a>, and you can view it through the following link: <https://atlas.nomic.ai/map/a43a6a84-4453-428a-8738-2534d7bf0b89/b2b8134b-a37e-45d2-a0d9-765911f27df6>.

**Dataset comparison.** As shown in Table <a href="#tab:dataset_comparison" data-reference-type="ref" data-reference="tab:dataset_comparison">7</a>, we compare the other datasets with <span class="smallcaps">ToolE</span>. Compared to other datasets, we believe that ToolE has two main advantages: (1) Our dataset exhibits greater diversity, and this diversity is tailored to real user scenarios, such as variations in expression style, mood, and level of detail. We employ various prompt methods to induce LLMs to generate a more diverse range of user inputs, ensuring that <span class="smallcaps">ToolE</span> covers a broad spectrum of inputs resembling those of actual users. (2) By employing a pipeline process to address overlapped issues, we can ensure the rigor of the data. As outlined in Appendix <a href="#app:details_overlapped_issue" data-reference-type="ref" data-reference="app:details_overlapped_issue">1.1</a>, we employ multiple steps to address overlapped issues, ensuring that there is no overlap between tools, which is crucial for maintaining the quality of the dataset.

<div id="tab:dataset_comparison">

<table>
<caption>Comparison of previous work and <span class="smallcaps">MetaTool</span>.</caption>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;"><strong>Dimension</strong></td>
<td style="text-align: center;"><strong>APIBank</strong></td>
<td style="text-align: center;"><strong>ToolLLM</strong></td>
<td style="text-align: center;"><strong>ToolAlpaca</strong></td>
<td style="text-align: center;"><strong>GPT4Tool</strong></td>
<td style="text-align: center;"><strong>ToolQA</strong></td>
<td style="text-align: center;"><strong>ToolE</strong></td>
</tr>
<tr>
<td style="text-align: center;"><span class="citation" data-cites="apibank"></span></td>
<td style="text-align: center;"><span class="citation" data-cites="toolllm"></span></td>
<td style="text-align: center;"><span class="citation" data-cites="toolalpaca"></span></td>
<td style="text-align: center;"><span class="citation" data-cites="gpt4tools"></span></td>
<td style="text-align: center;"><span class="citation" data-cites="ToolQA"></span></td>
<td style="text-align: center;">(Ours)</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Multi-Tool</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Real Scenario</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Overlapped Issue Solved</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><strong>Diversity Generation</strong></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

</div>

<div id="tab:data_example">

| **Example** | **Type** |
|:---|:---|
| I can’t seem to remember anything I study. I need help with learning and retaining information effectively. | Emotion-angry |
| I’m feeling really down today, can you summarize this YouTube video for me? \[Link to YouTube video\] | Emotion-depressed |
| Good day! I’m hoping to find some amazing bargains and discounts today. Can you guide me in the right direction? | Emotion-happy |
| Hey Chatbot, I’m looking for a new pair of sneakers on GoFynd. Can you help me find the latest designs and recommend some popular brands? | Emotion-excited |
| I’m interested in Japanese cosmetics. Can you recommend some good brands? | Direct-request |
| Draw a state diagram for a vending machine. | Direct-order |
| Twitter Trends: What are the trending topics on Twitter in Japan? | Keyword |
| Could you recommend some popular shopping malls or markets in Singapore that offer a wide range of products, including local and international brands, diverse food options, and unique shopping experiences? | Details |

Data examples in <span class="smallcaps">ToolE</span>.

</div>

<div id="tab: top tool rank">

| **Tool Name** | **Merged Tools** | **Tool Name** | **Merged Tools** | **Tool Name** | **Merged Tools** |
|:---|:--:|:---|:--:|:---|:--:|
| FinanceTool | 22 | ResearchFinder | 7 | TripAdviceTool | 6 |
| ProductSearch | 19 | NewsTool | 7 | WeatherTool | 6 |
| JobTool | 12 | RepoTool | 6 | HousePurchasingTool | 5 |
| TripTool | 10 | ResearchHelper | 6 | Discount | 5 |
| PDF&URLTool | 8 | CourseTool | 6 | MusicTool | 5 |

Top 15 tools ranked by the number of merged tools.

</div>

<div id="tab: tool list of scenarios">

| **Scenario** | **Tools** | **Ref.** |
|:---|:---|:---|
| Software Engineer | RepoTool, AI2sql, SSH, AutoInfra1, noteable, dart, hackit_web_scanner, LarkBaseImporter, webhooks, universal |   |
| Elders | NewsTool, PolishTool, CharityTool, MapTool, MemoryTool, WeatherTool, RestaurantBookingTool, DietTool, NotesTool, TripAdviceTool |   |
| Finance Staff | FinanceTool, ChartTool, PDF&URLTool, NotesTool, ExchangeTool, CreditYelp, LawTool, DataRetrievalTool, fundsdbsearch, CompanyInfoTool |   |
| House manager | tira, Discount, ProductSearch, ABCmouse, RestaurantBookingTool, IndoorPlants, recipe_retrieval, HouseRentingTool, TripTool, CreditYelp |   |
| Students | CourseTool, ResearchFinder, ResearchHelper, speak, noteable, search, MemoryTool, NotesTool, MixerBox_Translate_AI_language_tutor, ABCmouse |   |
| Artists&designers | placid, find_agency, ArtCollection, ChartTool, storybird_stories, MediaModifyTool, PolishTool, MusicTool, ImageSearch, BookTool |   |

The tool lists of different scenarios.

</div>

**Tool description.** These original tool descriptions encompass two distinct categories: machine-readable descriptions and user-facing descriptions. The machine-readable descriptions prioritize considerations such as token context length or keyword incorporation, aiming to enhance tool prompting within an 8,000-character limit. Conversely, the user-facing descriptions offer succinct and simplified explanations of each tool’s functionality. While the majority of tools share identical descriptions across both categories, we opt to adopt the user-facing descriptions as the definitive tool descriptions. This choice is informed by the tendency of machine-readable descriptions to be overly verbose, often delving into instructing the language models on how to handle tool input and output – aspects that are not pertinent to our benchmark.

**Generation times.** For each type of original tool, we perform two rounds of direct diverse generation, producing ten queries each time. In the case of emotional generation, we generate five samples for each of the four distinct emotions. For keyword generation, we extract five keywords from the tool’s description and subsequently formulate a query for each identified keyword. Concerning details generation, the number of samples generated aligns with that of the direct diverse generation.

<figure id="fig:embedding_vis">
<img src="./figures/ToolE_embedding.png"" style="width:100.0%" />
<figcaption><span class="smallcaps">ToolE</span> embedding visualization.</figcaption>
</figure>

<div id="tab:data_stat">

<table>
<caption>Dataset Statistics of <span class="smallcaps">ToolE</span>.</caption>
<thead>
<tr>
<th style="text-align: center;"><strong>Generation method</strong></th>
<th style="text-align: center;"><strong>Model</strong></th>
<th style="text-align: center;"><strong>Sample number</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;">Direct generation</td>
<td style="text-align: center;">ChatGPT, GPT-4</td>
<td style="text-align: center;">11,700</td>
</tr>
<tr>
<td style="text-align: center;">Emotional generation</td>
<td style="text-align: center;">ChatGPT</td>
<td style="text-align: center;">7,800</td>
</tr>
<tr>
<td style="text-align: center;">Keyword generation</td>
<td style="text-align: center;">ChatGPT</td>
<td style="text-align: center;">1,950</td>
</tr>
<tr>
<td style="text-align: center;">Details generation</td>
<td style="text-align: center;">ChatGPT</td>
<td style="text-align: center;">7,800</td>
</tr>
<tr>
<td style="text-align: center;">Multi-tool generation</td>
<td style="text-align: center;">ChatGPT, GPT-4</td>
<td style="text-align: center;">1624</td>
</tr>
<tr>
<td colspan="3" style="text-align: center;">After checking <span class="math inline">→</span> 21,127 (20630 single-tool + 497 multi-tool)</td>
</tr>
</tbody>
</table>

</div>

### Test Set for Evaluation on the Awareness of Tool Usage

To assess to what extent LLMs are aware of their limitations, we construct the dataset for the awareness of tool usage by merging the positive samples from a subset of <span class="smallcaps">ToolE</span> dataset and the negative samples from some subsets of public datasets. To exclude the ambiguous situation where the queries in <span class="smallcaps">ToolE</span> can be solved either with or without the assistance of external tools, we manually check the output of the queries by feeding them directly into the LLMs and retain those whose responses are unsatisfactory(i.e. the model apologizes, the response contains errors, and etc). In this way, we verify that solving the queries in our selected subset(containing 515 samples) is beyond the capacity of the existing LLMs and the queries can therefore be treated as positive samples when evaluating the awareness of tool usage. Meanwhile, we collect negative samples of equal size from the public datasets of daily conversations and common sense , the queries of which are expected to be answered by LLMs without the help of external tools.

In order to gain a clearer understanding of the Tool Usage Awareness dataset, we will describe how we selected positive and negative samples for the dataset. Firstly, we categorized user queries into three types:

- The first type is "Queries must be solved by tools" (positive), such as multimodal input and real-time information retrieval.

- The second type is "Queries can be solved well by all LLMs" (negative), such as telling jokes, basic conversational functions, sentiment classification, and other basic NLP tasks.

- The third type represents the middle ground between the first and second types of user queries, i.e., queries we hope LLMs can solve but currently cannot, such as complex calculations, long text summarization, and information extraction.

With the aforementioned types of user queries, we classified user queries into positive or negative samples using human evaluation and model checking.

(1) Human evaluation: For the first and second types, we determined the samples through unanimous agreement from two human experts and referenced the four reasons in Appendix <a href="#sec:flexibility" data-reference-type="ref" data-reference="sec:flexibility">1.4</a> for selection.

(2) Model Checking: Regarding the third type of user queries, our objective is to find those within this category that can be solved well by all LLMs (classified as negative) and those that none of the LLMs can solve (classified as positive). We discarded user queries that only a portion of the LLMs can solve them. We conduct validation in the following two steps:

We initially input the queries into GPT-4. Since GPT-4 currently has the best performance in terms of utility, *if GPT-4 declines to answer (i.e., unable to solve the problem or refuse to answer), we classify it as a positive query.*

Following the above operation, if the query is not classified as positive, we conducted inference on eight LLMs and then evaluated the answers through two human experts. *If all output from LLMs solves the problem well, we classify them as negative queries.*

### Experimental Settings

#### Models and Test Samples

We conducted extensive experiments on eight commonly used LLMs including ChatGPT  , ChatGLM2 (6B)  , Llama2 (7b, 13b)  , Vicuna (7b, 13b, 33b)  , Baichuan2 (13b)  and Koala (13b)  . The temperature parameter for these models was uniformly set to $`0.0`$ in our experiments. For tool selection with similar choices and with possible reliability issues, we sampled five samples for each tool, resulting in a total of 975 samples. For tool selection in specific scenarios, we sampled 20 samples for each tool within each scenario. This means that for popularity-related scenarios, namely the top 5, 10, and 15 scenarios, we obtained 100, 200, and 300 samples, respectively. As for group-related scenarios, each scenario yielded 200 samples. In the case of"multi-tool selection, given the relatively small dataset size, we utilized the entire set of samples for testing.

#### Answer Matching

Due to the model’s practice of providing an explanatory context for its responses rather than directly outputting answers, it is necessary to perform answer matching (obtaining "yes," "no," or the name of a tool) for a more accurate evaluation.

**Tool usage awareness.** We use the following rules to match the results:

- If "yes" is present in the sentence and "no," "not," or "don’t" is absent from the sentence: The answer is "yes."

- If "no" is present in the sentence and "yes" is absent from the sentence: The answer is "yes." If phrases like "not seem necessary," "not think it is necessary," "not need to use," "not necessary to use," and "do not think I need to use" are present in the sentence: The answer is "no."

- If phrases like "I need to use," "I think it is necessary," "I may need to use," "I would need to use," "I believe it is necessary to use," "would need access," "might be necessary to use," "I might need to use," "tools would be necessary," "may be necessary to use," "be beneficial to use," "I will need to use," "might need to use," "would need to rely," and "may need to access" are present in the sentence: The answer is "yes."

For cases not covered by the above rules, the answers are analyzed manually.

**Tool selection.** We match the names of tools from the results that contain a tool list. For single-tool tasks, if no matches are found, it is recorded as "None." If one match is found, and "None" is not present in the results, it is compared with the ground-truth label. If more than two matches are found, manual analysis is performed. For multi-tool tasks, if the number of matches is less than two, it is considered an incorrect answer. If the number of matches is equal to two, it is compared with the ground-truth label. If more than two matches are found, the answers are manually evaluated.

#### Task Comparison

In order to have a more intuitive understanding of the different subtasks in tool selection, we show the comparison of the four sub-tasks in Table <a href="#tab:task_overview" data-reference-type="ref" data-reference="tab:task_overview">12</a>.

<div id="tab:task_overview">

| **Task** | **Tool list** | **Ideal output** |
|:--:|:---|:--:|
| Tool selection with similar choices | Tool $`t`$ and its most similar $`(n-1)`$ tools. | $`t`$ |
| Tool selection in specific scenarios | Specified tools in a certain scenario. | $`t`$ |
| Tool selection with possible reliability issues | The tools randomly chosen are from the remaining set of tools, which excludes tool $`t`$ and its $`k`$ most similar tools. | $`\varnothing`$ |
| Multi-tool Selection | The tools randomly chosen are from the remaining set of tools, which excludes the union set of $`k`$ most similar tools of $`t \in S_t`$, where $`S_t`$ is the ground-truth tool set. | $`S_t`$ |

Comparison of four tasks in tool selection.

</div>

#### Few-Shot Prompt

For Subtask1, we randomly sampled five different tools and, for each tool, randomly selected a corresponding query. The sampling method for Sub-task 2 was the same as Sub-task 1. For Subtask3, to balance different types of answers, we ensured that the ratio of answers being ’none’ to answers being tool was either 2:3 or 3:2. Due to the task setup of Sub-task 4, we did not conduct few-shot experiments on Sub-task 4. This is because we have ensured that the 15 tools most similar to the ground-truth tool were not present in the tool list, and the number of multi-tool queries was limited, making it impossible to guarantee that each query could have five exemplars containing two-tool combinations.

<table>
<caption>CSR (%) results of zero-shot and five-shot in different scenarios. <span class="math inline"><em>Δ</em></span> is the percentage change of CSR between zore-shot and five-shot, as calculated by CSR<span class="math inline"><sub><em>x</em> = 5</sub></span> <span class="math inline">−</span> CSR<span class="math inline"><sub><em>x</em> = 0</sub></span>. Abbreviation: Finance Staff (Finan.), House manager (Home.), Software Engineer (Soft.), Student (Stud.), and Artist &amp; Designer (Artis.).</caption>
<thead>
<tr>
<th colspan="2" style="text-align: center;"><strong>Scenario</strong></th>
<th style="text-align: center;"><strong>Finan.</strong></th>
<th style="text-align: center;"><strong>Home.</strong></th>
<th style="text-align: center;"><strong>Soft.</strong></th>
<th style="text-align: center;"><strong>Stud.</strong></th>
<th style="text-align: center;"><strong>Artis.</strong></th>
<th style="text-align: center;"><strong>Elders</strong></th>
<th style="text-align: center;"><strong>Top 5</strong></th>
<th style="text-align: center;"><strong>Top 10</strong></th>
<th style="text-align: center;"><strong>Top 15</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">46.70</td>
<td style="text-align: center;">57.59</td>
<td style="text-align: center;">28.50</td>
<td style="text-align: center;">40.31</td>
<td style="text-align: center;">42.78</td>
<td style="text-align: center;">68.42</td>
<td style="text-align: center;">82.29</td>
<td style="text-align: center;">56.19</td>
<td style="text-align: center;">48.07</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">62.57</td>
<td style="text-align: center;">62.94</td>
<td style="text-align: center;">41.53</td>
<td style="text-align: center;">34.97</td>
<td style="text-align: center;">45.64</td>
<td style="text-align: center;">69.66</td>
<td style="text-align: center;">86.17</td>
<td style="text-align: center;">57.22</td>
<td style="text-align: center;">50.54</td>
</tr>
<tr>
<td style="text-align: center;"><strong>ChatGLM2</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>15.87</strong></span></td>
<td style="text-align: center;"><span> <strong>5.35</strong></span></td>
<td style="text-align: center;"><span> <strong>13.03</strong></span></td>
<td style="text-align: center;"><span> <strong>-5.34</strong></span></td>
<td style="text-align: center;"><span> <strong>2.86</strong></span></td>
<td style="text-align: center;"><span> <strong>1.24</strong></span></td>
<td style="text-align: center;"><span> <strong>3.88</strong></span></td>
<td style="text-align: center;"><span> <strong>1.03</strong></span></td>
<td style="text-align: center;"><span> <strong>2.47</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">69.70</td>
<td style="text-align: center;">82.23</td>
<td style="text-align: center;">76.70</td>
<td style="text-align: center;">66.67</td>
<td style="text-align: center;">80.00</td>
<td style="text-align: center;">83.76</td>
<td style="text-align: center;">88.89</td>
<td style="text-align: center;">77.39</td>
<td style="text-align: center;">75.08</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">74.24</td>
<td style="text-align: center;">86.73</td>
<td style="text-align: center;">69.11</td>
<td style="text-align: center;">72.45</td>
<td style="text-align: center;">87.00</td>
<td style="text-align: center;">89.45</td>
<td style="text-align: center;">91.00</td>
<td style="text-align: center;">80.50</td>
<td style="text-align: center;">77.33</td>
</tr>
<tr>
<td style="text-align: center;"><strong>ChatGPT</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>4.54</strong></span></td>
<td style="text-align: center;"><span> <strong>4.50</strong></span></td>
<td style="text-align: center;"><span> <strong>-7.59</strong></span></td>
<td style="text-align: center;"><span> <strong>5.78</strong></span></td>
<td style="text-align: center;"><span> <strong>7.00</strong></span></td>
<td style="text-align: center;"><span> <strong>5.69</strong></span></td>
<td style="text-align: center;"><span> <strong>2.11</strong></span></td>
<td style="text-align: center;"><span> <strong>3.11</strong></span></td>
<td style="text-align: center;"><span> <strong>2.25</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">59.32</td>
<td style="text-align: center;">69.47</td>
<td style="text-align: center;">47.69</td>
<td style="text-align: center;">43.75</td>
<td style="text-align: center;">73.58</td>
<td style="text-align: center;">77.78</td>
<td style="text-align: center;">86.42</td>
<td style="text-align: center;">55.56</td>
<td style="text-align: center;">52.98</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">51.83</td>
<td style="text-align: center;">61.85</td>
<td style="text-align: center;">46.99</td>
<td style="text-align: center;">53.44</td>
<td style="text-align: center;">64.38</td>
<td style="text-align: center;">82.35</td>
<td style="text-align: center;">81.71</td>
<td style="text-align: center;">69.82</td>
<td style="text-align: center;">52.59</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Koala-13b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>-7.49</strong></span></td>
<td style="text-align: center;"><span> <strong>-7.62</strong></span></td>
<td style="text-align: center;"><span> <strong>-0.70</strong></span></td>
<td style="text-align: center;"><span> <strong>9.69</strong></span></td>
<td style="text-align: center;"><span> <strong>-9.20</strong></span></td>
<td style="text-align: center;"><span> <strong>4.57</strong></span></td>
<td style="text-align: center;"><span> <strong>-4.71</strong></span></td>
<td style="text-align: center;"><span> <strong>14.26</strong></span></td>
<td style="text-align: center;"><span> <strong>-0.39</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">52.04</td>
<td style="text-align: center;">57.07</td>
<td style="text-align: center;">30.00</td>
<td style="text-align: center;">32.99</td>
<td style="text-align: center;">59.09</td>
<td style="text-align: center;">57.14</td>
<td style="text-align: center;">65.66</td>
<td style="text-align: center;">45.73</td>
<td style="text-align: center;">44.86</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">64.97</td>
<td style="text-align: center;">62.12</td>
<td style="text-align: center;">42.35</td>
<td style="text-align: center;">40.00</td>
<td style="text-align: center;">69.00</td>
<td style="text-align: center;">76.17</td>
<td style="text-align: center;">67.68</td>
<td style="text-align: center;">47.24</td>
<td style="text-align: center;">54.08</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Llama2-7b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>12.93</strong></span></td>
<td style="text-align: center;"><span> <strong>5.05</strong></span></td>
<td style="text-align: center;"><span> <strong>12.35</strong></span></td>
<td style="text-align: center;"><span> <strong>7.01</strong></span></td>
<td style="text-align: center;"><span> <strong>9.91</strong></span></td>
<td style="text-align: center;"><span> <strong>19.03</strong></span></td>
<td style="text-align: center;"><span> <strong>2.02</strong></span></td>
<td style="text-align: center;"><span> <strong>1.51</strong></span></td>
<td style="text-align: center;"><span> <strong>9.22</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">35.18</td>
<td style="text-align: center;">47.00</td>
<td style="text-align: center;">34.85</td>
<td style="text-align: center;">35.00</td>
<td style="text-align: center;">30.00</td>
<td style="text-align: center;">41.54</td>
<td style="text-align: center;">42.00</td>
<td style="text-align: center;">43.72</td>
<td style="text-align: center;">38.93</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">55.50</td>
<td style="text-align: center;">65.33</td>
<td style="text-align: center;">48.73</td>
<td style="text-align: center;">47.47</td>
<td style="text-align: center;">56.78</td>
<td style="text-align: center;">67.86</td>
<td style="text-align: center;">74.00</td>
<td style="text-align: center;">51.76</td>
<td style="text-align: center;">61.09</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Llama2-13b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>20.32</strong></span></td>
<td style="text-align: center;"><span> <strong>18.33</strong></span></td>
<td style="text-align: center;"><span> <strong>13.88</strong></span></td>
<td style="text-align: center;"><span> <strong>12.47</strong></span></td>
<td style="text-align: center;"><span> <strong>26.78</strong></span></td>
<td style="text-align: center;"><span> <strong>26.32</strong></span></td>
<td style="text-align: center;"><span> <strong>32.00</strong></span></td>
<td style="text-align: center;"><span> <strong>8.04</strong></span></td>
<td style="text-align: center;"><span> <strong>22.16</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">50.54</td>
<td style="text-align: center;">67.35</td>
<td style="text-align: center;">45.45</td>
<td style="text-align: center;">30.77</td>
<td style="text-align: center;">64.58</td>
<td style="text-align: center;">74.53</td>
<td style="text-align: center;">74.23</td>
<td style="text-align: center;">69.73</td>
<td style="text-align: center;">63.20</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">50.51</td>
<td style="text-align: center;">73.30</td>
<td style="text-align: center;">43.32</td>
<td style="text-align: center;">36.79</td>
<td style="text-align: center;">63.75</td>
<td style="text-align: center;">77.51</td>
<td style="text-align: center;">82.65</td>
<td style="text-align: center;">52.82</td>
<td style="text-align: center;">55.47</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vicuna-7b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>-0.03</strong></span></td>
<td style="text-align: center;"><span> <strong>5.95</strong></span></td>
<td style="text-align: center;"><span> <strong>-2.13</strong></span></td>
<td style="text-align: center;"><span> <strong>6.02</strong></span></td>
<td style="text-align: center;"><span> <strong>-0.83</strong></span></td>
<td style="text-align: center;"><span> <strong>2.98</strong></span></td>
<td style="text-align: center;"><span> <strong>8.42</strong></span></td>
<td style="text-align: center;"><span> <strong>-16.91</strong></span></td>
<td style="text-align: center;"><span> <strong>-7.73</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">69.19</td>
<td style="text-align: center;">83.92</td>
<td style="text-align: center;">49.24</td>
<td style="text-align: center;">57.07</td>
<td style="text-align: center;">79.50</td>
<td style="text-align: center;">81.03</td>
<td style="text-align: center;">87.76</td>
<td style="text-align: center;">68.50</td>
<td style="text-align: center;">65.77</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">72.45</td>
<td style="text-align: center;">83.82</td>
<td style="text-align: center;">59.78</td>
<td style="text-align: center;">59.69</td>
<td style="text-align: center;">80.25</td>
<td style="text-align: center;">79.59</td>
<td style="text-align: center;">88.66</td>
<td style="text-align: center;">72.86</td>
<td style="text-align: center;">65.31</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vicuna-13b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>3.26</strong> </span></td>
<td style="text-align: center;"><span> <strong>-0.10</strong> </span></td>
<td style="text-align: center;"><span><strong>10.54</strong></span></td>
<td style="text-align: center;"><span> <strong>2.62</strong></span></td>
<td style="text-align: center;"><span><strong>0.75</strong></span></td>
<td style="text-align: center;"><span><strong>-1.44</strong> </span></td>
<td style="text-align: center;"><span> <strong>0.90</strong></span></td>
<td style="text-align: center;"><span> <strong>4.36</strong></span></td>
<td style="text-align: center;"><span> <strong>-0.46</strong></span></td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 0</span></td>
<td style="text-align: center;">79.90</td>
<td style="text-align: center;">84.92</td>
<td style="text-align: center;">69.63</td>
<td style="text-align: center;">64.50</td>
<td style="text-align: center;">86.08</td>
<td style="text-align: center;">94.42</td>
<td style="text-align: center;">92.00</td>
<td style="text-align: center;">73.23</td>
<td style="text-align: center;">70.90</td>
</tr>
<tr>
<td style="text-align: center;"></td>
<td style="text-align: center;"><span class="math inline"><em>x</em> = 5</span></td>
<td style="text-align: center;">76.26</td>
<td style="text-align: center;">84.44</td>
<td style="text-align: center;">66.15</td>
<td style="text-align: center;">65.66</td>
<td style="text-align: center;">90.48</td>
<td style="text-align: center;">91.88</td>
<td style="text-align: center;">90.00</td>
<td style="text-align: center;">77.16</td>
<td style="text-align: center;">70.99</td>
</tr>
<tr>
<td style="text-align: center;"><strong>Vicuna-33b</strong></td>
<td style="text-align: center;"><span class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span> <strong>-3.64</strong></span></td>
<td style="text-align: center;"><span> <strong>-0.48</strong></span></td>
<td style="text-align: center;"><span> <strong>-3.48</strong></span></td>
<td style="text-align: center;"><span> <strong>1.16</strong></span></td>
<td style="text-align: center;"><span> <strong>4.40</strong></span></td>
<td style="text-align: center;"><span> <strong>-2.54</strong></span></td>
<td style="text-align: center;"><span> <strong>-2.00</strong></span></td>
<td style="text-align: center;"><span> <strong>3.93</strong></span></td>
<td style="text-align: center;"><span> <strong>0.09</strong></span></td>
</tr>
</tbody>
</table>

#### Human Evaluation

To investigate human performance in tool selection, we evaluated human abilities through questionnaires.

Specifically, we mixed questions from four sub-tasks, asking participants to select 0 to 2 tools for each question. Each questionnaire comprised 10 or 15 questions, with participants making choices based on provided queries and candidate tools as options. We collected a total of 240 valid responses. The results of the human evaluation are presented in Table <a href="#tab:human_res" data-reference-type="ref" data-reference="tab:human_res">13</a>.

<div id="tab:human_res">

| **Sub-task** | **Similar** | **Scenario** | **Reliability** | **Multi-tool** |
|:---|:--:|:--:|:--:|:--:|
| **Model$`_\textsc{Max}`$** | 73.46 | 79.51 | 50.35 | 88.28 |
| **Model$`_\textsc{Avg}`$** | 56.90 | 61.51 | 8.59 | 55.14 |
| **Human** | 86.00 | 91.00 | 96.00 | 66.00 |

Comparison between human and LLMs. Model$`_\textsc{Max}`$ is the best performance of LLMs and Model$`_\textsc{Avg}`$ is the average performance of eight LLMs.

</div>

We observe a notable discrepancy between the CSR of humans and LLMs in sub-task 1, sub-task 2, and sub-task 3. Human CSR surpasses both the average and maximum CSR of LLMs. Notably, in sub-task 3, human performance reaches an impressive 96%, a stark contrast to the model’s meager 9%. This discrepancy highlights the challenges LLMs face, particularly in addressing issues like hallucination, significantly impacting their reliability.

Moreover, in sub-task 4, human performance, while surpassing the average level of LLMs, falls short of reaching their maximum CSR. This implies that, when confronted with intricate language tasks, such as multiple-choice questions, LLMs still maintain a distinct advantage.

### Prompt Template

#### <span class="smallcaps">ToolE</span> Dataset Generation

We show the prompt templates of <span class="smallcaps">ToolE</span> dataset generation as follows:

<div class="tcolorbox">

`Here is a tool for ChatGPT, which can help it solve users’ requests better. The description of this tool includes a description of users and a description of ChatGPT. `

`The description of users: {human description}`

`The description of ChatGPT: {model description}`

`Please give 10 examples where you would use this plugin to answer a user’s question and you should only tell me what users will say.`

`Please ensure that the provided examples are distinct from one another. Feel free to employ various sentence styles, such as instructions or requests, and vary the level of detail as needed.`

`The format of your answer should be like: 1. User: [Your Answer]. 2. User: [Your Answer] ...`

</div>

<div class="tcolorbox">

`Here is a plugin designed to enhance ChatGPT’s responsiveness to users’ needs. ChatGPT only uses the plugin when it thinks the tools will enhance its response. Now, I would like you to complete the following tasks:`

`I will provide you with a description of the plugin, and based on that description, you need to provide five examples of user inputs that would prompt ChatGPT to utilize the plugin in order to enhance its responses for users.`

`Please ensure that your answers satisfy the following conditions:`

`1. Each example should be the first input in a new conversation, without any prior context.`

`2. The sentence should contain description information.`

`3. Your answers should be as detailed as possible.`

`4. Format your answers as follows: 1. User: [Answer], 2. User: [Answer], ...`

`5. Utilizing this plugin has the potential to significantly improve ChatGPT’s ability to address users’ requests.`

`The plugin includes descriptions for both users and ChatGPT.`

`The description for users is as follows: {human description}, and the description for ChatGPT is: {model description}.`

</div>

<div class="tcolorbox">

`Here is a tool for ChatGPT, which can help it solve users’ requests better. `

`The description of this tool includes a description of users and a description of ChatGPT.`

`The description of users: {human description}`

`The description of ChatGPT: {model description}.`

`Now your task is to give me five words or phrases to label this tool.`

`These labels can be not mentioned in the description and labels should be as short as possible.`

`The format of your answer should be 1. [label 1], 2. [label 2], ..., 5. [label 5].`

</div>

<div class="tcolorbox">

`Here is a tool for ChatGPT, which can help it solve users’ requests better.`

`The description of this tool includes a description of users and a description of ChatGPT.`

`The description of users: {human description}`

`The description of ChatGPT: {model description}`

`Now I will give you five labels of this tool and what you need to do is give me five sentences the user will input to ChatGPT when they may need the help of this tool.`

`Here are the labels: {labels}`

`The format of your answer should be: 1. [label 1]: [sentence 1], 2. [label 2]: [sentence 2], ..., 5. [label 5]: [sentence 5].`

</div>

<div class="tcolorbox">

`Here is a plugin designed to enhance ChatGPT’s responsiveness to users’ needs. ChatGPT only uses the plugin when it thinks the tools will enhance its response. Now, I would like you to complete the following tasks:`

`I will provide you with a description of the plugin, and based on that description, you need to provide five examples of user inputs that would prompt ChatGPT to utilize the plugin in order to enhance its responses for users.`

`Please ensure that your answers satisfy the following conditions:`

`1. Each example should be the first input in a new conversation, without any prior context.`

`2. The sentence should contain description information.`

`3. The example should be in a {Emotion type} mood.`

`4. Format your answers as follows: 1. User: [Answer], 2. User: [Answer], ...`

`5. Utilizing this plugin has the potential to significantly improve ChatGPT’s ability to address users’ requests.`

`The plugin includes descriptions for both users and ChatGPT.`

`The description for users is as follows: {human description}, and the description for ChatGPT is: {model description}.`

</div>

<div class="tcolorbox">

`Now you are a query generation assistant.`

`Here are two tools for ChatGPT, which can help it solve users’ requests better.`

`The first tool is {first tool name}, its description is "{first tool description}".`

`The second tool is {second tool name}, its description is "{second tool description}".`

`Please give five examples where you would use these two tools AT THE SAME TIME to answer a user’s query and you should only tell me what users will say.`

`Remember that the queries you give must be related to both two tools!`

`Please ensure that the provided examples are distinct from one another. Feel free to employ various sentence styles, such as instructions or requests, and vary the level of detail as needed.`

`Remember that the 5 queries you generate should include both parallel tool usage (two tools without any relationship) and causal tool usage (one tool dependent on the result of the other tool). Also, the user’s query cannot include the specific name of the tool.`

`The format of your answer should be like: 1. User: [Your Answer]. 2. User: [Your Answer] ... `

</div>

#### Prompt Template of Experiments

We show the experimental prompt in this section including the Thought part () and the Action part ().

<div class="tcolorbox">

`You are an intelligent agent, and you need to constantly be aware of your own limitations. I will provide you with a user’s query, and you should assess, based on your own capabilities, whether you need to use external tools to better address the user’s query. Typically, there are four reasons why you might need to use external tools:`

`[Reasons Begin]`

`{tool_reason}`

`[Reasons End]`

`Here is the user’s query:`

`[User Query Begins]`

`{user_query}`

`[User Query Ends]`

`Based on the above query, if you think it’s necessary to use external tools, please respond with "yes"; otherwise, respond with "no." Additionally, you should provide a brief explanation for your answer.`

</div>

<div class="tcolorbox">

`You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user’s query based on their question. I will provide you with the user’s question and information about the tools.`

`If there is a tool in the list that is applicable to this query, please return the name of the tool (you can only choose one tool). If there isn’t, please return ’None.’ Additionally, you will need to support your answer with a brief explanation.`

`User’s Query:`

`[User’s Query Start]`

`{user_query}`

`[User’s Query End].`

`List of Tools with Names and Descriptions:`

`[List of Tools with Names and Descriptions Start]`

`{tool_list}`

`[List of Tools with Names and Descriptions End]`

</div>

<div class="tcolorbox">

`You are a helpful AI assistant. Your current task is to choose the appropriate tool to solve the user’s query based on their question. I will provide you with the user’s question and information about the tools.`

`If there is a tool in the list that is applicable to this query, please return the name of the tool (you can choose two tools at most). If there isn’t, please return ’None.’ Additionally, you will need to support your answer with a brief explanation.`

`User’s Query:`

`[User’s Query Start]`

`{user_query}`

`[User’s Query End].`

`List of Tools with Names and Descriptions:`

`[List of Tools with Names and Descriptions Start]`

`{tool_list}`

`[List of Tools with Names and Descriptions End]`

</div>

### Failure Case Study

We show some failure cases of tool selection motivation in LLMs, as shown in Table <a href="#tab:failurecase1" data-reference-type="ref" data-reference="tab:failurecase1">14</a> and Table <a href="#tab:failurecase2" data-reference-type="ref" data-reference="tab:failurecase2">15</a>. The following conclusions can be drawn from these cases: **(1) LLMs can be overconfident or lack confidence.** On one hand, LLMs demonstrate overconfidence in their capabilities, as exemplified by the first three instances in Table <a href="#tab:failurecase1" data-reference-type="ref" data-reference="tab:failurecase1">14</a>, where LLMs erroneously perceive themselves as having the capacity to offer reminders and conduct internet searches. On the other hand, LLMs display an unquestioning reliance on external tools, as evidenced by three instances in Table <a href="#tab:failurecase2" data-reference-type="ref" data-reference="tab:failurecase2">15</a>. Even for conventional language model tasks like sentiment classification and text generation, they prefer tool-based solutions. **(2) LLMs have conflicting perceptions of their own capabilities.** As demonstrated by the 4th, 5th, and 6th examples in Table <a href="#tab:failurecase1" data-reference-type="ref" data-reference="tab:failurecase1">14</a>, ChatGPT provides contradictory responses regarding the utilization of external tools when handling mathematical and logical queries. Based on the aforementioned findings, it is imperative to conduct additional research and training to enhance LLMs’ proficiency in correctly utilizing tools.

<div id="tab:failurecase1">

| **ID** | **Query** | **Answer** |
|:---|:---|:---|
| 1 | I need to remember to attend a meeting at 10 a.m. Can you set a reminder for me? | No, it is not necessary to use external tools for this query. As an intelligent agent, I have the capability to set reminders for users and can do so without the need for external tools. |
| 2 | Can you tell me more about the services offered by Magnetis? | No. As an AI language model, I have access to a vast amount of information on the internet and can gather information about services offered by Magnetis by searching relevant web pages and extracting information. Therefore, I do not need to use external tools to address this query. |
| 3 | My brother is a music lover. Can you find a gift related to music for him? | No. The query is straightforward and does not require external tools. The agent can search for music-related gift ideas using its own search capabilities. |
| 4 | What is the result of $`3.0117^{3.82488}`$? | No. As an AI language model, I can perform basic mathematical calculations, including exponentiation. I can directly calculate the result of 3.0117 raised to the power of 3.82488 without the need for external tools. |
| 5 | Find the four smallest perfect numbers. | Yes, it is necessary to use external tools to find the four smallest perfect numbers. |

Examples of tool usage awareness on ChatGPT.

</div>

<div id="tab:failurecase2">

<table>
<caption>Examples of tool usage awareness on vicuna.</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>ID</strong></th>
<th style="text-align: left;"><strong>Query</strong></th>
<th style="text-align: left;"><strong>Answer</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">1</td>
<td style="text-align: left;"><p>Detect if the Reddit thread contains hate speech. r/worldnews</p>
<p><span class="math inline">•</span> Posted by u/madneon_ 3 hours ago</p>
<p>Russia ‘borrows $13.6bn in largest ever debt issuance in a single day’ to continue faltering war</p>
<p><span class="math inline">•</span> macross1984 · 3 hr. ago</p>
<p>Russia will become deadbeat.</p></td>
<td style="text-align: left;">Yes. To detect if the Reddit thread contains hate speech, external tools such as natural language processing (NLP) models or sentiment analysis algorithms may be required. These tools can analyze the text and identify any offensive or hateful language used in the comments.</td>
</tr>
<tr>
<td style="text-align: left;">2</td>
<td style="text-align: left;">Prove that every set is countable.</td>
<td style="text-align: left;">Yes. The query involves proving a mathematical statement, specifically about the countability of sets. As a language model, I can provide a general explanation or approach to solving this problem, but I cannot perform real-time calculations or generate formal proofs. To fully address the user’s query, it would be beneficial to use external tools such as mathematical software (e.g., Mathematica, MATLAB) or consult with a human expert in the field of mathematics..</td>
</tr>
<tr>
<td style="text-align: left;">3</td>
<td style="text-align: left;">Create a fun math question for children.</td>
<td style="text-align: left;">Yes, it is necessary to use external tools. The query requires the creation of a fun math question for children, which involves generating a specific type of content that may not fall within the capabilities of the language model. External tools or resources may be needed to generate appropriate and engaging math questions that are suitable for children.</td>
</tr>
</tbody>
</table>

</div>

[^1]: We separate the prompt of *Thought* () and *Action* () to avoid the influence taken from different tool lists.
