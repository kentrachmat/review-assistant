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