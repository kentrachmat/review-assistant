# Quadratic models for understanding catapult dynamics of neural networks

## Abstract


Large language models (LLMs) have garnered significant attention due to their impressive natural language processing (NLP) capabilities. Recently, many studies have focused on the tool utilization ability of LLMs. They primarily investigated how LLMs effectively collaborate with given specific tools. However, in scenarios where LLMs serve as intelligent agents, as seen in applications like AutoGPT and MetaGPT, LLMs are expected to engage in intricate decision-making processes that involve deciding whether to employ a tool and selecting the most suitable tool(s) from a collection of available tools to fulfill user requests. Therefore, in this paper, we introduce <span class="smallcaps">MetaTool</span>, a benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. Specifically, we create a dataset called <span class="smallcaps">ToolE</span> within the benchmark. This dataset contains various types of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-tool scenarios. Subsequently, we set the tasks for both tool usage awareness and tool selection. We define four subtasks from different perspectives in tool selection, including *tool selection with similar choices*, *tool selection in specific scenarios* , *tool selection with possible reliability issues*, and *multi-tool selection*. We conduct experiments involving eight popular LLMs and find that the majority of them still struggle to effectively select tools, highlighting the existing gaps between LLMs and genuine intelligent agents. However, through the error analysis, we found there is still significant room for improvement. Finally, we conclude with insights for tool developers – we strongly recommend that tool developers choose an appropriate rewrite model for generating new descriptions based on the downstream LLM the tool will apply to. Our <span class="smallcaps">ToolE</span> dataset is available at [URL](https://atlas.nomic.ai/map/a43a6a84-4453-428a-8738-2534d7bf0b89/b2b8134b-a37e-45d2-a0d9-765911f27df6) and code is in [Github](https://github.com/HowieHwong/MetaTool).

# Introduction

A recent remarkable finding on neural networks, originating from  and termed as the “transition to linearity” , is that, as network width goes to infinity, such models become linear functions in the parameter space. Thus, a linear (in parameters) model can be built to accurately approximate wide neural networks under certain conditions. While this finding has helped improve our understanding of trained neural networks , not all properties of finite width neural networks can be understood in terms of linear models, as is shown in several recent works . In this work, we show that properties of finitely wide neural networks in optimization and generalization that cannot be captured by linear models are, in fact, manifested in quadratic models.

The training dynamics of linear models with respect to the choice of the learning rates[^1] are well-understood . Indeed, such models exhibit *linear* training dynamics, i.e., there exists a critical learning rate, $`{\eta_{\mathrm{crit}}}`$, such that the loss converges monotonically if and only if the learning rate is smaller than $`{\eta_{\mathrm{crit}}}`$ (see Figure <a href="#fig:op_regimes" data-reference-type="ref" data-reference="fig:op_regimes">1</a>a).

<figure id="fig:op_regimes">
<figure>
<img src="./figures/linear-regime.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figure>
<img src="./figures/non-linear-regime.png"" />
<figcaption aria-hidden="true"></figcaption>
</figure>
<figcaption><span><strong><span>Optimization dynamics for linear and non-linear models based on choice of learning rate.</span></strong></span> (<span><strong><span>a</span></strong></span>) Linear models either converge monotonically if learning rate is less than <span class="math inline"><em>η</em><sub>crit</sub></span> and diverge otherwise. (<span><strong><span>b</span></strong></span>) Unlike linear models, <span><em>finitely wide neural networks</em></span> and <span><em>NQMs Eq. (<a href="#eq:nn_quad" data-reference-type="ref" data-reference="eq:nn_quad">[eq:nn_quad]</a>) (or general quadratic models Eq. (<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>))</em></span> can additionally observe a catapult phase when <span class="math inline"><em>η</em><sub>crit</sub> &lt; <em>η</em> &lt; <em>η</em><sub>max</sub></span>. </figcaption>
</figure>

<figure id="fig:ball_illustration">
<figure>
<img src="./figures/ball_fig_new.png"" />
<figcaption>Optimization dynamics for <span class="math inline"><em>f</em></span> (wide neural networks): linear dynamics and catapult dynamics.</figcaption>
</figure>
<figure>
<img src="./figures/cifar-2.png"" />
<figcaption>Generalization performance for <span class="math inline"><em>f</em></span>, <span class="math inline"><em>f</em><sub>lin</sub></span> and <span class="math inline"><em>f</em><sub>quad</sub></span>.</figcaption>
</figure>
<figcaption><span><strong><span>(a) Optimization dynamics of wide neural networks with sub-critical and super-critical learning rates.</span></strong></span> With sub-critical learning rates (<span class="math inline">0 &lt; <em>η</em> &lt; <em>η</em><sub>crit</sub>)</span>, the tangent kernel of wide neural networks is nearly constant during training, and the loss decreases monotonically. The whole optimization path is contained in the ball <span class="math inline"><em>B</em>(<strong>w</strong><sub>0</sub>, <em>R</em>) := {<strong>w</strong> : ∥<strong>w</strong> − <strong>w</strong><sub>0</sub>∥ ≤ <em>R</em>}</span> with a finite radius <span class="math inline"><em>R</em></span>. With super-critical learning rates (<span class="math inline"><em>η</em><sub>crit</sub> &lt; <em>η</em> &lt; <em>η</em><sub>max</sub>)</span>, the catapult phase happens: the loss first increases and then decreases, along with a decrease of the norm of the tangent kernel . The optimization path goes beyond the finite radius ball. <span><strong>(b) Test loss of <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span> and <span class="math inline"><em>f</em><sub>lin</sub></span> plotted against different learning rates.</strong></span> With sub-critical learning rates, all three models have nearly identical test loss for any sub-critical learning rate. With super-critical learning rates, <span class="math inline"><em>f</em></span> and <span class="math inline"><em>f</em><sub>quad</sub></span> have smaller best test loss than the one with sub-critical learning rates. Experimental details are in Appendix <a href="#subsec:exp_quad_setting" data-reference-type="ref" data-reference="subsec:exp_quad_setting">19.5</a>.</figcaption>
</figure>

Recent work  showed that the training dynamics of a wide neural network $`f({\mathbf{w}};{\boldsymbol{x}})`$ can be accurately approximated by that of a linear model $`f_{{{\mathrm{lin}}}}({\mathbf{w}};{\boldsymbol{x}})`$:
``` math
\begin{aligned}
\label{eq:nn_linear}
    f_{\mathrm{lin}}({\mathbf{w}};{\boldsymbol{x}}) = f({\mathbf{w}}_0;{\boldsymbol{x}}) + ({\mathbf{w}}- {\mathbf{w}}_0)^T \nabla f({\mathbf{w}}_0;{\boldsymbol{x}}),
\end{aligned}
```
where $`\nabla f({\mathbf{w}}_0;{\boldsymbol{x}})`$ denotes the gradient[^2] of $`f`$ with respect to trainable parameters $`{\mathbf{w}}`$ at an initial point $`{\mathbf{w}}_0`$ and input sample $`{\boldsymbol{x}}`$. This approximation holds for learning rates less than $`{\eta_{\mathrm{crit}}}\approx 2/\|\nabla f({\mathbf{w}}_0;{\boldsymbol{x}})\|^2`$, when the width is sufficiently large.

However, the training dynamics of finite width neural networks, $`f`$, can sharply differ from those of linear models when using large learning rates. A striking non-linear property of wide neural networks discovered in  is that when the learning rate is larger than $`{\eta_{\mathrm{crit}}}`$ but smaller than a certain maximum learning rate, $`{\eta_{\mathrm{max}}}`$, gradient descent still converges but experiences a “catapult phase.” Specifically, the loss initially grows exponentially and then decreases after reaching a large value, along with the decrease of the norm of tangent kernel (see Figure <a href="#fig:ball_illustration" data-reference-type="ref" data-reference="fig:ball_illustration">2</a>a), and therefore, such training dynamics are *non-linear* (see Figure <a href="#fig:op_regimes" data-reference-type="ref" data-reference="fig:op_regimes">1</a>b).

As linear models cannot exhibit such a catapult phase, under what models and conditions does this phenomenon arise? The work of  first observed the catapult phase phenomenon in finite width neural networks and analyzed this phenomenon for a two-layer linear neural network. However, a theoretical understanding of this phenomenon for general non-linear neural networks remains open. In this work, we utilize a quadratic model as a tool to shed light on the optimization and generalization discrepancies between finite and infinite width neural networks. We define *Neural Quadratic Model (NQM)* by the second order Taylor series expansion of $`f({\mathbf{w}};{\boldsymbol{x}})`$ around the point $`{\mathbf{w}}_0`$:
``` math
\begin{aligned}
\label{eq:nn_quad}

\hspace*{-1cm}\mathbf{NQM:} ~~~~f_{\mathrm{quad}}({\mathbf{w}})=  f({\mathbf{w}}_0) + ({\mathbf{w}}- {\mathbf{w}}_0)^T \nabla f({\mathbf{w}}_0) + \frac{1}{2}({\mathbf{w}}- {\mathbf{w}}_0)^T H_f({\mathbf{w}}_0) ({\mathbf{w}}- {\mathbf{w}}_0).
\end{aligned}
```
Here in the notation we suppress the dependence on the input data $`{\boldsymbol{x}}`$, and $`H_f({\mathbf{w}}_0)`$ is the Hessian of $`f`$ with respect to $`{\mathbf{w}}`$ evaluated at $`{\mathbf{w}}_0`$. Note that $`f_{\mathrm{quad}}({\mathbf{w}}) = f_{\mathrm{lin}}({\mathbf{w}}) + \frac{1}{2}({\mathbf{w}}- {\mathbf{w}}_0)^T H_f({\mathbf{w}}_0) ({\mathbf{w}}- {\mathbf{w}}_0)`$.

Indeed, we note that NQMs are contained in a more general class of quadratic models:
``` math
\begin{aligned}
\label{eq:quadratic}
  \hspace*{-3.3cm} \mathbf{General~Quadratic~Model:} ~~~~~~~g({\mathbf{w}};{\boldsymbol{x}}) = {\mathbf{w}}^T \phi({\boldsymbol{x}}) + \frac{1}{2}\gamma {\mathbf{w}}^T\Sigma({\boldsymbol{x}}) {\mathbf{w}},
\end{aligned}
```
where $`{\mathbf{w}}`$ are trainable parameters and $`{\boldsymbol{x}}`$ is input data. We discuss the optimization dynamics of such general quadratic models in Section <a href="#subsec:summary_catapult" data-reference-type="ref" data-reference="subsec:summary_catapult">3.3</a> and show empirically that they exhibit the catapult phase phenomenon in Appendix <a href="#subsec:exp_gqm" data-reference-type="ref" data-reference="subsec:exp_gqm">19.4</a>. Note that the two-layer linear network analyzed in  is a special case of Eq. (<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>), when $`\phi({\boldsymbol{x}})=0`$ (See Appendix <a href="#sec:phi_x_0" data-reference-type="ref" data-reference="sec:phi_x_0">18</a>).

**Main Contributions.** We prove that NQMs, $`f_{{\mathrm{quad}}}`$, which approximate shallow fully-connected ReLU activated neural networks, exhibit catapult phase dynamics. Specifically, we analyze the optimization dynamics of $`f_{{\mathrm{quad}}}`$ by deriving the evolution of $`f_{{\mathrm{quad}}}`$ and the tangent kernel during gradient descent with squared loss, for a single training example and multiple uni-dimensional training examples. We identify three learning rate regimes yielding different optimization dynamics for $`f_{{\mathrm{quad}}}`$, which are (1) converging monotonically (linear dynamics); (2) converging via a catapult phase (catapult dynamics); and (3) diverging. We provide a number of experimental results corroborating our theoretical analysis (See Section <a href="#sec:catapult" data-reference-type="ref" data-reference="sec:catapult">3</a>).

We then empirically show that NQMs, for the architectures of shallow (see Figure <a href="#fig:ball_illustration" data-reference-type="ref" data-reference="fig:ball_illustration">2</a>b as an example) and deep networks, have better test performances when catapult dynamics happens. While this was observed for some synthetic examples of neural networks in , we systematically demonstrate the improved generalization of NQMs across a range of experimental settings. Namely, we consider fully-connected and convolutional neural networks with ReLU and other activation functions trained with GD/SGD on multiple vision, speech and text datatsets (See Section <a href="#sec:exp_catapult" data-reference-type="ref" data-reference="sec:exp_catapult">4</a>).

To the best of our knowledge, our work is the first to analyze the non-linear wide neural networks in the catapult regime through the perspective of the quadratic approximation. While NQMs (or quadratic models) were proposed and analyzed in , our work focuses on the properties of NQMs in the large learning rate regime, which has not been discussed in . Similarly, the following related works did not study catapult dynamics. analyzed higher order approximations to neural networks under gradient flow (infinitesimal learning rates). studied different quadratic models with randomized second order terms and considered the loss in the quadratic form, where no catapult phase happens. A recent work showed the existence of the catapult phase in two-layer, homogenous networks .
