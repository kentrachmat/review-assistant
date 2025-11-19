# Quadratic models for understanding catapult dynamics of neural networks

## Abstract

The CRISPR (clustered regularly interspaced short palindromic repeat)–Cas9 (CRISPR-associated nuclease 9) system is poised to transform developmental biology by providing a simple, efficient method to precisely manipulate the genome of virtually any developing organism. This RNA-guided nuclease (RGN)-based approach already has been effectively used to induce targeted mutations in multiple genes simultaneously, create conditional alleles, and generate endogenously tagged proteins. Illustrating the adaptability of RGNs, the genomes of >20 different plant and animal species as well as multiple cell lines and primary cells have been successfully modified. Here we review the current and potential uses of RGNs to investigate genome function during development

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

#### Discontinuity in dynamics transition.

In the ball $`B({\mathbf{w}}_0,R):=\{{\mathbf{w}}: \|{\mathbf{w}}- {\mathbf{w}}_0\|\leq R\}`$ with constant radius $`R>0`$, the transition to linearity of a wide neural network (with linear output layer) is continuous in the network width $`m`$. That is, the deviation from the network function to its linear approximation within the ball can be continuously controlled by the Hessian of the network function, i.e. $`H_f`$, which scales with $`m`$ :
``` math
\begin{aligned}
\label{eq:deviation}

     \|f({\mathbf{w}}) - f_{{{\mathrm{lin}}}}({\mathbf{w}})\|\leq \sup_{{\mathbf{w}}\in B({\mathbf{w}}_0,R)}\|H_f({\mathbf{w}})\|R^2 =\tilde{O}(1/\sqrt{m}).
     
\end{aligned}
```
Using the inequality from Eq. (<a href="#eq:deviation" data-reference-type="ref" data-reference="eq:deviation">[eq:deviation]</a>), we obtain $`\|f_{{\mathrm{quad}}} - f_{{{\mathrm{lin}}}}\| =\tilde{O}(1/\sqrt{m})`$, hence $`f_{{\mathrm{quad}}}`$ transitions to linearity continuously as well in $`B({\mathbf{w}}_0,R)`$[^3]. Given the continuous nature of the transition to linearity, one may expect that the transition from non-linear dynamics to linear dynamics for $`f`$ and $`f_{{\mathrm{quad}}}`$ is continuous in $`m`$ as well. Namely, one would expect that the domain of catapult dynamics, $`[{\eta_{\mathrm{crit}}},{\eta_{\mathrm{max}}}]`$, shrinks and ultimately converges to a single point, i.e., $`{\eta_{\mathrm{crit}}}={\eta_{\mathrm{max}}}`$, as $`m`$ goes to infinity, with non-linear dynamics turning into linear dynamics. However, as shown both analytically and empirically, the transition is *not* continuous, for both network functions $`f`$ and NQMs $`f_{{\mathrm{quad}}}`$, since the domain of the catapult dynamics can be independent of the width $`m`$ (or $`\gamma`$). Additionally, the length of the optimization path of $`f`$ in catapult dynamics grows with $`m`$ since otherwise, the optimization path could be contained in a ball with a constant radius independent of $`m`$, in which $`f`$ can be approximated by $`f_{{{\mathrm{lin}}}}`$. Since the optimization of $`f_{{{\mathrm{lin}}}}`$ diverges in catapult dynamics, by the approximation, the optimization of $`f`$ diverges as well, which contradicts the fact that the optimization of $`f`$ can converge in catapult dynamics (See Figure <a href="#fig:ball_illustration" data-reference-type="ref" data-reference="fig:ball_illustration">2</a>a).

# Notation and preliminary

We use bold lowercase letters to denote vectors and capital letters to denote matrices. We denote the set $`\{1, 2, \cdots , n\}`$ by $`[n]`$. We use $`\|\cdot\|`$ to denote the Euclidean norm for vectors and the spectral norm for matrices. We use $`\odot`$ to denote element-wise multiplication (Hadamard product) for vectors. We use $`\lambda_{\max}(A)`$ and $`\lambda_{\min}(A)`$ to denote the largest and smallest eigenvalue of a matrix $`A`$, respectively.

Given a model $`f({\mathbf{w}}; {\boldsymbol{x}})`$, where $`{\boldsymbol{x}}`$ is input data and $`{\mathbf{w}}`$ are model parameters, we use $`\nabla_{\mathbf{w}}f`$ to represent the partial first derivative $`\partial f({\mathbf{w}}; {\boldsymbol{x}})/\partial {\mathbf{w}}`$. When clear from context, we let $`\nabla f:=\nabla_{{\mathbf{w}}}f`$ for ease of notation. We use $`H_f`$ and $`H_{{\mathcal{L}}}`$ to denote the Hessian (second derivative matrix) of the function $`f({\mathbf{w}};{\boldsymbol{x}})`$ and the loss $`{\mathcal{L}}({\mathbf{w}})`$ with respect to parameters $`{\mathbf{w}}`$, respectively.

In the paper, we consider the following supervised learning task: given training data $`\{({\boldsymbol{x}}_i,y_i)\}_{i=1}^n`$ with data $`{\boldsymbol{x}}_i \in \mathbb{R}^d`$ and labels $`y_i \in\mathbb{R}`$ for $`i\in[n]`$, we minimize the empirical risk with the squared loss $`{\mathcal{L}}({\mathbf{w}}) = \frac{1}{2}\sum_{i=1}^n (f({\mathbf{w}};{\boldsymbol{x}}_i) - y_i)^2`$. Here $`f({\mathbf{w}};\cdot)`$ is a parametric family of models, e.g., a neural network or a kernel machine, with parameters $`{\mathbf{w}}\in \mathbb{R}^p`$. We use full-batch gradient descent to minimize the loss, and we denote trainable parameters $`{\mathbf{w}}`$ at iteration $`t`$ by $`{\mathbf{w}}(t)`$. With constant step size (learning rate) $`\eta`$, the update rule for the parameters is:
``` math
\begin{aligned}

    {\mathbf{w}}(t+1) = {\mathbf{w}}(t) - \eta \frac{d{\mathcal{L}}({\mathbf{w}})}{d{\mathbf{w}}}(t),~~\forall t\geq0. 
\end{aligned}
```

<div id="def:ntk" class="definition">

**Definition 1** (Tangent Kernel). *The tangent kernel $`K({\mathbf{w}};\cdot,\cdot)`$ of $`f({\mathbf{w}};\cdot)`$ is defined as
``` math
\begin{aligned}
\label{eq:ntk}
    K({\mathbf{w}};{\boldsymbol{x}},{\boldsymbol{z}}) =\langle \nabla f({\mathbf{w}};{\boldsymbol{x}}),\nabla f({\mathbf{w}};{\boldsymbol{z}})\rangle,~~~~\forall {\boldsymbol{x}},{\boldsymbol{z}}\in\mathbb{R}^d.
    
\end{aligned}
```*

</div>

In the context of the optimization problem with $`n`$ training examples, the tangent kernel matrix $`K\in\mathbb{R}^{n\times n}`$ satisfies $`K_{i,j}({\mathbf{w}}) = K({\mathbf{w}};{\boldsymbol{x}}_i,{\boldsymbol{x}}_j)`$, $`i,j\in[n]`$. The critical learning rate for optimization is given as follows.

<div id="def:crit" class="definition">

**Definition 2** (Critical learning rate). *With an initialization of parameters $`{\mathbf{w}}_0`$, the critical learning rate of $`f({\mathbf{w}};\cdot)`$ is defined as
``` math
\begin{aligned}
\label{eq:crit}
 
    {\eta_{\mathrm{crit}}}:= 2/\lambda_{\max}(H_{{\mathcal{L}}}({\mathbf{w}}_0)).
 
\end{aligned}
```
A learning rate $`\eta`$ is said to be *sub-critical* if $`0< \eta < {\eta_{\mathrm{crit}}}`$ or *super-critical* if $`{\eta_{\mathrm{crit}}}<\eta <\eta_{\max}`$. Here $`{\eta_{\mathrm{max}}}`$ is the maximum leaning rate such that the optimization of $`{\mathcal{L}}({\mathbf{w}})`$ initialized at $`{\mathbf{w}}_0`$ can converge.*

</div>

#### Dynamics for Linear models.

When $`f`$ is linear in $`{\mathbf{w}}`$, the gradient, $`\nabla f`$, and tangent kernel are constant: $`K({\mathbf{w}}(t))= K({\mathbf{w}}_0)`$. Therefore, gradient descent dynamics are:
``` math
\begin{aligned}
\label{eq:linear}
    F({\mathbf{w}}(t+1)) - {\mathbf{y}}&= (I - \eta K({\mathbf{w}}_0))(F({\mathbf{w}}(t))-{\mathbf{y}}),~~~\forall t\geq 0,
\end{aligned}
```
where $`F({\mathbf{w}}_0) = [f_1({\mathbf{w}}_0),...,f_n({\mathbf{w}}_0)]^T`$ with $`f_i({\mathbf{w}}_0) = f({\mathbf{w}}_0;{\boldsymbol{x}}_i)`$.

Noting that $`H_{{\mathcal{L}}}({\mathbf{w}}_0) = \nabla F({\mathbf{w}}_0)^T \nabla F({\mathbf{w}}_0)`$ and that tangent kernel $`K({\mathbf{w}}_0) = \nabla F({\mathbf{w}}_0) \nabla F({\mathbf{w}}_0)^T`$ share the same positive eigenvalues, we have $`\lambda_{\max}(H_{{\mathcal{L}}}({\mathbf{w}}_0)) = \lambda_{\max}(K({\mathbf{w}}_0))`$, and hence,
``` math
{\eta_{\mathrm{crit}}}= 2/\lambda_{\max}(K({\mathbf{w}}_0)).
```
Therefore, from Eq. <a href="#eq:linear" data-reference-type="eqref" data-reference="eq:linear">[eq:linear]</a>, if $`0<\eta<{\eta_{\mathrm{crit}}}`$, the loss $`{\mathcal{L}}`$ decreases monotonically and if $`\eta>{\eta_{\mathrm{crit}}}`$, the loss $`{\mathcal{L}}`$ keeps increasing. Note that the critical and maximum learning rates are equal in this setting.

# Optimization dynamics in Neural Quadratic Models

In this section, we analyze the gradient descent dynamics of the NQM corresponding to a two-layer fully-connected neural network. We show that, unlike a linear model, the NQM exhibits a catapult dynamics: the loss increases at the early stage of training then decreases afterwards. We further show that the top eigenvalues of the tangent kernel typically become smaller as a consequence of the catapult.

#### Neural Quadratic Model (NQM).

Consider the NQM that approximates the following two-layer neural network:
``` math
\begin{aligned}
\label{eq:2relu-nn}
    f({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) = \frac{1}{\sqrt{m}}\sum_{i=1}^m v_i \sigma\left(\frac{1}{\sqrt{d}}{\mathbf{u}}_i^T {\boldsymbol{x}}\right),
\end{aligned}
```
where $`{\mathbf{u}}_i \in \mathbb{R}^{d}`$, $`v_i\in\mathbb{R}`$ for $`i\in[m]`$ are trainable parameters, $`{\boldsymbol{x}}\in \mathbb{R}^d`$ is the input, and $`\sigma(\cdot)`$ is the ReLU activation function. We initialize $`{\mathbf{u}}_i\sim\mathcal{N}(0,I_d)`$ and $`v_i \in \mathrm{Unif}[\{-1,1\}]`$ for each $`i`$ independently. Letting $`g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) := f_{\mathrm{quad}}({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})`$, this NQM has the following expression (See the full derivation in Appendix <a href="#sec:derivation_nqm" data-reference-type="ref" data-reference="sec:derivation_nqm">6</a>):
``` math
\begin{aligned}
\label{eq:nn_quad_relu}
    g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) &= f({\mathbf{u}}_0,{\mathbf{v}}_0;{\boldsymbol{x}}) + \frac{1}{\sqrt{md}}\sum_{i=1}^m v_{0,i}({\mathbf{u}}_i - {\mathbf{u}}_{0,i})^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}   
    +\frac{1}{\sqrt{md}}\sum_{i=1}^m (v_i - v_{0,i}) \sigma\left({\mathbf{u}}_{0,i}^T{\boldsymbol{x}}\right)\nonumber\\ &~~~~+ \frac{1}{\sqrt{md}}\sum_{i=1}^m (v_i-v_{0,i})({\mathbf{u}}_i - {\mathbf{u}}_{0,i})^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}.
\end{aligned}
```

Given training data $`\{{\boldsymbol{x}}_i,y_i\}_{i=1}^n`$, we minimize the empirical risk with the squared loss $`{\mathcal{L}}({\mathbf{w}}) = \frac{1}{2}\sum_{i=1}^n (g({\mathbf{w}};{\boldsymbol{x}}_i) - y_i)^2`$ using GD with constant learning rate $`\eta`$. Throughout this section, we denote $`g({\mathbf{u}}(t),{\mathbf{v}}(t);{\boldsymbol{x}})`$ by $`g(t)`$ and its tangent kernel $`K({\mathbf{u}}(t),{\mathbf{v}}(t))`$ by $`K(t)`$, where $`t`$ is the iteration of GD. We assume $`\|{\boldsymbol{x}}_i\|=O(1)`$ and $`|y_i| = O(1)`$ for $`i\in[n]`$, and we assume the width of $`f`$ is much larger than the input dimension $`d`$ and the data size $`n`$, i.e., $`m \gg \max\{d,n\}`$. Hence, $`d`$ and $`n`$ can be regarded as small constants. In the whole paper, we use the big-O and small-o notation with respect to the width $`m`$. Below, we start with the single training example case, which already showcases the non-linear dynamics of NQMs.

## Catapult dynamics with a single training example

In this subsection, we consider training dynamics of NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>) with a single training example $`({\boldsymbol{x}},y)`$ where $`{\boldsymbol{x}}\in \mathbb{R}^d`$ and $`y\in\mathbb{R}`$. In this case, the tangent kernel matrix $`K`$ reduces to a scalar, and we denote $`K`$ by $`\lambda`$ to distinguish it from a matrix.

By gradient descent with step size $`\eta`$, the updates for $`g(t)-y`$ and $`\lambda(t)`$, which we refer to as dynamics equations, can be derived as follows (see the derivation in Appendix <a href="#subsec:deri:single" data-reference-type="ref" data-reference="subsec:deri:single">7.1</a>):

#### Dynamics equations.

``` math
\begin{aligned}
    g(t+1) - y &= \left(1-\eta \lambda(t) + \underbrace{\frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2 (g(t)-y)g(t)}_{R_g(t)} \right)(g(t) - y) := \mu(t)(g(t)-y),\label{eq:g_evolve}\\
    \lambda(t+1) &= \lambda(t)- \underbrace{\eta\frac{\|{\boldsymbol{x}}\|^2}{md}   (g(t)-y)^2\left( 4\frac{g(t)}{g(t)-y} -\eta\lambda(t)\right) }_{R_{\lambda}(t)},~~~~\forall t\geq 0.\label{eq:k_evolve}
    
\end{aligned}
```

Note that as the loss is given by $`{\mathcal{L}}(t) = \frac{1}{2} (g(t)-y)^2`$, to understand convergence, it suffices to analyze the dynamics equations above. Compared to the linear dynamics Eq. (<a href="#eq:linear" data-reference-type="ref" data-reference="eq:linear">[eq:linear]</a>), this non-linear dynamics has extra terms $`R_g(t)`$ and $`R_{\lambda}(t)`$, which are induced by the non-linear term in the NQM. We will see that the convergence of gradient descent depends on the scale and sign of $`R_g(t)`$ and $`R_\lambda(t)`$. For example, for constant learning rate that is slightly larger than $`{\eta_{\mathrm{crit}}}`$ (which would result in divergence for linear models), $`R_\lambda(t)`$ stays positive during training, resulting in both monotonic decrease of tangent kernel $`\lambda`$ and the loss.

As $`\lambda(t) = \lambda_0 - \sum_{\tau=0}^{t-1} R_\lambda(\tau)`$, to track the scale of $`|\mu(t)|`$, we will focus on the scale and sign of $`R_g(t)`$ and $`R_\lambda(t)`$ in the following analysis. For the scale of $`\lambda_0`$, which is non-negative by Definition <a href="#def:ntk" data-reference-type="ref" data-reference="def:ntk">1</a>, we can show that with high probability over random initialization, $`|\lambda_0| = \Theta(1)`$ (see Appendix <a href="#proof:lower_bound_k_single" data-reference-type="ref" data-reference="proof:lower_bound_k_single">14</a>). And $`|g(0)| = O(1)`$ with high probability as well . Therefore the following discussion is with high probability over random initialization. We start by establishing monotonic convergence for sub-critical learning rates.

#### Monotonic convergence: sub-critical learning rates ($`\eta<2/\lambda_0={\eta_{\mathrm{crit}}}`$).

The key observation is that when $`|g(t)|=O(1)`$, and $`\lambda(t) = \Theta(1)`$, $`|R_g(t)|`$ and $`|R_\lambda(t)|`$ are of the order $`o(1)`$. Then, the dynamics equations approximately reduce to the ones of linear dynamics:
``` math
\begin{aligned}
    g(t+1) - y &= \left(1-\eta \lambda(t) + o(1) \right)(g(t) - y),\\
    \lambda(t+1) &= \lambda(t)+ o(1).
\end{aligned}
```
Note that at initialization, the output satisfies $`|g(0)|=O(1)`$, and we have shown $`\lambda_0 = \Theta(1)`$. With the choice of $`\eta`$, we have for all $`t\geq 0`$, $`|\mu(t)| = |1-\eta\lambda(t)+o(1)|<1`$; hence, $`|g(t)-y|`$ decreases monotonically. The cumulative change on the tangent kernel will be $`o(1)`$, i.e., $`\sum_t |R_{\lambda}(t)| = o(1)`$, since for all $`t`$, $`|R_{\lambda}(t)| = O(1/m)`$ and the loss decreases exponentially hence $`\sum |R_\lambda(t)| = O(1 /m)\cdot \log O(1) = o(1)`$. See Appendix <a href="#sec:sub_crit" data-reference-type="ref" data-reference="sec:sub_crit">8</a> for a detailed discussion.

#### Catapult convergence: super-critical learning rates ($`{\eta_{\mathrm{crit}}}= 2/\lambda_0 <\eta < 4/\lambda_0 = {\eta_{\mathrm{max}}}`$).

The training dynamics are given by the following theorem.

<div id="thm:single" class="thm">

**Theorem 1** (Catapult dynamics on a single training example). *Consider training the NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>) with squared loss on a single training example by GD. With a super-critical learning rate $`\eta \in \left[\frac{2+ \epsilon}{\lambda_0}, \frac{4-\epsilon}{\lambda_0} \right]`$ where $`\epsilon =  \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$, the catapult happens: with high probability over random initialization, the loss increases to the order of $`\Omega\left(\frac{m(\eta\lambda_0-2)^2}{\log m}\right)`$ then decreases to $`O(1)`$.*

</div>

<div class="proof">

*Proof of Theorem <a href="#thm:single" data-reference-type="ref" data-reference="thm:single">1</a>.* We use the following transformation of the variables to simplify notations.
``` math
\begin{aligned}
    u(t) = \frac{\left\|{\boldsymbol{x}}\right\|^2}{md}\eta^2 (g(t)-y)^2,~~~w(t) =\frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)y,~~~v(t) = \eta\lambda(t).
\end{aligned}
```
Then the Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and Eq. (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>) are reduced to
``` math
\begin{aligned}
    u(t+1) &= (1-v(t)+u(t)+w(t))^2 u(t):=\kappa(t)u(t),\label{eq:u}\\
    v(t+1) &=v(t) - u(t)(4-v(t))-4w(t)\label{eq:v}.
\end{aligned}
```
At initialization, since $`|g(0)| = O(1)`$, we have $`u(0) = O\left(\frac{1}{m}\right)`$ and $`w(0) = O\left(\frac{1}{m}\right)`$. Note that by definition, for all $`t\geq 0`$, $`u(t)\geq 0`$ and we have $`v(t)\geq 0`$ since $`\lambda(t)`$ is the tangent kernel for a single training example.

In the following, we will analyze the above dynamical equations. To make the analysis more understandable, we separate the dynamics during training into increasing phase and decreasing phase. We denote $`\delta:= (\eta-{\eta_{\mathrm{crit}}})\lambda_0  = \eta\lambda_0-2`$.

#### Increasing phase.

In this phase, $`|u(t)|`$ increases exponentially from $`O\left(\frac{1}{m}\right)`$ to $`\Theta\left(\frac{\delta^2}{\log m}\right)`$ and $`|v(t) - v(0)| = O\left(\frac{\delta}{\log m}\right)`$. This can be shown by the following lemma.

<div id="lemma:increase" class="lemma">

**Lemma 1**. *For $`T>0`$ such that $`\sup_{t\in[0,T]}u(t) = O\left(\frac{\delta^2}{\log m}\right)`$, $`u(t)`$ increases exponentially with $`\inf_{t\in[0,T]}\kappa(t) \geq \left(1+\delta -O\left(\frac{\delta}{\log m}\right)\right)^2 >1`$ and $`\sup_{t\in[0,T]}|v(t)-v(0)| = O\left(\frac{\delta}{\log m}\right)`$.*

</div>

<div class="proof">

*Proof.* See the proof in Section <a href="#proof:increase" data-reference-type="ref" data-reference="proof:increase">9</a>. ◻

</div>

After the increasing phase, based on the order of $`u(t)`$ we can infer the order of loss is $`\Theta\left(\frac{m\delta^2}{\log m}\right)`$.

#### Decreasing phase.

When $`u(t)`$ is sufficiently large, $`v(t)`$ will have non-negligible decrease which leads to the decreasing of $`\kappa(t)`$, hence in turn making $`u(t)`$ decrease as well. Consequently, we have:

<div id="lemma:kappa_t" class="lemma">

**Lemma 2**. *There exists $`T^*>0`$ such that $`u(T^*) = O\left(\frac{1}{m}\right)`$.*

</div>

<div class="proof">

*Proof.* See the proof in Section <a href="#proof:kappa_t" data-reference-type="ref" data-reference="proof:kappa_t">10</a>. ◻

</div>

Then accordingly, the loss is of the order $`O(1)`$. ◻

</div>

Once the loss decreases to the order of $`O(1)`$, the catapult finishes and we in general have $`\eta < 2/\lambda(t)`$ as $`|\mu(t)| = |1-\eta\lambda(t) + R_{\mathbf{g}}(t)|<1`$ where $`|R_{\mathbf{g}}(t)| =O({\mathcal{L}}(t)/m)= O(1/m)`$. Therefore the training dynamics fall into linear dynamics, and we can use the same analysis for sub-critical learning rates for the remaining training dynamics. The stableness of the steady-state equilibria of dynamical equations can be guaranteed by the following:

<div id="thm:equi" class="thm">

**Theorem 2**. *For dynamical equations Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>), the stable steady-state equilibria satisfy $`g(t)=y`$ (i.e.,loss is $`0`$), and $`\lambda(t)\in[\epsilon,2/\eta-\epsilon]`$ with $`\epsilon = \Theta(\log m/\sqrt{m})`$.*

</div>

#### Divergence ($`\eta > \eta_{\max} = 4/\lambda_0`$).

Initially, it follows the same dynamics with that in the increasing phase in catapult convergence: $`|g(t)-y|`$ increases exponentially as $`|\mu(t)|>1`$ and the $`\lambda(t)`$ almost does not change as $`R_\lambda(t)`$ is small. However, note that $`R_\lambda(t)>0`$ since 1) $`g(t)/(g(t)-y)\approx 1`$ when $`g(t)`$ becomes large and 2) $`\eta>4/\lambda(t)`$. Therefore, $`\lambda(t)`$ keeps increasing during training, which consequently leads to the divergence of the optimization. See Appendix <a href="#sec:max_lr" data-reference-type="ref" data-reference="sec:max_lr">12</a> for a detailed discussion.

## Catapult dynamics with multiple training examples

In this subsection we show the catapult phase will happen for NQMs Eq. (<a href="#eq:2relu-nn" data-reference-type="ref" data-reference="eq:2relu-nn">[eq:2relu-nn]</a>) with multiple training examples. We assume unidimensional input data, which is common in the literature and simplifies the analysis for neural networks (see for example ).

<div id="assump:input" class="assumption">

**Assumption 1**. *The input dimension $`d=1`$ and not all $`x_i`$ is $`0`$, i.e., $`\sum |x_i|>0`$.*

</div>

We similarly analyze the dynamics equations with different learning rates for multiple training examples (see the derivation of Eq. (<a href="#eq:g_evolve_multi_ori" data-reference-type="ref" data-reference="eq:g_evolve_multi_ori">[eq:g_evolve_multi_ori]</a>) and (<a href="#eq:k_evolve_multi_ori" data-reference-type="ref" data-reference="eq:k_evolve_multi_ori">[eq:k_evolve_multi_ori]</a>) in Appendix) which are update equations of $`{\mathbf{g}}(t)-{\mathbf{y}}`$ and $`K(t)`$. And similarly, we show there are three training dynamics: monotonic convergence, catapult convergence and divergence.

In the analysis, we consider the training dynamics projected to two orthogonal eigenvectors of the tangent kernel, i.e., $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$, and we show with different learning rates, the catapult phase can occur only in the direction of $`{\boldsymbol{p}}_1`$, or occur in both directions. We consider the case where $`2/\lambda_2(0) < 4/\lambda_1(0)`$ hence the catapult can occur in both directions. The analysis for the other case can be directly obtained from our results. We denote the loss projected to $`{\boldsymbol{p}}_i`$ by $`\Pi_i {\mathcal{L}}:= \frac{1}{2}\left<{\mathbf{g}}-{\mathbf{y}},{\boldsymbol{p}}_i\right>^2`$ for $`i=1,2`$. We have $`\Pi_i {\mathcal{L}}(0) = O(1)`$ with high probability over random initialization of weights.

We formulate the result for the catapult dynamics, which happens when training with super-critical learning rates, into the following theorem, and defer the proof of it and the full discussion of training dynamics to Appendix <a href="#sec:multi" data-reference-type="ref" data-reference="sec:multi">16</a>.

<div id="thm:multi" class="thm">

**Theorem 3** (Catapult dynamics on multiple training examples). *Supposing Assumption <a href="#assump:input" data-reference-type="ref" data-reference="assump:input">1</a> holds, consider training the NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>) with squared loss on multiple training examples by GD. Then, with high probability over random initialization we have*

1.  *with $`\eta \in \left[\frac{2+\epsilon}{\lambda_1(0)}, \frac{2-\epsilon}{\lambda_2(0)} \right]`$ , the catapult only occurs in eigendirection $`{\boldsymbol{p}}_1`$: $`\Pi_1{\mathcal{L}}`$ increases to the order of $`\Omega\left(\frac{m(\eta\lambda_1(0)-2)^2}{\log m}\right)`$ then decreases to $`O(1)`$;*

2.  *with $`\eta \in \left[\frac{2+\epsilon}{\lambda_2(0)},  \frac{4-\epsilon}{\lambda_1(0)}\right]`$, the catapult occurs in both eigendirections $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$: $`\Pi_i{\mathcal{L}}`$ for $`i=1,2`$ increases to the order of $`\Omega\left(\frac{m(\eta\lambda_i(0)-2)^2}{\log m}\right)`$ then decreases to $`O(1)`$,*

*where $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$.*

</div>

We verify the results for multiple training examples via the experiments in Figure <a href="#fig:multi_quad" data-reference-type="ref" data-reference="fig:multi_quad">3</a>.

<figure id="fig:multi_quad">
<figure>
<img src="./figures/quad-nn-loss.png"" />
<figcaption>Training loss</figcaption>
</figure>
<figure>
<img src="./figures/quad-nn-ntk_1.png"" />
<figcaption>Largest eigenvalue of tangent kernel</figcaption>
</figure>
<figure>
<img src="./figures/quad-nn-ntk_2.png"" />
<figcaption>Second largest eigenvalue of tangent kernel</figcaption>
</figure>
<figcaption><span><strong><span>Training dynamics of NQMs for multiple examples case with different learning rates.</span></strong></span> By our analysis, two critical values are <span class="math inline">2/<em>λ</em><sub>1</sub>(0) = 0.37</span> and <span class="math inline">2/<em>λ</em><sub>2</sub>(0) = 0.39</span>. When <span class="math inline"><em>η</em> &lt; 0.37</span>, linear dynamics dominate hence the kernel is nearly constant; when <span class="math inline">0.37 &lt; <em>η</em> &lt; 0.39</span>, the catapult phase happens in <span class="math inline"><strong>p</strong><sub>1</sub></span> and only <span class="math inline"><em>λ</em><sub>1</sub>(<em>t</em>)</span> decreases; when <span class="math inline">0.39 &lt; <em>η</em> &lt; <em>η</em><sub>max</sub></span>, the catapult phase happens in <span class="math inline"><strong>p</strong><sub>1</sub></span> and <span class="math inline"><strong>p</strong><sub>2</sub></span> hence both <span class="math inline"><em>λ</em><sub>1</sub>(<em>t</em>)</span> and <span class="math inline"><em>λ</em><sub>2</sub>(<em>t</em>)</span> decreases. The experiment details can be found in Appendix <a href="#subsec:multi_quad" data-reference-type="ref" data-reference="subsec:multi_quad">19.1</a>. </figcaption>
</figure>

## Connection to general quadratic models and wide neural networks

#### General quadratic models.

As mentioned in the introduction, NQMs are contained in a general class of quadratic models of the form given in Eq. (<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>). Additionally, we show that the two-layer linear neural network analyzed in  is a special case of Eq. (<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>), and we provide a more general condition for such models to have catapult dynamics in Appendix <a href="#sec:phi_x_0" data-reference-type="ref" data-reference="sec:phi_x_0">18</a>. Furthermore, we empirically observe that a broader class of quadratic models $`g`$ can have catapult dynamics simply by letting $`\phi({\boldsymbol{x}})`$ and $`\Sigma`$ be random and assigning a small value to $`\gamma`$ (See Appendix <a href="#subsec:exp_gqm" data-reference-type="ref" data-reference="subsec:exp_gqm">19.4</a>).

#### Wide neural networks.

We have seen that NQMs, with fixed Hessian, exhibit the catapult phase phenomenon. Therefore, the change in the Hessian of wide neural networks during training is not required to produce the catapult phase. We will discuss the high-level idea of analyzing the catapult phase for a general NQM with large learning rates, and empirically show that this idea applies to neural networks. We train an NQM Eq. (<a href="#eq:nn_quad" data-reference-type="ref" data-reference="eq:nn_quad">[eq:nn_quad]</a>) $`f_{\mathrm{quad}}`$ on $`n`$ data points $`\{({\boldsymbol{x}}_i,y_i)\}_{i=1}^n\in\mathbb{R}^{d}\times \mathbb{R}`$ with GD. The dynamics equations take the following form:
``` math
\begin{aligned}
    {\mathbf{f}}_{\mathrm{quad}}(t+1)-{\mathbf{y}}&=\left(I-\eta K(t) + \underbrace{\frac{1}{2}\eta^2 G(t) \nabla {\mathbf{f}}_{\mathrm{quad}}(t)^T}_{R_{{\mathbf{f}}_{\mathrm{quad}}}(t)}\right)({\mathbf{f}}_{\mathrm{quad}}(t)-{\mathbf{y}}),\label{eq:gqm_g}\\
    K(t+1) &= K(t) - \underbrace{\frac{1}{4}\eta\left(4 G(t) \nabla{\mathbf{f}}_{\mathrm{quad}}(t)^T- \eta G(t)G(t)^T \right) }_{R_K(t)}\label{eq:gqm_k},
\end{aligned}
```
where $`G_{i,:}(t) =({\mathbf{f}}_{\mathrm{quad}}(t)-{\mathbf{y}})^T\nabla{\mathbf{f}}_{\mathrm{quad}}(t)H_f({\boldsymbol{x}}_i) \in\mathbb{R}^m`$ for $`i\in[n]`$.

In our analysis for $`f_{{\mathrm{quad}}}`$ which approximates two-layer networks in Section <a href="#subsec:multiple" data-reference-type="ref" data-reference="subsec:multiple">3.2</a>, we show that catapult dynamics occur in the top eigenspace of the tangent kernel. Specifically, we analyze the dynamics equations confined to the top eigendirection of the tangent kernel $`{\boldsymbol{p}}_1`$ (i.e, $`\Pi_1{\mathcal{L}}`$ and $`\lambda_1(t)`$). We show that $`{\boldsymbol{p}}_1^T R_{{\mathbf{f}}_{\mathrm{quad}}}{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_1^T R_K{\boldsymbol{p}}_1`$ scale with the loss and remain positive when the loss becomes large, therefore $`{\boldsymbol{p}}_1^T K{\boldsymbol{p}}_1`$ (i.e., $`\lambda_{\max}(K)`$) as well as the loss will be driven down, and consequently we yield catapult convergence.

We empirically verify catapults indeed happen in the top eigenspace of the tangent kernel for additional NQMs and wide neural networks in Appendix <a href="#subsec:exp_top_eig" data-reference-type="ref" data-reference="subsec:exp_top_eig">19.3</a>. Furthermore, a similar behaviour of top eigenvalues of the tangent kernel with the one for NQMs is observed for wide neural networks when training with different learning rates (See Figure <a href="#fig:multi_nn" data-reference-type="ref" data-reference="fig:multi_nn">5</a> in Appendix <a href="#sec:exp_add" data-reference-type="ref" data-reference="sec:exp_add">19</a>).

# Quadratic models parallel neural networks in generalization

In this section, we empirically compare the test performance of three different models considered in this paper upon varying learning rate. In particular, we consider (1) the NQM, $`f_{\mathrm{quad}}`$; (2) corresponding neural networks, $`f`$; and (3) the linear model, $`f_{\mathrm{lin}}`$.

<figure id="fig:generalization">
<figure>
<img src="./figures/FSDD-fc.png"" />
</figure>
<figure>
<img src="./figures/text.png"" />
</figure>
<figure>
<img src="./figures/mnist.png"" />
</figure>
<figure>
<img src="./figures/cifar-sgd.png"" />
</figure>
<figure>
<img src="./figures/cifar2-cnn.png"" />
</figure>
<figure>
<img src="./figures/svhn-cnn.png"" />
</figure>
<figcaption><strong>Best test loss plotted against different learning rates for <span class="math inline"><em>f</em>(<strong>w</strong>)</span>, <span class="math inline"><em>f</em><sub>lin</sub>(<strong>w</strong>)</span> and <span class="math inline"><em>f</em><sub>quad</sub>(<strong>w</strong>)</span> across a variety of datasets and network architectures.</strong></figcaption>
</figure>

We implement our experiments on 3 vision datasets: CIFAR-2 (a 2-class subset of CIFAR-10 ), MNIST , and SVHN (The Street View House Numbers) , 1 speech dataset: Free Spoken Digit dataset (FSDD)  and 1 text dataset: AG NEWS .

In all experiments, we train the models by minimizing the squared loss using standard GD/SGD with constant learning rate $`\eta`$. We report the best test loss achieved during the training process with each learning rate. Experimental details can be found in Appendix <a href="#subsec:exp_quad_setting" data-reference-type="ref" data-reference="subsec:exp_quad_setting">19.5</a>. We also report the best test accuracy in Appendix <a href="#subsec:catapult_acc" data-reference-type="ref" data-reference="subsec:catapult_acc">19.6</a>. For networks with $`3`$ layers, see Appendix <a href="#subsec:3_layer_fc" data-reference-type="ref" data-reference="subsec:3_layer_fc">19.7</a>. From the experimental results, we observe the following:

#### Sub-critical learning rates.

In accordance with our theoretical analyses, we observe that all three models have nearly identical test loss for any sub-critical learning rate. Specifically, note that as the width $`m`$ increases, $`f`$ and $`f_{{\mathrm{quad}}}`$ will transition to linearity in the ball $`B({\mathbf{w}}_0,R)`$:
``` math
\begin{aligned}
    \|f - f_{{{\mathrm{lin}}}}\| = \tilde{O}(1/\sqrt{m}),~~~ \|f_{{\mathrm{quad}}} - f_{{{\mathrm{lin}}}}\| = \tilde{O}(1/\sqrt{m}), 
\end{aligned}
```
where $`R>0`$ is a constant which is large enough to contain the optimization path with respect to sub-critical learning rates. Thus, the generalization performance of these three models will be similar when $`m`$ is large, as shown in Figure <a href="#fig:generalization" data-reference-type="ref" data-reference="fig:generalization">4</a>.

#### Super-critical learning rates.

The best test loss of both $`f({\mathbf{w}})`$ and $`f_{{\mathrm{quad}}}({\mathbf{w}})`$ is consistently smaller than the one with sub-critical learning rates, and decreases for an increasing learning rate in a range of values beyond $`{\eta_{\mathrm{crit}}}`$, which was observed for wide neural networks in .

As discussed in the introduction, with super-critical learning rates, both $`f_{{\mathrm{quad}}}`$ and $`f`$ can be observed to have catapult phase, while the loss of $`f_{{{\mathrm{lin}}}}`$ diverges. Together with the similar behaviour of $`f_{{\mathrm{quad}}}`$ and $`f`$ in generalization with super-critical learning rates, we believe NQMs are a better model to understand $`f`$ in training and testing dynamics, than the linear approximation $`f_{{{\mathrm{lin}}}}`$.

In Figure <a href="#fig:generalization" data-reference-type="ref" data-reference="fig:generalization">4</a> we report the results for networks with ReLU activation function. We also implement the experiments using networks with Tanh and Swish  activation functions, and observe the same phenomena in generalization for $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$ (See Appendix <a href="#subsec:tanh_swish" data-reference-type="ref" data-reference="subsec:tanh_swish">19.8</a>).

# Summary and Discussion

#### Summary.

In this paper, we use quadratic models as a tool to better understand optimization and generalization properties of finite width neural networks trained using large learning rates. Notably, we prove that quadratic models exhibit properties of neural networks such as the catapult phase that cannot be explained using linear models, which importantly includes linear approximations to neural networks given by the neural tangent kernel. Interestingly, we show empirically that quadratic models mimic the generalization properties of neural networks when trained with large learning rate, and that such models perform better than linearized neural networks.

#### Future directions.

As quadratic models are more analytically tractable than finite width neural networks, these models open further avenues for understanding the good performance of finite width networks in practice. In particular, one interesting direction of future work is to understand the change in the kernel corresponding to a trained quadratic model. As we showed, training a quadratic model with large learning rate causes a decrease in the eigenvalues of the neural tangent kernel, and it would be interesting to understand the properties of this changed kernel that correspond with improved generalization. Indeed, prior work  has analyzed the properties of the “after kernel” corresponding to finite width neural networks, and it would be interesting to observe whether similar properties hold for the kernel corresponding to trained quadratic models.

Another interesting avenue of research is to understand whether quadratic models can be used for representation learning. Indeed, prior work  argues that networks in the neural tangent kernel regime do not learn useful representations of data through training. As quadratic models trained with large learning rate can already exhibit nonlinear dynamics and better capture generalization properties of finite width networks, it would be interesting to understand whether such models learn useful representations of data as well.

# Acknowledgements

We thank Boris Hanin, Daniel A. Roberts and Sho Yaida for the discussion about quadratic models and catapults. A.R. is funded by the George F. Carrier fellowship at Harvard School of Engineering and Applied Sciences. We are grateful for the support from the National Science Foundation (NSF) and the Simons Foundation for the Collaboration on the Theoretical Foundations of Deep Learning (<https://deepfoundations.ai/>) through awards DMS-2031883 and \#814639 and the TILOS institute (NSF CCF-2112665). This work used NVIDIA V100 GPUs NVLINK and HDR IB (Expanse GPU) at SDSC Dell Cluster through allocation TG-CIS220009 and also, Delta system at the National Center for Supercomputing Applications through allocation bbjr-delta-gpu from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants \#2138259, \#2138286, \#2138307, \#2137603, and \#2138296.

# References

<div class="thebibliography">

Yu Bai and Jason D Lee Beyond linearization: On quadratic and higher-order approximation of wide neural networks In *International Conference on Learning Representations*, 2019. **Abstract:** Recent theoretical work has established connections between over-parametrized neural networks and linearized models governed by he Neural Tangent Kernels (NTKs). NTK theory leads to concrete convergence and generalization results, yet the empirical performance of neural networks are observed to exceed their linearized models, suggesting insufficiency of this theory. Towards closing this gap, we investigate the training of over-parametrized neural networks that are beyond the NTK regime yet still governed by the Taylor expansion of the network. We bring forward the idea of \\}emph{randomizing} the neural networks, which allows them to escape their NTK and couple with quadratic models. We show that the optimization landscape of randomized two-layer networks are nice and amenable to escaping-saddle algorithms. We prove concrete generalization and expressivity results on these randomized networks, which lead to sample complexity bounds (of learning certain simple functions) that match the NTK and can in addition be better by a dimension factor when mild distributional assumptions are present. We demonstrate that our randomization technique can be generalized systematically beyond the quadratic case, by using it to find networks that are coupled with higher-order terms in their Taylor series. (@bai2019beyond)

Nicoletta Bof, Ruggero Carli, and Luca Schenato Lyapunov theory for discrete time systems *arXiv preprint arXiv:1809.05289*, 2018. **Abstract:** In this work, we present the equivalent of many theorems available for continuous time systems. In particular, the theory is applied to Averaging Theory and Separation of time scales. In particular the proofs developed for Averaging Theory and Separation of time scales departs from those typically used in continuous time systems that are based on twice differentiable change of variables and the multiple use of the Implicit Function Theorem and Mean Value Theorem. More specifically, by constructing a suitable Lyapunov function only Lipschitz conditions are necessary. Finally, it is shown that under mild condition on the so-called "interconnection conditions" the proposed tools can guarantee semi-global exponential stability rather than the more stringent local exponential stability typically found in the literature (@bof2018lyapunov)

Lenaic Chizat, Edouard Oyallon, and Francis Bach On lazy training in differentiable programming *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** In a series of recent theoretical works, it was shown that strongly over-parameterized neural networks trained with gradient-based methods could converge exponentially fast to zero training loss, with their parameters hardly varying. In this work, we show that this "lazy training" phenomenon is not specific to over-parameterized neural networks, and is due to a choice of scaling, often implicit, that makes the model behave as its linearization around the initialization, thus yielding a model equivalent to learning with positive-definite kernels. Through a theoretical analysis, we exhibit various situations where this phenomenon arises in non-convex optimization and we provide bounds on the distance between the lazy and linearized optimization paths. Our numerical experiments bring a critical note, as we observe that the performance of commonly used non-linear deep convolutional neural networks in computer vision degrades when trained in the lazy regime. This makes it unlikely that "lazy training" is behind the many successes of neural networks in difficult high dimensional tasks. (@chizat2019lazy)

Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai Gradient descent finds global minima of deep neural networks In *International Conference on Machine Learning*, pp. 1675–1685, 2019. **Abstract:** Gradient descent finds a global minimum in training deep neural networks despite the objective function being non-convex. The current paper proves gradient descent achieves zero training loss in polynomial time for a deep over-parameterized neural network with residual connections (ResNet). Our analysis relies on the particular structure of the Gram matrix induced by the neural network architecture. This structure allows us to show the Gram matrix is stable throughout the training process and this stability implies the global optimality of the gradient descent algorithm. We further extend our analysis to deep residual convolutional neural networks and obtain a similar convergence result. (@du2018gradientdeep)

Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh Gradient descent provably optimizes over-parameterized neural networks *arXiv preprint arXiv:1810.02054*, 2018. **Abstract:** One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth. This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks. For an $m$ hidden node shallow neural network with ReLU activation and $n$ training data, we show as long as $m$ is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. Our analysis relies on the following observation: over-parameterization and random initialization jointly restrict every weight vector to be close to its initialization for all iterations, which allows us to exploit a strong convexity-like property to show that gradient descent converges at a global linear rate to the global optimum. We believe these insights are also useful in analyzing deep models and other first order methods. (@du2018gradientshallow)

Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M Roy, and Surya Ganguli Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel *Advances in Neural Information Processing Systems*, 33: 5850–5861, 2020. **Abstract:** In suitably initialized wide networks, small learning rates transform deep neural networks (DNNs) into neural tangent kernel (NTK) machines, whose training dynamics is well-approximated by a linear weight expansion of the network at initialization. Standard training, however, diverges from its linearization in ways that are poorly understood. We study the relationship between the training dynamics of nonlinear deep networks, the geometry of the loss landscape, and the time evolution of a data-dependent NTK. We do so through a large-scale phenomenological analysis of training, synthesizing diverse measures characterizing loss landscape geometry and NTK dynamics. In multiple neural architectures and datasets, we find these diverse measures evolve in a highly correlated manner, revealing a universal picture of the deep learning process. In this picture, deep network training exhibits a highly chaotic rapid initial transient that within 2 to 3 epochs determines the final linearly connected basin of low loss containing the end point of training. During this chaotic transient, the NTK changes rapidly, learning useful features from the training data that enables it to outperform the standard initial NTK by a factor of 3 in less than 3 to 4 epochs. After this rapid chaotic transient, the NTK changes at constant velocity, and its performance matches that of full network training in 15% to 45% of training time. Overall, our analysis reveals a striking correlation between a diverse set of metrics over training time, governed by a rapid chaotic to stable transition in the first few epochs, that together poses challenges and opportunities for the development of more accurate theories of deep learning. (@fort2020deep)

Antonio Gulli <http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>. **Abstract:** Text classification plays a crucial role in organizing and understanding huge amounts of text data. However, traditional text classification methods often face challenges when dealing with unseen or novel classes. Zero-shot learning (ZSL) offers a promising solution to this problem by enabling the classification of text instances into classes that have not been encountered during training. There is a plethora of potential benefits of ZSL in several applications, emphasizing its ability to handle new classes and adapt to evolving domains. In this paper, we have used the AG news dataset which is a commonly used benchmark dataset for text classification tasks. It consists of news articles from the AG’s corpus, collected from four different categories: World, Sports, Business, and Science/Technology. Each article is assigned a label corresponding to one of these categories. We applied state-of-the-art deep learning algorithms such as Convolutional Neural Networks and Recurrent Neural Networks to compare the performance with Zero Shot Learning (ZSL). ZSL proved to be robust and performed better compared to the other algorithms in terms of accuracy and F1 Score. (@agnews)

Jiaoyang Huang and Horng-Tzer Yau Dynamics of deep neural networks and neural tangent hierarchy In *International Conference on Machine Learning*, pp. 4542–4551. PMLR, 2020. **Abstract:** The evolution of a deep neural network trained by the gradient descent can be described by its neural tangent kernel (NTK) as introduced in \[20\], where it was proven that in the infinite width limit the NTK converges to an explicit limiting kernel and it stays constant during training. The NTK was also implicit in some other recent papers \[6,13,14\]. In the overparametrization regime, a fully-trained deep neural network is indeed equivalent to the kernel regression predictor using the limiting NTK. And the gradient descent achieves zero training loss for a deep overparameterized neural network. However, it was observed in \[5\] that there is a performance gap between the kernel regression using the limiting NTK and the deep neural networks. This performance gap is likely to originate from the change of the NTK along training due to the finite width effect. The change of the NTK along the training is central to describe the generalization features of deep neural networks. In the current paper, we study the dynamic of the NTK for finite width deep fully-connected neural networks. We derive an infinite hierarchy of ordinary differential equations, the neural tangent hierarchy (NTH) which captures the gradient descent dynamic of the deep neural network. Moreover, under certain conditions on the neural network width and the data set dimension, we prove that the truncated hierarchy of NTH approximates the dynamic of the NTK up to arbitrary precision. This description makes it possible to directly study the change of the NTK for deep neural networks, and sheds light on the observation that deep neural networks outperform kernel regressions using the corresponding limiting NTK. (@huang2020dynamics)

Arthur Jacot, Franck Gabriel, and Clément Hongler Neural tangent kernel: Convergence and generalization in neural networks In *Advances in neural information processing systems*, pp. 8571–8580, 2018. **Abstract:** At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods. We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function $f\_\\}theta$ (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK). This kernel is central to describe the generalization features of ANNs. While the NTK is random at initialization and varies during training, in the infinite-width limit it converges to an explicit limiting kernel and it stays constant during training. This makes it possible to study the training of ANNs in function space instead of parameter space. Convergence of the training can then be related to the positive-definiteness of the limiting NTK. We prove the positive-definiteness of the limiting NTK when the data is supported on the sphere and the non-linearity is non-polynomial. We then focus on the setting of least-squares regression and show that in the infinite-width limit, the network function $f\_\\}theta$ follows a linear differential equation during training. The convergence is fastest along the largest kernel principal components of the input data with respect to the NTK, hence suggesting a theoretical motivation for early stopping. Finally we study the NTK numerically, observe its behavior for wide networks, and compare it to the infinite-width limit. (@jacot2018neural)

Jakobovski <https://github.com/Jakobovski/free-spoken-digit-dataset>. **Abstract:** Recognition of spoken words is one of the fields that attract researchers due to its importance. It involves in various applications as many applications can be fed with data in voice format. One of vocal recognition fields is Spoken Digit Recognition (SDR) which aims to translate the input spoken audio file to its relative numerical value. Machine Learning (ML) and Deep Learning (DL) are used in various digital signal processing applications due to their ability to extract useful features from input data which enhances the accuracy of the target models as well as their ability to adapt with wide range of applications. This paper proposes two approaches to tackle the problem of spoken digits recognition. First approach is to recognize the spoken digits using wavelet time scattering and Support Vector Machine (SVM) classifier. While the second aims to solve this problem using Mel-frequency spectrograms and Deep Convolutional Neural Networks (DCNN). Experiments are performed with Free Spoken Digit Dataset (FSDD) as a training and testing dataset. Although the second approach outperforms the first one by an average increase of 1.1% in terms of accuracy metric, results shows that both approaches are suitable for such an application. (@fsdd)

Ziwei Ji and Matus Telgarsky Polylogarithmic width suffices for gradient descent to achieve arbitrarily small test error with shallow relu networks In *International Conference on Learning Representations*, 2019. **Abstract:** Recent work has revealed that overparameterized networks trained by gradient descent achieve arbitrarily low training error, and sometimes even low test error. The required width, however, is always polynomial in at least one of the sample size n, the (inverse) training error 1/epsilon, and the (inverse) failure probability 1/delta. This work shows that O(1/epsilon) iterations of gradient descent on two-layer networks of any width exceeding polylog(n, 1/epsilon, 1/delta) and Omega(1/epsilon^2) training examples suffices to achieve a test error of epsilon. The analysis further relies upon a margin property of the limiting kernel, which is guaranteed positive, and can distinguish between true labels and random labels. (@ji2019polylogarithmic)

Alex Krizhevsky, Geoffrey Hinton, et al Learning multiple layers of features from tiny images . **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@krizhevsky2009learning)

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner Gradient-based learning applied to document recognition *Proceedings of the IEEE*, 86 (11): 2278–2324, 1998. **Abstract:** Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day. (@lecun1998gradient)

Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington Wide neural networks of any depth evolve as linear models under gradient descent In *Advances in neural information processing systems*, pp. 8570–8581, 2019. **Abstract:** A longstanding goal in deep learning research has been to precisely characterize training and generalization. However, the often complex loss landscapes of neural networks have made a theory of learning dynamics elusive. In this work, we show that for wide neural networks the learning dynamics simplify considerably and that, in the infinite width limit, they are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters. Furthermore, mirroring the correspondence between wide Bayesian neural networks and Gaussian processes, gradient-based training of wide neural networks with a squared loss produces test set predictions drawn from a Gaussian process with a particular compositional kernel. While these theoretical results are only exact in the infinite width limit, we nevertheless find excellent empirical agreement between the predictions of the original network and those of the linearized version even for finite practically-sized networks. This agreement is robust across different architectures, optimization methods, and loss functions. (@lee2019wide)

Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, and Guy Gur-Ari The large learning rate phase of deep learning: the catapult mechanism *arXiv preprint arXiv:2003.02218*, 2020. **Abstract:** The choice of initial learning rate can have a profound effect on the performance of deep networks. We present a class of neural networks with solvable training dynamics, and confirm their predictions empirically in practical deep learning settings. The networks exhibit sharply distinct behaviors at small and large learning rates. The two regimes are separated by a phase transition. In the small learning rate phase, training can be understood using the existing theory of infinitely wide neural networks. At large learning rates the model captures qualitatively distinct phenomena, including the convergence of gradient descent dynamics to flatter minima. One key prediction of our model is a narrow range of large, stable learning rates. We find good agreement between our model’s predictions and training dynamics in realistic deep learning settings. Furthermore, we find that the optimal performance in such settings is often found in the large learning rate phase. We believe our results shed light on characteristics of models trained at different learning rates. In particular, they fill a gap between existing wide neural network theory, and the nonlinear, large learning rate, training dynamics relevant to practice. (@lewkowycz2020large)

Chaoyue Liu, Libin Zhu, and Mikhail Belkin On the linearity of large non-linear models: when and why the tangent kernel is constant *Advances in Neural Information Processing Systems*, 33, 2020. **Abstract:** The goal of this work is to shed light on the remarkable phenomenon of transition to linearity of certain neural networks as their width approaches infinity. We show that the transition to linearity of the model and, equivalently, constancy of the (neural) tangent kernel (NTK) result from the scaling properties of the norm of the Hessian matrix of the network as a function of the network width. We present a general framework for understanding the constancy of the tangent kernel via Hessian scaling applicable to the standard classes of neural networks. Our analysis provides a new perspective on the phenomenon of constant tangent kernel, which is different from the widely accepted "lazy training". Furthermore, we show that the transition to linearity is not a general property of wide neural networks and does not hold when the last layer of the network is non-linear. It is also not necessary for successful optimization by gradient descent. (@liu2020linearity)

Philip M Long Properties of the after kernel *arXiv preprint arXiv:2105.10585*, 2021. **Abstract:** The Neural Tangent Kernel (NTK) is the wide-network limit of a kernel defined using neural networks at initialization, whose embedding is the gradient of the output of the network with respect to its parameters. We study the "after kernel", which is defined using the same embedding, except after training, for neural networks with standard architectures, on binary classification problems extracted from MNIST and CIFAR-10, trained using SGD in a standard way. For some dataset-architecture pairs, after a few epochs of neural network training, a hard-margin SVM using the network’s after kernel is much more accurate than when the network’s initial kernel is used. For networks with an architecture similar to VGG, the after kernel is more "global", in the sense that it is less invariant to transformations of input images that disrupt the global structure of the image while leaving the local statistics largely intact. For fully connected networks, the after kernel is less global in this sense. The after kernel tends to be more invariant to small shifts, rotations and zooms; data augmentation does not improve these invariances. The (finite approximation to the) conjugate kernel, obtained using the last layer of hidden nodes, sometimes, but not always, provides a good approximation to the NTK and the after kernel. Training a network with a larger learning rate (while holding the training error constant) produces a better kernel, as measured by the test error of a hard-margin SVM. The after kernels of networks trained with larger learning rates tend to be more global, and more invariant to small shifts, rotations and zooms. (@long2021properties)

David Meltzer and Junyu Liu Catapult dynamics and phase transitions in quadratic nets *arXiv preprint arXiv:2301.07737*, 2023. **Abstract:** Neural networks trained with gradient descent can undergo non-trivial phase transitions as a function of the learning rate. In (Lewkowycz et al., 2020) it was discovered that wide neural nets can exhibit a catapult phase for super-critical learning rates, where the training loss grows exponentially quickly at early times before rapidly decreasing to a small value. During this phase the top eigenvalue of the neural tangent kernel (NTK) also undergoes significant evolution. In this work, we will prove that the catapult phase exists in a large class of models, including quadratic models and two-layer, homogenous neural nets. To do this, we show that for a certain range of learning rates the weight norm decreases whenever the loss becomes large. We also empirically study learning rates beyond this theoretically derived range and show that the activation map of ReLU nets trained with super-critical learning rates becomes increasingly sparse as we increase the learning rate. (@meltzer2023catapult)

Andrea Montanari and Yiqiao Zhong The interpolation phase transition in neural networks: Memorization and generalization under lazy training *arXiv preprint arXiv:2007.12826*, 2020. **Abstract:** Modern neural networks are often operated in a strongly overparametrized regime: they comprise so many parameters that they can interpolate the training set, even if actual labels are replaced by purely random ones. Despite this, they achieve good prediction error on unseen data: interpolating the training set does not lead to a large generalization error. Further, overparametrization appears to be beneficial in that it simplifies the optimization landscape. Here we study these phenomena in the context of two-layers neural networks in the neural tangent (NT) regime. We consider a simple data model, with isotropic covariates vectors in $d$ dimensions, and $N$ hidden neurons. We assume that both the sample size $n$ and the dimension $d$ are large, and they are polynomially related. Our first main result is a characterization of the eigenstructure of the empirical NT kernel in the overparametrized regime $Nd\\}gg n$. This characterization implies as a corollary that the minimum eigenvalue of the empirical NT kernel is bounded away from zero as soon as $Nd\\}gg n$, and therefore the network can exactly interpolate arbitrary labels in the same regime. Our second main result is a characterization of the generalization error of NT ridge regression including, as a special case, min-$\\}ell_2$ norm interpolation. We prove that, as soon as $Nd\\}gg n$, the test error is well approximated by the one of kernel ridge regression with respect to the infinite-width kernel. The latter is in turn well approximated by the error of polynomial ridge regression, whereby the regularization parameter is increased by a ‘self-induced’ term related to the high-degree components of the activation function. The polynomial degree depends on the sample size and the dimension (in particular on $\\}log n/\\}log d$). (@montanari2020interpolation)

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng Reading digits in natural images with unsupervised feature learning . **Abstract:** Detecting and reading text from natural images is a hard computer vision task that is central to a variety of emerging applications. Related problems like document character recognition have been widely studied by computer vision and machine learning researchers and are virtually solved for practical applications like reading handwritten digits. Reliably recognizing characters in more complex scenes like photographs, however, is far more difficult: the best existing methods lag well behind human performance on the same tasks. In this paper we attack the problem of recognizing digits in a real application using unsupervised feature learning methods: reading house numbers from street level photos. To this end, we introduce a new benchmark dataset for research use containing over 600,000 labeled digits cropped from Street View images. We then demonstrate the difficulty of recognizing these digits when the problem is approached with hand-designed features. Finally, we employ variants of two recently proposed unsupervised feature learning methods and find that they are convincingly superior on our benchmarks. (@netzer2011reading)

Eshaan Nichani, Adityanarayanan Radhakrishnan, and Caroline Uhler Increasing depth leads to U-shaped test risk in over-parameterized convolutional networks In *International Conference on Machine Learning Workshop on Over-parameterization: Pitfalls and Opportunities*, 2021. **Abstract:** Recent works have demonstrated that increasing model capacity through width in over-parameterized neural networks leads to a decrease in test risk. For neural networks, however, model capacity can also be increased through depth, yet understanding the impact of increasing depth on test risk remains an open question. In this work, we demonstrate that the test risk of over-parameterized convolutional networks is a U-shaped curve (i.e. monotonically decreasing, then increasing) with increasing depth. We first provide empirical evidence for this phenomenon via image classification experiments using both ResNets and the convolutional neural tangent kernel (CNTK). We then present a novel linear regression framework for characterizing the impact of depth on test risk, and show that increasing depth leads to a U-shaped test risk for the linear CNTK. In particular, we prove that the linear CNTK corresponds to a depth-dependent linear transformation on the original space and characterize properties of this transformation. We then analyze over-parameterized linear regression under arbitrary linear transformations and, in simplified settings, provably identify the depths which minimize each of the bias and variance terms of the test risk. (@nichani2020increasing)

Guillermo Ortiz-Jiménez, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard What can linearized neural networks actually say about generalization? *Advances in Neural Information Processing Systems*, 34, 2021. **Abstract:** For certain infinitely-wide neural networks, the neural tangent kernel (NTK) theory fully characterizes generalization, but for the networks used in practice, the empirical NTK only provides a rough first-order approximation. Still, a growing body of work keeps leveraging this approximation to successfully analyze important deep learning phenomena and design algorithms for new applications. In our work, we provide strong empirical evidence to determine the practical validity of such approximation by conducting a systematic comparison of the behavior of different neural networks and their linear approximations on different tasks. We show that the linear approximations can indeed rank the learning complexity of certain tasks for neural networks, even when they achieve very different performances. However, in contrast to what was previously reported, we discover that neural networks do not always perform better than their kernel approximations, and reveal that the performance gap heavily depends on architecture, dataset size and training task. We discover that networks overfit to these tasks mostly due to the evolution of their kernel during training, thus, revealing a new type of implicit bias. (@ortiz2021can)

Boris T Polyak Introduction to optimization *Optimization Software, Inc, New York*, 1987. **Abstract:** Preface. MATHEMATICAL REVIEW. Methods of Proof and Some Notation. Vector Spaces and Matrices. Transformations. Concepts from Geometry. Elements of Calculus. UNCONSTRAINED OPTIMIZATION. Basics of Set–Constrained and Unconstrained Optimization. One–Dimensional Search Methods. Gradient Methods. Newton’s Method. Conjugate Direction Methods. Quasi–Newton Methods. Solving Ax = b. Unconstrained Optimization and Neural Networks. Genetic Algorithms. LINEAR PROGRAMMING. Introduction to Linear Programming. Simplex Method. Duality. Non–Simplex Methods. NONLINEAR CONSTRAINED OPTIMIZATION. Problems with Equality Constraints. Problems with Inequality Constraints. Convex Optimization Problems. Algorithms for Constrained Optimization. References. Index. (@polyakintroduction)

Prajit Ramachandran, Barret Zoph, and Quoc V Le Searching for activation functions *arXiv preprint arXiv:1710.05941*, 2017. **Abstract:** The choice of activation functions in deep networks has a significant effect on the training dynamics and task performance. Currently, the most successful and widely-used activation function is the Rectified Linear Unit (ReLU). Although various hand-designed alternatives to ReLU have been proposed, none have managed to replace it due to inconsistent gains. In this work, we propose to leverage automatic search techniques to discover new activation functions. Using a combination of exhaustive and reinforcement learning-based search, we discover multiple novel activation functions. We verify the effectiveness of the searches by conducting an empirical evaluation with the best discovered activation function. Our experiments show that the best discovered activation function, $f(x) = x \\}cdot \\}text{sigmoid}(\\}beta x)$, which we name Swish, tends to work better than ReLU on deeper models across a number of challenging datasets. For example, simply replacing ReLUs with Swish units improves top-1 classification accuracy on ImageNet by 0.9\\}% for Mobile NASNet-A and 0.6\\}% for Inception-ResNet-v2. The simplicity of Swish and its similarity to ReLU make it easy for practitioners to replace ReLUs with Swish units in any neural network. (@ramachandran2017searching)

Daniel A Roberts, Sho Yaida, and Boris Hanin *The principles of deep learning theory* Cambridge University Press, 2022. **Abstract:** This textbook establishes a theoretical framework for understanding deep learning models of practical relevance. With an approach that borrows from theoretical physics, Roberts and Yaida provide clear and pedagogical explanations of how realistic deep neural networks actually work. To make results from the theoretical forefront accessible, the authors eschew the subject’s traditional emphasis on intimidating formality without sacrificing accuracy. Straightforward and approachable, this volume balances detailed first-principle derivations of novel results with insight and intuition for theorists and practitioners alike. This self-contained textbook is ideal for students and researchers interested in artificial intelligence with minimal prerequisites of linear algebra, calculus, and informal probability theory, and it can easily fill a semester-long course on deep learning theory. For the first time, the exciting practical advances in modern artificial intelligence capabilities can be matched with a set of effective principles, providing a timeless blueprint for theoretical research in deep learning. (@roberts2022principles)

Pedro Savarese, Itay Evron, Daniel Soudry, and Nathan Srebro How do infinite width bounded norm networks look in function space? In *Conference on Learning Theory*, pp. 2667–2690. PMLR, 2019. **Abstract:** We consider the question of what functions can be captured by ReLU networks with an unbounded number of units (infinite width), but where the overall network Euclidean norm (sum of squares of all weights in the system, except for an unregularized bias term for each unit) is bounded; or equivalently what is the minimal norm required to approximate a given function. For functions $f : \\}mathbb R \\}rightarrow \\}mathbb R$ and a single hidden layer, we show that the minimal network norm for representing $f$ is $\\}max(\\}int \|f”(x)\| dx, \|f’(-\\}infty) + f’(+\\}infty)\|)$, and hence the minimal norm fit for a sample is given by a linear spline interpolation. (@savarese2019infinite)

Francis Williams, Matthew Trager, Daniele Panozzo, Claudio Silva, Denis Zorin, and Joan Bruna Gradient dynamics of shallow univariate relu networks *Advances in Neural Information Processing Systems*, 32, 2019. **Abstract:** We present a theoretical and empirical study of the gradient dynamics of overparameterized shallow ReLU networks with one-dimensional input, solving least-squares interpolation. We show that the gradient dynamics of such networks are determined by the gradient flow in a non-redundant parameterization of the network function. We examine the principal qualitative features of this gradient flow. In particular, we determine conditions for two learning regimes:kernel and adaptive, which depend both on the relative magnitude of initialization of weights in different layers and the asymptotic behavior of initialization coefficients in the limit of large network widths. We show that learning in the kernel regime yields smooth interpolants, minimizing curvature, and reduces to cubic splines for uniform initializations. Learning in the adaptive regime favors instead linear splines, where knots cluster adaptively at the sample points. (@williams2019gradient)

Greg Yang and Edward J Hu Feature learning in infinite-width neural networks *arXiv preprint arXiv:2011.14522*, 2020. **Abstract:** As its width tends to infinity, a deep neural network’s behavior under gradient descent can become simplified and predictable (e.g. given by the Neural Tangent Kernel (NTK)), if it is parametrized appropriately (e.g. the NTK parametrization). However, we show that the standard and NTK parametrizations of a neural network do not admit infinite-width limits that can learn features, which is crucial for pretraining and transfer learning such as with BERT. We propose simple modifications to the standard parametrization to allow for feature learning in the limit. Using the \*Tensor Programs\* technique, we derive explicit formulas for such limits. On Word2Vec and few-shot learning on Omniglot via MAML, two canonical tasks that rely crucially on feature learning, we compute these limits exactly. We find that they outperform both NTK baselines and finite-width networks, with the latter approaching the infinite-width feature learning performance as width increases. More generally, we classify a natural space of neural network parametrizations that generalizes standard, NTK, and Mean Field parametrizations. We show 1) any parametrization in this space either admits feature learning or has an infinite-width training dynamics given by kernel gradient descent, but not both; 2) any such infinite-width limit can be computed using the Tensor Programs technique. Code for our experiments can be found at github.com/edwardjhu/TP4. (@yang2020feature)

Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George Dahl, Chris Shallue, and Roger B Grosse Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model *Advances in neural information processing systems*, 32, 2019. **Abstract:** Increasing the batch size is a popular way to speed up neural network training, but beyond some critical batch size, larger batch sizes yield diminishing returns. In this work, we study how the critical batch size changes based on properties of the optimization algorithm, including acceleration and preconditioning, through two different lenses: large scale experiments and analysis using a simple noisy quadratic model (NQM). We experimentally demonstrate that optimization algorithms that employ preconditioning, specifically Adam and K-FAC, result in much larger critical batch sizes than stochastic gradient descent with momentum. We also demonstrate that the NQM captures many of the essential features of real neural network training, despite being drastically simpler to work with. The NQM predicts our results with preconditioned optimizers, previous results with accelerated gradient descent, and other results around optimal learning rates and large batch training, making it a useful tool to generate testable predictions about neural network optimization. We demonstrate empirically that the simple noisy quadratic model (NQM) displays many similarities to neural networks in terms of large-batch training. We prove analytical convergence results for the NQM model that predict such behavior and hence provide possible explanations and a better understanding for many large-batch training phenomena. (@zhang2019algorithmic)

Difan Zou and Quanquan Gu An improved analysis of training over-parameterized deep neural networks In *Advances in Neural Information Processing Systems*, pp. 2053–2062, 2019. **Abstract:** A recent line of research has shown that gradient-based algorithms with random initialization can converge to the global minima of the training loss for over-parameterized (i.e., sufficiently wide) deep neural networks. However, the condition on the width of the neural network to ensure the global convergence is very stringent, which is often a high-degree polynomial in the training sample size $n$ (e.g., $O(n^{24})$). In this paper, we provide an improved analysis of the global convergence of (stochastic) gradient descent for training deep neural networks, which only requires a milder over-parameterization condition than previous work in terms of the training sample size and other problem-dependent parameters. The main technical contributions of our analysis include (a) a tighter gradient lower bound that leads to a faster convergence of the algorithm, and (b) a sharper characterization of the trajectory length of the algorithm. By specializing our result to two-layer (i.e., one-hidden-layer) neural networks, it also provides a milder over-parameterization condition than the best-known result in prior work. (@zou2019improved)

</div>

# Appendix

# Derivation of NQM

We will derive the NQM that approximate the two-layer fully connected ReLU activated neural networks based on Eq. (<a href="#eq:nn_quad" data-reference-type="ref" data-reference="eq:nn_quad">[eq:nn_quad]</a>).

The first derivative of $`f`$ can be computed by:
``` math
\begin{aligned}
    \frac{\partial f}{\partial {\mathbf{u}}_i} = \frac{1}{\sqrt{md}}v_i \mathbbm{1}_{\left\{{\mathbf{u}}_i^T {\boldsymbol{x}}\geq 0\right\}}{\boldsymbol{x}}^T, ~~~~\frac{\partial f}{\partial v_i} =\frac{1}{\sqrt{m}}\sigma\left(\frac{1}{\sqrt{d}}{\mathbf{u}}_i^T {\boldsymbol{x}}\right),~~~ \forall i\in[m].
\end{aligned}
```

And each entry of the Hessian of $`f`$, i.e., $`H_f`$, can be computed by
``` math
\begin{aligned}
    \frac{\partial^2 f}{\partial {\mathbf{u}}_i^2} =\mathbf{0},~~\frac{\partial^2 f}{\partial v_i^2} =0,~~~\frac{\partial^2 f}{\partial {\mathbf{u}}_i v_i} = \frac{1}{\sqrt{md}} \mathbbm{1}_{\left\{{\mathbf{u}}_i^T {\boldsymbol{x}}\geq 0\right\}}{\boldsymbol{x}}^T,~~~~\forall i\in[m].
\end{aligned}
```

Now we get $`f_{{\mathrm{quad}}}`$ taking the following form
``` math
\begin{aligned}
    \mathbf{NQM:}~~f_{\mathrm{quad}}({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) 
    &= f({\mathbf{u}}_0,{\mathbf{v}}_0;{\boldsymbol{x}}) + \frac{1}{\sqrt{md}}\sum_{i=1}^m ({\mathbf{u}}_i - {\mathbf{u}}_{0,i})^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}v_{0,i}  +\frac{1}{\sqrt{m}}\sum_{i=1}^m (v_i - v_{0,i}) \sigma\left(\frac{1}{\sqrt{d}}{\mathbf{u}}_{0,i}^T{\boldsymbol{x}}\right) \nonumber\\
    &~~~~+ \frac{1}{\sqrt{md}}\sum_{i=1}^m ({\mathbf{u}}_i - {\mathbf{u}}_{0,i})^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}(v_i-v_{0,i}).
\end{aligned}
```

# Derivation of dynamics equations

For simplicity of notation, we denote $`f_{\mathrm{quad}}`$ by $`g`$. Note that at initialization, the first and second derivatives of $`f`$ with respect to parameters are the same as those of $`g`$.

## Single training example

The NQM can be equivalently written as:
``` math
\begin{aligned}
    g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) &= g({\mathbf{u}}_0,{\mathbf{v}}_0;{\boldsymbol{x}}) + \left<{\mathbf{u}}- {\mathbf{u}}_0, \eval{\nabla_{{\mathbf{u}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}\right> + \left<{\mathbf{v}}- {\mathbf{v}}_0, \eval{\nabla_{{\mathbf{v}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}\right>\\
    &~~~~+ \left<{\mathbf{u}}- {\mathbf{u}}_0, \eval{\frac{\partial^2 g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}{\partial {\mathbf{u}}\partial {\mathbf{v}}}}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0} ({\mathbf{v}}-{\mathbf{v}}_0)\right>,
\end{aligned}
```
since $`\frac{\partial^2 g}{\partial {\mathbf{u}}^2} = 0`$ and $`\frac{\partial^2 g}{\partial {\mathbf{v}}^2} = 0`$.

And the tangent kernel $`\lambda({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})`$ takes the form
``` math
\begin{aligned}
    \lambda({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}}) &= \left\|\eval{\nabla_{{\mathbf{u}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}+\eval{\frac{\partial^2 g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}{\partial {\mathbf{u}}\partial {\mathbf{v}}}}_{{\mathbf{u}}= {\mathbf{u}}_0}({\mathbf{v}}-{\mathbf{v}}_0)\right\|_F^2\\
    &~~~~+ \left\|\eval{\nabla_{{\mathbf{v}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0} + ({\mathbf{u}}-{\mathbf{u}}_0)^T\eval{\frac{\partial^2 g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}{\partial {\mathbf{u}}\partial {\mathbf{v}}}}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}\right\|^2. 
\end{aligned}
```

Here
``` math
\begin{aligned}
   \eval{\nabla_{{\mathbf{u}}_i}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0} &= \frac{1}{\sqrt{md}}\sum_{i=1}^m v_{0,i} \mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}{\boldsymbol{x}}, ~~~\forall i\in[m],\\
    \eval{\nabla_{{\mathbf{v}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0} &=\frac{1}{\sqrt{md}}\sigma\left({\mathbf{u}}_0^T{\boldsymbol{x}}\right).
\end{aligned}
```

In the following, we will consider the dynamics of $`g`$ and $`\lambda`$ with GD, hence for simplicity of notations, we denote
``` math
\begin{aligned}
    \nabla_{\mathbf{u}}g(0) &:= \eval{\nabla_{{\mathbf{u}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0},\\
    \nabla_{\mathbf{v}}g(0) &:= \eval{\nabla_{{\mathbf{v}}}g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0},\\
    \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} &:=\eval{\frac{\partial^2 g({\mathbf{u}},{\mathbf{v}};{\boldsymbol{x}})}{\partial {\mathbf{u}}\partial {\mathbf{v}}}}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}.
\end{aligned}
```

By gradient descent with learning rate $`\eta`$, at iteration $`t`$, we have the update equations for weights $`{\mathbf{u}}`$ and $`{\mathbf{v}}`$:
``` math
\begin{aligned}
    {\mathbf{u}}(t+1) &= {\mathbf{u}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),\\
    {\mathbf{v}}(t+1) &= {\mathbf{v}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right).
\end{aligned}
```

Then we plug them in the expression of $`\lambda(t+1)`$ and we get
``` math
\begin{aligned}
    \lambda(t+1) &= \left\|\nabla_{{\mathbf{u}}}g(0)+\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t+1)-{\mathbf{v}}(0))\right\|_F^2+ \left\|\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t+1)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right\|^2\\
    &= \left\|\nabla_{{\mathbf{u}}}g(0)+\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left({\mathbf{v}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)-{\mathbf{v}}(0)\right)\right\|_F^2\\
    &~~~~+ \left\|\nabla_{{\mathbf{v}}}g(0) + \left({\mathbf{u}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)-{\mathbf{u}}(0)\right)^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right\|^2 \\
    &= \lambda(t) + \eta^2(g(t)-y)^2\left\|\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right\|_F^2 \\
    &~~~~+\eta^2(g(t)-y)^2\left\|\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} \right\|^2\\
    &~~~~-2\eta(g(t)-y) \left< \nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)), \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right>\\
    &~~~~-2\eta(g(t)-y) \left<\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}, \left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right>.
\end{aligned}
```

Due to the structure of $`\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}`$, we have
``` math
\begin{aligned}
   \left\|\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right\|_F^2  &= \frac{\|{\boldsymbol{x}}\|^2}{md} \left\|\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right\|^2\\
   &= \frac{\|{\boldsymbol{x}}\|^2}{md}\left\|\nabla_{{\mathbf{v}}}g(t)\right\|^2,
\end{aligned}
```
and
``` math
\begin{aligned}
    \left\|\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} \right\|^2 &=\frac{\|{\boldsymbol{x}}\|^2}{md} \left\|\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right\|_F^2\\
    &= \frac{\|{\boldsymbol{x}}\|^2}{md}\left\|\nabla_{{\mathbf{u}}}g(t)\right\|_F^2.
\end{aligned}
```

Furthermore,
``` math
\begin{aligned}
    &\left< \nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)), \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right>\\
    &~~= \frac{\|{\boldsymbol{x}}\|^2}{md}\left<{\mathbf{v}}(t)-{\mathbf{v}}(0),\nabla_{\mathbf{v}}g(0)\right> + \frac{\|{\boldsymbol{x}}\|^2}{md}\left<\nabla_{\mathbf{u}}g(0), {\mathbf{u}}(t)-{\mathbf{u}}(0)\right> + \left<\nabla_{\mathbf{u}}g(0), \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\nabla_{{\mathbf{v}}}g(0)\right> \\
    &~~~~+ \left<\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)),\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right> \\
    &~~= \frac{\|{\boldsymbol{x}}\|^2}{md}\left<{\mathbf{v}}(t)-{\mathbf{v}}(0),\nabla_{\mathbf{v}}g(0)\right> + \frac{\|{\boldsymbol{x}}\|^2}{md}\left<\nabla_{\mathbf{u}}g(0), {\mathbf{u}}(t)-{\mathbf{u}}(0)\right> + g(0)+ \frac{\|{\boldsymbol{x}}\|^2}{md}\left<{\mathbf{v}}(t)-{\mathbf{v}}(0),\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\right> \\
    &~~=g(t){\|{\boldsymbol{x}}\|^2}/{md} .
\end{aligned}
```

Similarly, we have
``` math
\begin{aligned}
     \left<\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}, \left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right> = g(t){\|{\boldsymbol{x}}\|^2}/{md} .
\end{aligned}
```

As a result,
``` math
\begin{aligned}
     \lambda(t+1) &= \lambda(t) + \frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)^2\lambda(t) - \frac{4\|{\boldsymbol{x}}\|^2}{md}\eta(g(t)-y)g(t)\\
     &= \lambda(t) +\eta\frac{\|{\boldsymbol{x}}\|^2}{md}(g(t)-y)^2\left(\eta\lambda(t) - 4\frac{g(t)}{g(t)-y}\right).
\end{aligned}
```

For $`g`$, we plug the update equations for $`{\mathbf{u}}`$ and $`{\mathbf{v}}`$ in the expression of $`g(t+1)`$ and we can get
``` math
\begin{aligned}
    g(t+1) &= g(0) + \left<{\mathbf{u}}(t+1)-{\mathbf{u}}(0), \nabla_{\mathbf{u}}g(0)\right> + \left<{\mathbf{v}}(t+1)-{\mathbf{v}}(0),\nabla_{\mathbf{v}}g(0)\right>\\
    &~~~+ \left<{\mathbf{u}}(t+1)-{\mathbf{u}}(0), \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t+1)-{\mathbf{v}}(0)\right>\\
    &= g(0) + \left<{\mathbf{u}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right) - {\mathbf{u}}(0), \nabla_{\mathbf{u}}g(0)\right>\\
    &~~~+ \left<{\mathbf{v}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)-{\mathbf{v}}(0), \nabla_{\mathbf{v}}g(0)\right> \\
    &~~~+ \left\langle {\mathbf{u}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right) - {\mathbf{u}}(0)\right.,\\
    &~~~~~~\left.\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left({\mathbf{v}}(t) - \eta(g(t)-y)\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)-{\mathbf{v}}(0) \right)\right\rangle\\
    &= g(t) - \eta(g(t)-y)\left<\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)),\nabla_{\mathbf{u}}g(0) \right>\\
    &~~~-\eta(g(t)-y) \left<\nabla_{{\mathbf{v}}}g(0) +({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}},\nabla_{\mathbf{v}}g(0)\right>\\
    &~~~+\eta^2(g(t)-y)^2\left<\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)),\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right) \right>\\
    &~~~-\eta(g(t)-y)\left<{\mathbf{u}}(t)-{\mathbf{u}}(0),\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right) \right>\\
    &~~~~-\eta(g(t)-y)\left<\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)), \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right> \\
    &= g(t) - \eta(g(t)-y)\lambda(t) \\
    &~~~+\eta^2(g(t)-y)^2\left<\nabla_{{\mathbf{u}}}g(0) + \frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)),\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\left(\nabla_{{\mathbf{v}}}g(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right) \right>\\
    &= g(t) - \eta(g(t)-y)\lambda(t) + \frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)^2 g(t)
\end{aligned}
```
Therefore,
``` math
\begin{aligned}
    g(t+1)- y = \left(1-\eta\lambda(t) + \frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)g(t)\right)(g(t)-y).
\end{aligned}
```

## Multiple training examples

We follow the similar notation on the first and second order derivative of $`g`$ with Appendix <a href="#subsec:deri:single" data-reference-type="ref" data-reference="subsec:deri:single">7.1</a>. Specifically, for $`k\in[n]`$, we denote
``` math
\begin{aligned}
    \nabla_{\mathbf{u}}g_k(0) &:= \eval{\nabla_{{\mathbf{u}}}g({\mathbf{u}},{\mathbf{v}};x_k)}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0},\\
    \nabla_{\mathbf{v}}g_k(0) &:= \eval{\nabla_{{\mathbf{v}}}g({\mathbf{u}},{\mathbf{v}};x_k)}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0},\\
    \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} &:=\eval{\frac{\partial^2 g({\mathbf{u}},{\mathbf{v}};x_k)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}}_{{\mathbf{u}}= {\mathbf{u}}_0,{\mathbf{v}}= {\mathbf{v}}_0}.
\end{aligned}
```

By GD with learning rate $`\eta`$, we have the update equations for weights $`{\mathbf{u}}`$ and $`{\mathbf{v}}`$ at iteration $`t`$:
``` math
\begin{aligned}
    {\mathbf{u}}(t+1) &= {\mathbf{u}}(t) - \eta\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),\\
    {\mathbf{v}}(t+1) &= {\mathbf{v}}(t) - \eta\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right).
\end{aligned}
```

We consider the evolution of $`K(t)`$ first.
``` math
\begin{aligned}
    K_{i,j}(t+1) &= \left<\nabla_{\mathbf{u}}g_i(0) + \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t+1)-{\mathbf{v}}(0)),\nabla_{\mathbf{u}}g_j(0) + \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t+1)-{\mathbf{v}}(0))\right>\\
    &~~~+ \left<\nabla_{\mathbf{v}}g_i(0) + ({\mathbf{u}}(t+1)-{\mathbf{u}}(0))^T\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}},\nabla_{\mathbf{v}}g_j(0) + ({\mathbf{u}}(t+1)-{\mathbf{u}}(0))^T\frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right>\\
    &= K_{i,j}(t) - \left\langle\eta \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right.,\\
    &~~~~~~~~~~~~~~~~~~~~~~~\left.\nabla_{\mathbf{u}}g_j(0) + \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)) \right\rangle\\
    &~~~-\left<\eta \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right),  \nabla_{\mathbf{u}}g_i(0) + \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)) \right>\\
        &~~~+ \left\langle\eta \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right),\right.\\
    &~~~~~~~~~\left.\eta \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right\rangle\\
     &~~~-\left<\eta \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),  \nabla_{\mathbf{v}}g_i(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} \right>\\
    &~~~-\left<\eta \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),  \nabla_{\mathbf{v}}g_j(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} \right>\\
 &~~~+ \left\langle\eta \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),\right.\\
    &~~~~~~~~~\left.\eta \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n(g_k(t)-y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)\right\rangle.
    
\end{aligned}
```

We separate the data into two sets according to their sign:
``` math
\begin{aligned}
{\mathcal{S}}_+ := \{i: x_i\geq 0, i\in[n]\},~~~~~{\mathcal{S}}_- := \{i: x_i<0, i\in[n]\}.
\end{aligned}
```

We consider two scenarios: (1) $`x_i`$ and $`x_j`$ have different signs; (2) $`x_i`$ and $`x_j`$ have the same sign.

#### (1)

With simple calculation, we get if $`x_i`$ and $`x_j`$ have different signs, i.e., $`i\in{\mathcal{S}}_+,j\in{\mathcal{S}}_-`$ or $`i\in{\mathcal{S}}_-, j\in{\mathcal{S}}_+`$,
``` math
\begin{aligned}
    \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} = 0, ~~~\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\nabla_{\mathbf{u}}g_j(0) = 0,~~~\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\nabla_{\mathbf{v}}g_j(0) = 0.
\end{aligned}
```

Without lose of generality, we assume $`i\in{\mathcal{S}}_+`$, $`j\in{\mathcal{S}}_-`$. Then we have
``` math
\begin{aligned}
    K_{i,j}(t+1) &= K_{i,j}(t).
\end{aligned}
```

#### (2)

If $`x_i`$ and $`x_j`$ have the same sign, i.e., $`i,j\in {\mathcal{S}}_+`$ or $`i,j\in {\mathcal{S}}_-`$,
``` math
\begin{aligned}
    \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}} = \frac{1}{\sqrt{m}} \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}x_j,~~~ \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\nabla_{\mathbf{u}}g_j(0) = \frac{1}{\sqrt{m}}\nabla_{\mathbf{u}}g_i(0)x_j,~~~ \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\nabla_{\mathbf{v}}g_j(0) = \frac{1}{\sqrt{m}}\nabla_{\mathbf{v}}g_i(0)x_j.
\end{aligned}
```

For $`i,j\in{\mathcal{S}}_+`$, we have
``` math
\begin{aligned}
    K_{i,j}(t+1) 
    &= K_{i,j}(t)-\frac{2\eta}{\sqrt{m}} \sum_{k\in{\mathcal{S}}_+} (g_k(t)-y_k)x_i\left\langle\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right.,\\
    &~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\left.\nabla_{\mathbf{u}}g_j(0) + \frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right\rangle\\
    &~~~ -\frac{2\eta}{\sqrt{m}} \sum_{k\in{\mathcal{S}}_+} (g_k(t)-y_k)x_i\left<\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)),\nabla_{\mathbf{v}}g_j(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_j(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right>\\
    &~~~~+ \frac{\eta^2}{m} x_ix_j\left\|\sum_{k\in{\mathcal{S}}_+}(g_k(t)-y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right\|^2 \\
    &~~~~+ \frac{\eta^2}{m} x_ix_j\left\|\sum_{k\in{\mathcal{S}}_+}(g_k(t)-y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)\right\|^2 \\
    &= K_{i,j}(t) - \frac{4\eta}{m}x_ix_j\sum_{k\in{\mathcal{S}}_+}(g_k(t)-y_k) g_k(t) + \frac{\eta^2}{m}x_ix_j \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right)^T K(t) \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right)\\
    &= K_{i,j}(t) - \frac{4\eta}{m}x_i x_j \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+\right)^T\left({\mathbf{g}}(t)\odot {\boldsymbol{m}}_+\right)\\
    &~~~~+ \frac{\eta^2}{m}x_ix_j \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right)^T K(t) \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right).
\end{aligned}
```

Similarly, for $`i,j\in{\mathcal{S}}_-`$, we have
``` math
\begin{aligned}
     K_{i,j}(t+1) &= K_{i,j}(t) - \frac{4\eta}{m}x_i x_j \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_-\right)^T\left({\mathbf{g}}(t)\odot {\boldsymbol{m}}_-\right)\\
     &~~~~~+ \frac{\eta^2}{m}x_ix_j \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_-\right)^T K(t) \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_-\right).
\end{aligned}
```

Combining the results together, we have
``` math
\begin{aligned}
    K(t+1) &= K(t) + \frac{\eta^2}{m} \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right)^T K(t) \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_+\right) {\boldsymbol{p}}_1{\boldsymbol{p}}_1^T\\
    &~~~+ \frac{\eta^2}{m} \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_-\right)^T K(t) \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot{\boldsymbol{m}}_-\right) {\boldsymbol{p}}_2{\boldsymbol{p}}_2^T\\
    &~~~- \frac{4\eta}{m} \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+\right)^T\left({\mathbf{g}}(t)\odot {\boldsymbol{m}}_+\right){\boldsymbol{p}}_1{\boldsymbol{p}}_1^T\\
    &~~~- \frac{4\eta}{m}\left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_-\right)^T\left({\mathbf{g}}(t)\odot {\boldsymbol{m}}_-\right){\boldsymbol{p}}_2{\boldsymbol{p}}_2^T.
\end{aligned}
```

Now we derive the evolution of $`{\mathbf{g}}(t)-{\mathbf{y}}`$. Suppose $`i\in{\mathcal{S}}_+`$. Then we have

``` math
\begin{aligned}
     g_i(t+1) &= g_i(0) + \left<{\mathbf{u}}(t+1)-{\mathbf{u}}(0), \nabla_{\mathbf{u}}g_i(0)\right> + \left<{\mathbf{v}}(t+1)-{\mathbf{v}}(0),\nabla_{\mathbf{v}}g_i(0)\right> \\
     &~~~~~+\left<{\mathbf{u}}(t+1)-{\mathbf{u}}(0), \frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t+1)-{\mathbf{v}}(0)\right>\\
     &= g_i(t) -\eta\left<\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),\nabla_{\mathbf{u}}g_i(0)\right>\\
     &~~~-\eta\left<\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right),\nabla_{\mathbf{v}}g_i(0)\right>\\
     &~~~ -\eta\left<\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right),\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0)\right>\\
     &~~~ -\eta\left<\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right),({\mathbf{u}}(t)-{\mathbf{u}}(0)^T\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right>\\
     &~~~+\eta^2\left\langle\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{u}}g_k(0) + \frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}({\mathbf{v}}(t)-{\mathbf{v}}(0))\right)\right.,\\
     &~~~~~~~~~~~\left.\frac{\partial^2 g_i(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\sum_{k=1}^n (g_k(t) - y_k)\left(\nabla_{\mathbf{v}}g_k(0) + ({\mathbf{u}}(t)-{\mathbf{u}}(0))^T\frac{\partial^2 g_k(0)}{\partial {\mathbf{u}}\partial {\mathbf{v}}}\right)\right\rangle\\
     &= g_i(t) - \eta\sum_{k\in{\mathcal{S}}_+}(g_k(t)-y_k)K_{k,i}(t)+ \frac{\eta^2}{m}\sum_{k\in{\mathcal{S}}_+}\sum_{j\in{\mathcal{S}}_+}(g_k(t)-y_k)(g_j(t)-y_j)g_j(t)x_k x_i.
\end{aligned}
```

Similarly, for $`i\in{\mathcal{S}}_-`$, we have
``` math
\begin{aligned}
      g_i(t+1) =  g_i(t) - \eta\sum_{k\in{\mathcal{S}}_-}(g_k(t)-y_k)K_{k,i}(t)+ \frac{\eta^2}{m}\sum_{k\in{\mathcal{S}}_-f}\sum_{j\in{\mathcal{S}}_-}(g_k(t)-y_k)(g_j(t)-y_j)g_j(t)x_k x_i.
\end{aligned}
```

Combining the results together, we have
``` math
\begin{aligned}
    {\mathbf{g}}(t+1) - {\mathbf{y}}&= \left(I - \eta K(t) + \frac{\eta^2}{m}(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+)^T ({\mathbf{g}}(t)\odot {\boldsymbol{m}}_+){\boldsymbol{p}}_1{\boldsymbol{p}}_1^T\right.\\
    &~~~~~~~+ \left.\frac{\eta^2}{m}(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_-)^T ({\mathbf{g}}(t)\odot {\boldsymbol{m}}_-){\boldsymbol{p}}_2{\boldsymbol{p}}_2^T\right)({\mathbf{g}}(t)-{\mathbf{y}}).
\end{aligned}
```

# Optimization with sub-critical learning rates

<div class="thm">

**Theorem 4**. *Consider training the NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>), with squared loss on a single training example by GD. With a sub-critical learning rate $`\eta \in [\epsilon,\frac{2-\epsilon}{\lambda_0}]`$ with $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$, the loss decreases exponentially with
``` math
\begin{aligned}
    {\mathcal{L}}(t+1) \leq \left(1-\delta +O\left(\frac{1}{m\delta}\right) \right)^2{\mathcal{L}}(t) = (1-\delta + o(\delta))^2{\mathcal{L}}(t),
\end{aligned}
```
where $`\delta = \min(\eta\lambda_0,2-\eta\lambda_0)`$.*

*Furthermore, $`\sup_t|\lambda(t)-\lambda(0)| = O\left(\frac{1}{m\delta}\right)`$.*

</div>

We use the following transformation of the variables to simplify notations.
``` math
\begin{aligned}
    u(t) = \frac{\left\|{\boldsymbol{x}}\right\|^2}{md}\eta^2 (g(t)-y)^2,~~~w(t) =\frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)y,~~~v(t) = \eta\lambda(t).
\end{aligned}
```
Then the Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and Eq. (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>) are reduced to
``` math
\begin{aligned}
    u(t+1) &= (1-v(t)+u(t)+w(t))^2 u(t):=\kappa(t)u(t)\\
    v(t+1) &=v(t) - u(t)(4-v(t))-4w(t).
\end{aligned}
```

At initialization, since $`|g(0)| = O(1)`$, we have $`u(0) \leq C_u/m`$ for some constant $`C_u>0`$. As $`|w(t)| =  \frac{C\sqrt{u(t)}}{\sqrt{m}}`$. where $`C := \frac{\eta \|{\boldsymbol{x}}\||y|}{\sqrt{d}}>0`$, we have $`|w(0)| \leq C_u C / m^{3/2}`$. Note that by definition, for all $`t\geq 0`$, $`u(t)\geq 0`$ we have $`v(t)\geq 0`$ since $`\lambda(t)`$ is the tangent kernel for a single training example. From the definition of $`\delta`$, we can infer that $`\delta<1`$.

In the following, we will show that if $`v(0)\in[\epsilon,2-\epsilon]`$ with $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$, then there exist constant $`C_\kappa, C_v >0`$ such that for all $`t \geq 0`$,
``` math
\begin{aligned}
    \kappa(t) \leq \left(1-\delta+\frac{C_\kappa}{m\delta}\right)^2<1,
    |v(t)-v(0)| \leq \frac{C_v}{m\delta},
\end{aligned}
```

if $`C_\kappa \geq 9C_u + (C_u +C_uC)\delta`$ and $`m`$ satisfies
``` math
\begin{aligned}
    m > \max\left\{ \frac{12C_\kappa}{\delta^2},\frac{\sqrt{6}C_\kappa}{\delta^{3/2}}, C^2 \right\}.
\end{aligned}
```
Given the condition on $`m`$, we have $`(1-\delta+\frac{C_\kappa}{m\delta})^2<1`$.

We will prove the result by induction. When $`t=0`$, we have
``` math
\begin{aligned}
    \kappa(0) = \left(1-v(0)+u(0) + w(0)\right)^2< \left(1-\delta + \frac{C_u}{m} + \frac{C_u C}{m^{3/2}}\right)^2 < \left(1-\delta+{\frac{C_\kappa}{m\delta}}\right)^2,
\end{aligned}
```
where we use the assumption $`C_\kappa \geq (C_u +C_uC)\delta`$.

Therefore the result holds at $`t=0`$.

Suppose when $`t=T`$ the results hold. Then at $`t=T+1`$, by the inductive hypothesis that $`u(t)`$ decreases exponentially with $`\kappa(t) <\left(1-\delta+{\frac{C_\kappa}{m\delta}}\right)^2`$, we can bound the change of $`v(T+1)`$ from $`v(0)`$:
``` math
\begin{aligned}
    |v(T+1) - v(0)| &= \sum_{t=0}^T|u(t)(4-v(t)) + 4w(t)|\\
    &\leq \sum_{t=0}^T|u(t)||4-v(t)| + \sum_{t=0}^T 4|w(t)| \\
    &\leq \max_{t\in[0,T]}|v(t)-4| \cdot  \frac{u(0) - u(T) \max_{t\in[0,T]}\kappa(t)}{1 - \max_{t\in[0,T]}\kappa(t)} + \frac{w(0) - w(T) \max_{t\in[0,T]}\sqrt{\kappa(t)}}{1 - \max_{t\in[0,T]}\sqrt{\kappa(t)}} \\
    &\leq 4 \cdot \frac{u(0)}{1 - \max_{t\in[0,T]}\kappa(t)} +\frac{w(0)}{1 - \max_{t\in[0,T]}\sqrt{\kappa(t)}} \\
    &\leq 4 \cdot \frac{C_u/m}{\delta/2} + \frac{C_uC/m^{3/2}}{\delta/2} \\
    &\leq \frac{9C_u}{m\delta}.
\end{aligned}
```

For the summation of the “geometric sequence” i.e., $`\{u(0),u(1),\cdots,u(T)\}`$ where $`u(0)`$ and $`u(T)`$ have the determined order but the ratio has an upper bound, we use the maximum ratio, i.e., $`\max\kappa(t)`$ in the denominator to upper bound the summation.

For $`1 - \max_{t\in[0,T]}\kappa(t)`$, we use the bound that
``` math
\begin{aligned}
    1- \left(1-\delta+{\frac{C_\kappa}{m\delta}}\right)^2 &= 2\delta - \delta^2 -\frac{C_\kappa^2}{m^2\delta^2} -\frac{2C_\kappa}{m\delta} +\frac{2C_\kappa}{m} \\
    &\geq \delta - \frac{\delta}{6} - \frac{\delta}{6}- \frac{\delta}{6}\\
    &\geq \frac{\delta }{2},
\end{aligned}
```
where we use the assumption on $`m`$.

Furthermore,
``` math
\begin{aligned}
    \kappa(T+1) &= (1-v(T+1)+u(T+1)+w(T+1))^2\\
    &=(1-v(0) + v(0) - v(T+1)+u(T+1)+w(T+1))^2 \\
    &\leq \left(1 - \delta + \frac{9C_u}{m\delta} + \frac{C_u}{m}+ \frac{C_u C}{m^{3/2}}\right)^2\\
    &\leq \left(1-\delta+ {\frac{C_\kappa}{m\delta}}\right)^2.
\end{aligned}
```

Here we use the assumption $`C_\kappa \geq 9C_u + (C_u +C_uC)\delta`$.

Therefore, we finish the inductive step hence finishing the proof.

# Proof of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>

We present the formal statement of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>:

#### Lemma 1.

*Consider constants $`C_u, C'_u, C_v, C_\kappa, C_\epsilon >0`$ which satisfies $`C_\kappa \geq 28 C'_u + C'_u\delta + C\sqrt{C'_u}`$ where $`C = \eta \left\|{\boldsymbol{x}}\right\||y|/\sqrt{d}`$, and $`C_v \geq 28C'_u`$. If $`m`$ satisfies
``` math
\begin{aligned}
    m \geq \max\left\{\frac{C_\kappa \delta}{C_u(C+1)},
    % \frac{C_v \delta}{C_u(C+2)}, 
    \left(\frac{2C_u +4C\sqrt{C_u}}{C_\kappa C_\epsilon}\right)^2, \frac{576 C^2}{C'_u \delta^2}, \exp(C_v\delta), \exp(4C_\kappa)\right\},
\end{aligned}
```
then with high probability over random initialization of the weights, the following holds: for $`T>0`$ such that $`\sup_{t\in[0,T]}u(t) \leq \frac{C'_u \delta^2}{\log m}`$, $`u(t)`$ increases exponentially with ratio $`\inf_{t\in[0,T]}\kappa(t) \geq \left(1+\delta - \frac{C_\kappa \delta}{\log m}\right)^2 >1`$ and $`\sup_{t\in[0,T]}|v(t) - v(0)| \leq \frac{C_v \delta}{\log m}`$.*

<div class="proof">

*Proof.* Due to the random initialization of the weights, we have with probability, there exists constant $`C_u > 0`$ such that $`|u(0)| \leq C_u/m`$. As $`|w(t)| =  \frac{C\sqrt{u(t)}}{\sqrt{m}}`$, where $`C := \frac{\eta \|{\boldsymbol{x}}\||y|}{\sqrt{d}}>0`$, we have $`|w(0)| \leq \frac{C\sqrt{C_u}}{m^{3/2}}`$.

We prove the results by induction.

Recall that $`\delta := \eta\lambda_0 - 2 \in [\epsilon, 2-\epsilon]`$ where $`\epsilon \in [C_\epsilon\log m /\sqrt{m}, C_\epsilon'\log m /\sqrt{m}]`$ for some constant $`0<C_\epsilon<C_\epsilon'`$.

When $`t=0`$, as $`v(0) =\eta\lambda_0 =  \delta +2`$, we have
``` math
\begin{aligned}
    \kappa(0)  &= \left(1-v(0)+u(0) + w(0)\right)^2\\
    &= \left( 1 - (\delta+2) +u(0) + w(0)\right)^2\\
    &= \left(1+\delta-u(0) - w(0)\right)^2.\\
\end{aligned}
```
Based on the condition on $`m`$ that $`m \geq \frac{C_\kappa\delta}{C_u(C+1)}`$, we have,
``` math
\begin{aligned}
    \left(1+\delta-u(0) - w(0)\right)^2&\geq \left(1+\delta-\frac{C_u}{m} - \frac{C_uC}{m^{3/2}}\right)^2\\
    &\geq \left(1+\delta-\frac{C_u}{m} - \frac{C_uC}{m}\right)^2\\
    &\geq \left(1+\delta - \frac{C_\kappa \delta}{\log m}\right)^2.
\end{aligned}
```

And by the condition $`m \geq \left(\frac{2C_u +4c\sqrt{C_u}}{C_\kappa C_\epsilon}\right)^2`$, we get
``` math
\begin{aligned}
    |v(1) - v(0)|\leq |u(0)||2-\delta| + 4|w(0)| \leq 2C_u/m +\frac{4C\sqrt{C_u}}{m^{3/2}} \leq C_\kappa \delta /\log m.
\end{aligned}
```
Therefore the results hold at $`t=0`$.

Suppose when $`t=T'`$ the results hold. Then at $`t=T'+1`$, by the inductive hypothesis that $`u(t)`$ increases exponentially with a rate at least $`\left(1+\delta -\frac{C_\kappa\delta}{\log m}\right)^2`$ from $`u(0) \leq C_u/m`$ to $`u(T') \leq \frac{C_u' \delta^2}{\log m}`$, we can bound the change of $`v(t)`$:
``` math
\begin{aligned}
    |v(T'+1)-v(0)| &= \left|\sum_{t=1}^{T'} u(t)(v(t)-4)+4w(t)\right| \nonumber \\
    &\leq  \max_{t\in[0,T']}|v(t)-4| \sum_{t=1}^{T'} u(t) + 4\sum_{t=1}^{T'} |w(t)|\nonumber\\
    &\leq \max_{t\in[0,T']}|v(t)-4| \frac{u(T') \min_{t\in[0,T']}\kappa(t) }{\min_{t\in[0,T']}\kappa(t) -1}  + 4\frac{|w(T')| \min_{t\in[0,T']}\sqrt{\kappa(t)}}{\min_{t\in[0,T']}\sqrt{\kappa(t)} -1}\nonumber\\
    &\leq \left(\max_{t\in[0,T']}|v(t)-v(0)| + |v(0)-4|\right)\frac{\left(\frac{C_u'\delta^2}{\log m}\right)\cdot (1+\delta)^2 }{\left(1+\delta -\left(\frac{C_\kappa\delta}{\log m}\right)\right)^2-1}\nonumber\\
    &~~~~~+ \frac{4\left(\frac{C\sqrt{C'_u}\delta}{\sqrt{m\log m}}\right)\cdot (1+\delta) }{\left(1+\delta -\left(\frac{C_\kappa\delta}{\log m}\right)\right)-1} \nonumber\\
    &\leq \left(2-\delta + {\frac{C_v\delta}{\log m}}\right)\cdot  \left(\frac{9C_u'\delta}{\log m}\right) +{\frac{24C\sqrt{C_u'}}{\sqrt{m\log m}}}\nonumber \\
    &\leq \frac{28C_u'\delta}{\log m}\label{eq:bound_v_change}.
\end{aligned}
```

Here are the techniques we used for the above inequalities: for the summation of the “geometric sequence” i.e., $`\{u(0),u(1),\cdots,u(T')\}`$ where $`u(0)`$ and $`u(T')`$ have the determined order but the ratio has a lower bound, we use the smallest ratio, i.e., $`\inf\kappa(t)`$ to upper bound the summation. Specifically, we apply the following inequality to bound the summation:
``` math
\begin{aligned}
    \sum_{t=1}^{T'} u(t) \leq \sum_{t=1}^{T'} \frac{u(T')}{\left(\min_{t\in[0,T']}\kappa(t)\right)^{t-1}} = u(T') \sum_{t=1}^{T'} \frac{1}{\left(\min_{t\in[0,T']}\kappa(t)\right)^{t-1}} \leq u(T') \frac{\min_{t\in[0,T']}\kappa(t)}{\min_{t\in[0,T']}\kappa(t)-1}.
\end{aligned}
```

Additionally, sine $`m \geq \exp(4C_\kappa \delta)`$, we used the inequality
``` math
\begin{aligned}
    \left(1+\delta - \frac{C_\kappa \delta}{\log m}\right)^2 - 1 &= \left(1 - \frac{C_\kappa \delta}{\log m}\right)^2\delta^2 + 2 \left(1 - \frac{C_\kappa \delta}{\log m}\right)\delta\\
    &\geq 2 \left(1 - \frac{C_\kappa \delta}{\log m}\right)\delta \geq \delta,
\end{aligned}
```
and $`\left(1+\delta -\left(\frac{C_\kappa\delta}{\log m}\right)\right)-1 \geq \frac{\delta}{2}`$ to bound the denominator of the summation of the geometric sequence.

And we further used the inequality $`0<\delta<2`$ and $`{\frac{24C\sqrt{C_u'}}{\sqrt{m\log m}}} \leq \frac{C'_u \delta}{\log m}`$ by the condition on $`m`$ to get the final upper bound.

Consequently, by the assumption $`C_v \geq 28 C'_u \delta`$, we have $`|v(T'+1) - v(0)| \leq \frac{C_v \delta}{\log m}`$.

Now we bound the ratio $`\kappa(T'+1)`$. By our assumption, $`u(T'+1) \leq u(T) \leq \frac{C_u' \delta^2}{\log m}`$, and we can similarly bound $`|w(T'+1)| \leq \frac{C\sqrt{C'_u}\delta}{\sqrt{m\log m}}`$ as $`|w(T'+1)| = \frac{C\sqrt{u(T'+1)}}{\sqrt{m}}`$.

And the rate $`\kappa(T'+1)`$ satisfies
``` math
\begin{aligned}
    \kappa(T'+1) &= (1-v(T'+1)+u(T'+1) + w(T'+1))^2 \\
    &= \left(1- v(0) + v(0) - v(T'+1) + u(T'+1) + w(T'+1)\right)^2\\
    &= \left(1+\delta + v(T'+1) - v(0) - u(T'+1) - w(T'+1)\right)^2.
\end{aligned}
```
Note that $`|v(T'+1) - v(0)|\leq \frac{28 C_u'\delta}{\log m}`$ by Eq. (<a href="#eq:bound_v_change" data-reference-type="ref" data-reference="eq:bound_v_change">[eq:bound_v_change]</a>). By the assumption that $`m \geq \exp(4C_\kappa)`$ and $`C_\kappa \geq 28 C'_u + C'_u\delta +  C\sqrt{C'_u}`$, we have $`\delta > |v(T'+1) - v(0)| + u(T'+1) + |w(T'+1)|`$.

Consequently, we can get
``` math
\begin{aligned}
     \kappa(T'+1)  &= \left(1+\delta + v(T'+1) - v(0) - u(T'+1) - w(T'+1)\right)^2 \\
     &\geq \left(1+\delta - |v(T'+1) - v(0)| - u(T'+1) - |w(T'+1)|\right)^2\\
    &\geq \left(1 + \delta - \frac{28C_u' \delta}{\log m } -\frac{C_u'\delta^2}{\log m} -\frac{C\sqrt{C'_u}\delta}{\sqrt{m\log m}} \right)^2\\
    &\geq \left(1+\delta -\frac{C_\kappa\delta}{\log m}\right)^2.
\end{aligned}
```

Since $`m\geq \exp(4C_\kappa)`$, we have $`\left(1+\delta -\frac{C_\kappa\delta}{\log m}\right)^2 \geq \left(1 + \frac{3}{4}\delta\right)^2 >1`$.

Then we finish the inductive step hence finishing the proof. ◻

</div>

# Proof of Lemma <a href="#lemma:kappa_t" data-reference-type="ref" data-reference="lemma:kappa_t">2</a>

A formal statement of Lemma <a href="#lemma:kappa_t" data-reference-type="ref" data-reference="lemma:kappa_t">2</a> is as follows:

#### Lemma 2:

Under the condition of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, if we further assume that $`m`$ satisfies
``` math
\begin{aligned}
    m > \max \left\{\exp(2C_v\delta), \frac{256C^2}{(C_\epsilon - C'_v)^2 {C'_u}^2}, \exp(5(C'_u + 4C\sqrt{C'_u})), \exp\left(\frac{C'_u(C_\epsilon -2C'_v) - 8C\sqrt{C'_u}}{20CC'_u}\right) \right\},
\end{aligned}
```
where $`C'_v := 18C_u' + 2 C_v`$, and $`C_v \geq 4C\sqrt{C'_u}`$, $`C_\epsilon > 2C'_v`$, then with high probability over random initialization of the weights, the following holds: there exists $`T^*>0`$ such that $`u(T^*) = O\left(\frac{1}{m}\right)`$.

<div class="proof">

*Proof.* The main idea of the proof is the following: as $`u(t)`$ increases, $`v(t)`$ decreases since $`u(t)(4-v(t)) \gg w(t) = \Theta(\sqrt{u(t)}/\sqrt{m})`$ in Eq. (<a href="#eq:v" data-reference-type="ref" data-reference="eq:v">[eq:v]</a>) and $`u(t)(4-v(t))<0`$. Furthermore, the increase of $`u(t)`$ speeds up the decrease of $`v(t)`$. However, $`v(t)`$ cannot decrease infinitely as $`v(t)\geq 0`$ by definition. Therefore, $`u(t)`$ has to stop increasing at some point and decrease to a small value.

We first show that by the choice of the learning rate that $`4-v(0) \geq \epsilon`$ where $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$, we will have $`4-v(t) >0`$ for all $`t`$ in the increasing phase. Recall that $`\delta:=\eta\lambda_0-2`$.

<div id="prop:v_4" class="proposition">

**Proposition 1**. *Under the condition in Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, if we further assume $`m > \exp\left(\frac{48C\sqrt{C'_u}}{C_\epsilon}\right)^{2/3}`$, then for $`T>0`$ such that $`\sup_{t\in[0,T]}u(t) \leq \frac{C_u'\delta^2}{\log m}`$, we have $`v(T)<4 - \frac{C_\epsilon \log m}{2\sqrt{m}}`$.*

</div>

See the proof in Appendix <a href="#proof:v_4" data-reference-type="ref" data-reference="proof:v_4">13.1</a>

Given the constant $`C_u'`$ in Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, we define the end of the increasing phase by $`T_1`$, i.e.,
``` math
\begin{aligned}
\label{eq:end_increase}
    T_1:= \sup \left\{t:u(t) \leq \frac{C'_u\delta^2}{\log m}\right\}.
\end{aligned}
```

We further show that there exists $`T_2 \geq T_1`$ such that $`v(T_2) \leq 3`$.

Note that we indeed can show that there exists $`T_2`$ such that $`v(T_2) < \overline{C}`$ where $`\overline{C}\in(2,4)`$ is a constant independent of $`m`$. Here for the simplicity of the presentation, we take $`C`$ as $`3`$. Furthermore, we note that $`T_1, T_2`$ depends on $`m`$.

Before that, we present a useful result that controls the decrease of $`v(t)`$:

<div id="prop:v_when_decrease" class="proposition">

**Proposition 2**. *For $`t`$ such that $`v(t)<4`$, if $`u(t) > \frac{4C}{m (4-v(t))^2}`$, then $`v(t+1) <v(t)`$.*

</div>

See the proof in Appendix <a href="#proof:v_when_decrease" data-reference-type="ref" data-reference="proof:v_when_decrease">13.2</a>.

Now we are ready to show the existence of $`T_2`$ such that $`v(T_2) \leq 3`$.

<div id="prop:v_close_4" class="proposition">

**Proposition 3**. *Under the condition of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, if we further assume that $`m`$ satisfies
``` math
\begin{aligned}
    m> \max \left\{\exp\left(\frac{768^2 C^2}{C'_u C_\epsilon^2}\right), \exp\left(2C'_u+C_\epsilon\right),\exp\left(\frac{48C\sqrt{C'_u}}{C_\epsilon}\right)^{2/3},\frac{16C^2}{{C'_u}^2}\right\},
\end{aligned}
```
and $`C'_u \geq 4C^2`$, there exists $`T_2\geq T_1`$ such that $`v(T_2) \leq 3`$.*

</div>

See the proof in Appendix <a href="#proof:v_close_4" data-reference-type="ref" data-reference="proof:v_close_4">13.3</a>

Since $`v(T_2) <3`$ hence $`4-v(T_2) \geq 1`$. Simply using Proposition <a href="#prop:v_when_decrease" data-reference-type="ref" data-reference="prop:v_when_decrease">2</a>, we get

<div class="proposition">

**Proposition 4**. *$`v(t)`$ keeps decreasing after $`T_2`$ until $`u(t) = O\left(\frac{1}{m}\right)`$.*

</div>

By definition $`v(t)=\eta\lambda(t)`$ where $`\lambda(t)\geq 0`$, $`v(t)`$ will not keep decreasing for $`t\rightarrow \infty`$ hence there exists $`T^*`$ such that $`u(T^*) = O\left(\frac{1}{m}\right)`$. And it indicates that the loss will decrease to the order of $`O(1)`$.

# Proof of Theorem <a href="#thm:equi" data-reference-type="ref" data-reference="thm:equi">2</a>

We compute the steady-state equilibria of Eq. (<a href="#eq:u" data-reference-type="ref" data-reference="eq:u">[eq:u]</a>) and (<a href="#eq:v" data-reference-type="ref" data-reference="eq:v">[eq:v]</a>). By letting $`u(t+1)=u(t)`$ and $`v(t+1)=v(t)`$, we have the steady-state equilibria $`(u^*,v^*)`$ satisfy one of the following:

1.  $`u^* = 0`$, $`v^* \in \mathbb{R}`$;

2.  $`|1-v^*+u^*+w^*| = 1`$, $`u^*(4-v^*)+4w^* = 0`$.

As $`w(t)^2 = \frac{C^2u(t)}{{m}}`$ where $`C := \frac{\eta \|{\boldsymbol{x}}\||y|}{\sqrt{d}}>0`$, we write $`w`$ as a function of $`u`$ for simplicity, hence $`w^* = w(u^*)`$.

As the dynamics equations are non-linear, we analyze the local stability of the steady-state equilibria. We consider the Jacobian matrix of the dynamical systems:
``` math
\begin{aligned}
    J(u,v) = \begin{bmatrix}
  2(1-v+u+w)(1+\frac{dw}{du})u + (1-v+u+w)^2 & -2(1-v+u+w)u\\
v-4-4\frac{dw}{du} & 1+u
\end{bmatrix}.
\end{aligned}
```

We analyze the stability of two equilibria separately.

For Scenario (1), we evaluate $`J(u,v)`$ at the steady-state equilibrium $`(u^*,v^*)`$ then we get
``` math
\begin{aligned}
    J(u^*,v^*) = \begin{bmatrix}
  (1-v^*)^2 & 0\\
v^*-4-4\frac{dw}{du} & 1
\end{bmatrix}.
\end{aligned}
```
We get the two eigenvalues of $`J(u^*,v^*)`$ are $`1`$ and $`(1-v^*)^2`$. We will show the Lyapunov stability of the equilibrium $`(u^*,v^*)`$. Specifically, we apply Theorem 1.2 in . We find the domain
``` math
\begin{aligned}
    D =\{(u,v): u\leq C_1, |v-v^*|\leq \min(|C_2-v^*|,|2-C_2-v^*|\},
\end{aligned}
```
where $`C_1 = \Theta(1/m)`$ and $`C_2 = \Theta( 1/\sqrt{m})`$, and the Lyapunov function $`V(u,v) = u+(v-v^*)^2`$. It is not hard to verify that $`V`$ is locally Lipschitz in D as $`V`$ is continuous in a compact domain. Furthermore, we can see that $`(u^*,v^*)`$ with $`u^* = 0, v^*\in[\epsilon,2-\epsilon]`$ where $`\epsilon = \Theta(\log m/\sqrt{m})`$ satisfies the condtions Eq. (3,4) in Theorem 1.2 in . Therefore, $`(u^*,v^*)`$ with $`u^*=0`$ and $`v^*\in[\epsilon,2-\epsilon]`$ is a stable equilibrium point.

For Scenario (2), we again evaluate $`J(u,v)`$ at the steady-state equilibrium $`(u^*,v^*)`$ then we get
``` math
\begin{aligned}
    J(u^*,v^*) = \begin{bmatrix}
  -2u^*+\frac{C\sqrt{u^*}}{\sqrt{m}}+1 & 2u^*\\
-\frac{2C}{\sqrt{mu^*}} & 1+u^*
\end{bmatrix},
\end{aligned}
```
where we replace $`v^*`$ by $`4 + 4w^*/u^*`$ based on the second equality in Scenario (2). Note that $`u^*(4-v^*) >0`$ since $`v<4`$ during the whole training process, therefore we have $`w^*<0`$ to achieve the equilibrium.

We can compute the eigenvalue of $`J(u^*,v^*)`$ then we get
``` math
\begin{aligned}
    \lambda_J = 1 + \frac{C}{2\sqrt{m}}\sqrt{u^*} - \frac{u^*}{2} \pm \frac{1}{2}(u^*)^{1/4}\sqrt{16\frac{C}{\sqrt{m}} - \frac{C^2\sqrt{u^*}}{m} + 6\frac{Cu^*}{\sqrt{m}} - 9(u^*)^{3/2}}i.
\end{aligned}
```

Note that when Scenario (2) holds, there are only two possible cases

1.  $`u^* = \Theta(1/m)`$, $`|w^*| = \Theta(1/m)`$ and $`v^* = \Theta(1)`$;

2.  $`u^* = \Theta(1/m)`$, $`|w^*| = \Theta(1/m)`$ and $`v^* = \Theta(1/m)`$.

For (2.1), by the first equality $`v^* = 2-u^*+w^* \in(1,2)`$. Then plugging $`v^*`$ into the second equality yields $`u^*\in\left(\frac{4}{3}\frac{C^2}{m}, 2\frac{C^2}{m}\right)`$.

For (2.2), by the second equality that $`u^*(4-v^*) + 4w^*=0`$, we have $`u^* = \frac{C^2}{m} + o(1/m)`$.

By computing the modulo of $`\lambda_J`$, we have
``` math
\begin{aligned}
    |\lambda_J| = 1 + \frac{5C}{\sqrt{m}}\sqrt{u^*} -u^*  + o\left(\frac{1}{m}\right).
\end{aligned}
```

Therefore, for both (2.1) and (2.2) we have $`|\lambda_J|>1`$ which indicates $`(u^*,v^*)`$ is unstable. ◻

</div>

# Optimization with $`\eta>{\eta_{\mathrm{max}}}`$

<div id="thm:max_lr" class="thm">

**Theorem 5**. *Consider training the NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>), with squared loss on a single training example by GD. If the learning rate satisfies $`\eta \in \left[\frac{4+\epsilon}{\lambda_0},\infty\right)`$ with $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$, then GD diverges.*

</div>

<div class="proof">

*Proof.* We similarly use the transformation transformation of the variables to simplify notations.
``` math
\begin{aligned}
    u(t) = \frac{\left\|{\boldsymbol{x}}\right\|^2}{md}\eta^2 (g(t)-y)^2,~~~w(t) =\frac{\|{\boldsymbol{x}}\|^2}{md}\eta^2(g(t)-y)y,~~~v(t) = \eta\lambda(t).
\end{aligned}
```
Then the Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and Eq. (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>) are reduced to
``` math
\begin{aligned}
    u(t+1) &= (1-v(t)+u(t)+w(t))^2 u(t):=\kappa(t)u(t)\\
    v(t+1) &=v(t) - u(t)(4-v(t))-4w(t).
\end{aligned}
```

We similarly consider the interval $`[0,T]`$ such that $`\sup_{t\in[0,T]}u(t) = O\left(\frac{1}{\log m}\right)`$. By Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, in $`[0,T]`$, $`u(t)`$ increases exponentially with a rate $`\sup_{t\in[0,T]}\kappa(t) > 9`$. We assume $`|w(t)| > |u(t)(4-v(t))|`$ for all $`t\in[0,T]`$, which is the worst case as $`v(t)`$ will increase the least. By Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, we have $`\sum_{t=0}^T |w(t)| = O\left(\frac{1}{\sqrt{m\log m}}\right)`$, which is less than $`\epsilon`$. Therefore, we have $`v(T)>4`$.

Then at the end of the increasing phase, we have $`|u(T_1)(4-v(T_1))| = \Omega(1/\sqrt{m})`$ is of a greater order than $`|w(T_1)| = O(1/\sqrt{m\log m})`$, hence $`v(t)`$ will increase at $`T_1`$. Note that $`\kappa(T_1) = (1-4+o(1))^2 = 9+o(1)`$, hence $`u((t)`$ also increases at $`T_1`$.

It is not hard to see that $`v(t)`$ will keep increasing unless $`u(t)`$ decreases to a smaller order. Specifically, if $`|u(t)(4-v(t))|= |4w(t)|`$, it requires $`u(t)`$ to be of the order at least $`O(1/\sqrt{\log m})`$ (by letting $`\epsilon u(t) = \Theta(w(t)) =  \Theta(\sqrt{u(t)/m})`$), which will not happen as $`\kappa(t) = (1-v(t) +o(1))^2 >1`$ and it contradicts the decrease of $`u(t)`$.

Therefore, both $`u(t)`$ and $`v(t)`$ keep increasing which leads to the divergence of GD. ◻

</div>

# Proof of propositions

## Proof of Proposition <a href="#prop:v_4" data-reference-type="ref" data-reference="prop:v_4">1</a>

<div class="proof">

*Proof.* Note that $`4-v(0) = 2-\delta \geq \frac{C_\epsilon \log m}{\sqrt{m}}`$ by definition, where $`C_\epsilon>0`$ is a constant. To show $`4 - v(T) >\frac{C_\epsilon \log m}{2\sqrt{m}}`$, a sufficient condition is $`v(T) - v(0) < \frac{C_\epsilon \log m}{2\sqrt{m}}`$.

Specifically, we will prove for $`T>0`$ such that $`\sup_{t\in[0,T]}u(t) \leq \frac{C'_u \delta^2}{\log m}`$, the following holds:
``` math
\begin{aligned}
    v(T) - v(0) < 4\sum_{t=0}^T |w(t)| \leq \frac{24C\sqrt{C'_u}}{\sqrt{m\log m}},
\end{aligned}
```
where $`C,C'_u`$ are the same constants defined in Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>. Then by the condition that $`m > \exp\left(\frac{48C\sqrt{C'_u}}{C_\epsilon}\right)^{2/3}`$, we have $`v(T) - v(0) < \frac{C_\epsilon \log m}{2\sqrt{m}}`$.

We will prove the result by induction.

When $`T = 0`$, the result holds trivially.

Suppose $`T=T'`$ the result holds. When $`T = T'+1`$, since $`v(T') -v(0)<0`$, we have $`v(T') < 4`$. Therefore, by the update equation of $`v(t)`$ Eq. (<a href="#eq:v" data-reference-type="ref" data-reference="eq:v">[eq:v]</a>), we have
``` math
\begin{aligned}
    v(T'+1) &= v(T') - u(T')(4 - v(T')) - 4w(T')\\
    &\leq v(T') - 4w(T')\\
    &\leq v(T') + 4|w(T')|.
\end{aligned}
```

Then $`v(T'+1) - v(0) =v(T'+1) - v(T') + v(T') - v(0) \leq \sum_{t=0}^{T'+1}|w(t)|`$.

By Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, we have $`\sum_{t=0}^{T'+1}|w(t)| \leq \frac{24C\sqrt{C'_u}}{\sqrt{m\log m}}`$. Indeed, this inequality holds for any $`T'+1`$ such that $`\sup_{t\in[0,T'+1]}u(t) \leq \frac{C'_u \delta^2}{\log m}`$.

Therefore, we finish the inductive step hence finish the proof. ◻

</div>

## Proof of Proposition <a href="#prop:v_when_decrease" data-reference-type="ref" data-reference="prop:v_when_decrease">2</a>

<div class="proof">

*Proof.* A sufficient condition for $`v(t)`$ to decrease is
``` math
\begin{aligned}
        u(t)(4-v(t)) > 4|w(t)| = \frac{4C\sqrt{u(t)}}{\sqrt{m}}.
    
\end{aligned}
```
If $`u(t) > \frac{4C}{m (4-v(t))^2}`$, then the above condition is satisfied. ◻

</div>

## Proof of Proposition <a href="#prop:v_close_4" data-reference-type="ref" data-reference="prop:v_close_4">3</a>

<div class="proof">

*Proof.* Note that for $`t \in [0,T_1]`$, the change of $`v(t)`$ satisfies $`\sup_t|v(t)-v(0)| \leq \frac{C_v \delta}{\log m}`$ by Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>.

For $`\delta < 1-\frac{C_v \delta}{\log m}`$, i.e., $`v(0) < 3 -\frac{C_v \delta}{\log m}`$, we have $`v(T_1) < v(0) + |v(T_1) - v(0)| = 2+\delta + |v(0) - v(T_1)| < 3`$. Therefore, the existence of $`T_2`$ can be guaranteed by simply letting $`T_2 = T_1`$.

For $`\delta \geq 1-\frac{C_v \delta}{\log m}`$, i.e., $`v(0) \geq 3 -\frac{C_v \delta}{\log m}`$, we will show there exists $`T_2 \geq T_1`$ which depends on $`m`$ such that $`v(T_2)<3`$.

We prove the existence of $`T_2`$ by contradiction. Suppose that for all $`t \geq T_1+1`$ we have $`v(t)\geq 3`$.

For the simple case that if all $`u(t) > \frac{4C}{m(4-v(t))^2}`$, then by Proposition <a href="#prop:v_when_decrease" data-reference-type="ref" data-reference="prop:v_when_decrease">2</a>, $`v(t)`$ keeps decreasing which will ultimately lead to $`v(t)<3`$.

Suppose there is an iteration $`t \geq T_1+1`$ such that $`u(t) \leq  \frac{4C}{m(4-v(t))^2}`$. The following Proposition guarantees that $`v(t)`$ will decrease to a smaller value after $`t`$ once such $`t`$ occurs. Therefore, we can find $`T_2`$.

<div id="prop:v_decreasing" class="proposition">

**Proposition 5**. *Under the condition of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>, suppose $`m`$ further satisfies
``` math
\begin{aligned}
       m> \max\left\{\exp\left(\frac{768^2 C^2}{C'_u C_\epsilon^2}\right),\frac{16C^2}{{C'_u}^2},\exp\left(2C'_u+C_\epsilon\right)\right\},
    
\end{aligned}
```
where $`C'_u \geq 4C^2`$.*

*Then if there is $`T\geq 0`$ such that $`u(T) \leq \frac{4C}{m(4-v(T))^2}`$ and $`v(T) >3`$, we have $`v(T'+2) < v(T)`$ and $`u(T'+1) > \frac{4C}{m(4-v(T'))^2}`$, where $`T'`$ is the end of the increasing phase starting from $`T`$.*

</div>

See the proof in Appendix <a href="#proof:v_decreasing" data-reference-type="ref" data-reference="proof:v_decreasing">13.4</a>. ◻

</div>

## Proof of proposition <a href="#prop:v_decreasing" data-reference-type="ref" data-reference="prop:v_decreasing">5</a>

<div class="proof">

*Proof.* Since $`u(T) \leq \frac{C'_u}{\log m}`$ by the assumption that $`m > \frac{16C^2}{{C'_u}^2}`$, the training dynamics falls into the increasing phase from $`T`$. We denote the end of the increasing phase starting from $`T`$ by $`T'`$, i.e.,
``` math
\begin{aligned}
        T' = \sup\left\{t: u(t)\leq \frac{C'_u \delta^2}{\log m}, t \geq T\right\}.
    
\end{aligned}
```
We will prove the result by induction.

Suppose $`T'`$ is the end of the first increasing phase, i.e., $`T' = T_1`$. By proposition <a href="#prop:v_4" data-reference-type="ref" data-reference="prop:v_4">1</a>, $`v(T_1) < 4 - \frac{C_\epsilon \log m}{2\sqrt{m}}`$. And the magnitude of $`u(T_1+1)`$ can be lower bounded by
``` math
\begin{aligned}
\label{eq:lower_bound_u}
   u(T_1+1)\geq \frac{C'_u}{4 \log m},
\end{aligned}
```
where we use $`\delta \geq \frac{1}{2}`$ by the assumption on $`m`$ that and plug $`\delta`$ in $`\frac{C'_u \delta^2}{\log m}`$.

Note that when $`m > \exp\left(28 C'_u\right)`$, $`\delta \geq \frac{1}{2}`$ is necessary for $`v(T_1) > 3`$ as $`|v(T_1) - v(0)|\leq \frac{28C_u'\delta}{\log m}`$ by Inequality (<a href="#eq:bound_v_change" data-reference-type="ref" data-reference="eq:bound_v_change">[eq:bound_v_change]</a>).

Furthermore, with the above bound on $`u(T_1+1)`$, we have

``` math
\begin{aligned}
    v(T_1+1) &= v(T_1) - u(T_1)(4-v(T_1))- 4w(T_1) \nonumber\\
    &\leq  4 - \frac{C_\epsilon \log m}{2\sqrt{m}} + \frac{C'_u}{4 \log m}\frac{C_\epsilon \log m}{2\sqrt{m}} + 4|w(T_1)| \nonumber\\
    &\leq 4 - \frac{C_\epsilon \log m}{2\sqrt{m}} + \frac{C'_u C_\epsilon }{8\sqrt{m}} + \frac{2\sqrt{C'_u}}{\sqrt{m\log m}}\nonumber\\
    &\leq 4 - \frac{C_\epsilon \log m}{4\sqrt{m}} \label{eq:v_t_1+1}.
\end{aligned}
```

Consequently, when $`C'_u \geq 4C^2`$ and $`m>\exp\left(2C'_u+C_\epsilon\right)`$, we have
``` math
\begin{aligned}
    u(T_1+1) &= \kappa(T_1) u(T_1) = (1 - v(T_1) + u(T_1) +w(T_1))^2 u(T_1)\\
    &\leq  \left(1 - 4 + \frac{C_\epsilon \log m}{4 \sqrt{m}} + \frac{C'_u }{4 \log m} + \frac{C\sqrt{C'_u}}{2\sqrt{m\log m}}\right)^2\frac{C'_u }{4 \log m} \\
    &\leq  \frac{ 9C'_u}{4\log m}.
\end{aligned}
```

Therefore, at $`T_1+1`$, we have
``` math
\begin{aligned}
    v(T_1+1) - v(T_1+2) &\geq u(T_1+1)(4-v(T_1+1)) - \frac{4C\sqrt{u(T_1+1)}}{\sqrt{m}} \\
    &\geq \frac{C'_u}{4\log m} {\frac{C_\epsilon \log m}{4\sqrt{m}}}  - \frac{8C\sqrt{C'_u}}{\sqrt{m\log m}}\\
    &= \frac{C'_u C_\epsilon}{16\sqrt{m}}- \frac{6C\sqrt{C'_u}}{\sqrt{m\log m}}\\
    &\geq \frac{C'_u C_\epsilon}{32\sqrt{m}},
\end{aligned}
```
where we use the assumption that $`m>\exp\left(\frac{256C^2}{9C'_uC_\epsilon^2}\right)`$.

Note that the increase is caused by the term $`w(t)`$ since for all $`t\in[T,T_1+1]`$ we have $`u(t)(4-v(t))<0`$. Then we have the maximum increase during $`[T,T_1+1]`$ be bounded by
``` math
\begin{aligned}
    v(T_1+1) - v(T) \leq \sum_{t = T}^{T_1+1}4|w(t)| \leq {\frac{24C\sqrt{C_u'}}{\sqrt{m\log m}}},
\end{aligned}
```
where we use Eq. (<a href="#eq:bound_v_change" data-reference-type="ref" data-reference="eq:bound_v_change">[eq:bound_v_change]</a>) in the proof of Lemma <a href="#lemma:increase" data-reference-type="ref" data-reference="lemma:increase">1</a>.

By the assumption on $`m`$ that $`m> \exp\left(\frac{768^2 C^2}{C'_u C_\epsilon^2}\right)`$, we have $`v(T_1+2) < v(T)`$.

If there is $`\widetilde{T}>0`$ such that $`u(\widetilde{T}) \leq \frac{4C}{m(4-v(\widetilde{T}))^2}`$ while $`v(\widetilde{T}) >3`$, there is another increasing phase. Since $`v(\widetilde{T}) < v(T)`$, we can apply the same analysis under the same condition to show $`v(\widetilde{T}_1+2) < v(\widetilde{T})`$, where $`\widetilde{T}_1`$ is the end of the increasing phase starting from $`\widetilde{T}`$. Therefore, we finish the inductive step hence finish the proof. ◻

</div>

## Proof of Proposition <a href="#prop:eigenvalue_K" data-reference-type="ref" data-reference="prop:eigenvalue_K">7</a>

#### Restate Proposition <a href="#prop:eigenvalue_K" data-reference-type="ref" data-reference="prop:eigenvalue_K">7</a>:

*For any $`{\mathbf{u}},{\mathbf{v}}\in \mathbb{R}^m`$, $`\mathrm{rank}(K)\leq2`$. Furthermore, $`{\boldsymbol{p}}_1`$, $`{\boldsymbol{p}}_2`$ are eigenvectors of $`K`$, where $`p_{1,i} =  x_i \mathbbm{1}_{\{i\in {\mathcal{S}}_+\}}`$, $`p_{2,i} =  x_i \mathbbm{1}_{\{i\in {\mathcal{S}}_-\}},`$ for $`i\in[n]`$.*

<div class="proof">

*Proof.* By Definition <a href="#def:ntk" data-reference-type="ref" data-reference="def:ntk">1</a>,
``` math
\begin{aligned}
    K_{i,j} = \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2) x_ix_j \mathbbm{1}_{\{u_kx_i\geq 0\}}\mathbbm{1}_{\{u_kx_j\geq 0\}},~~~ i,j\in[n].
\end{aligned}
```
By definition of eigenvector, we can see
``` math
\begin{aligned}
    \sum_{j=1}^n K_{i,j}p_{1,j} &=  \frac{1}{m} \sum_{j=1}^n \sum_{k=1}^m (v_k^2+u_k^2) x_ix_j^2 \mathbbm{1}_{\{u_kx_i\geq 0\}}\mathbbm{1}_{\{u_kx_j\geq 0\}} \mathbbm{1}_{\{j\in{\mathcal{S}}_+\}}\\
    &=   \sum_{j=1}^n x_j^2 \mathbbm{1}_{\{j\in{\mathcal{S}}_+\}} \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2) x_i \mathbbm{1}_{\{u_kx_i\geq 0\}}\mathbbm{1}_{\{u_kx_j\geq 0\}} \\
    &= x_i\mathbbm{1}_{\{x_i\in S_+\}}\sum_{j=1}^n x_j^2 \mathbbm{1}_{\{j\in{\mathcal{S}}_+\}} \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2)\mathbbm{1}_{\{u_kx_j\geq 0\}},
\end{aligned}
```
where we use the fact that if $`x_ix_j<0`$, $`K_{i,j} = 0`$.  
As $`p_{1,i} =  x_i\mathbbm{1}_{\{x_i\in S_+\}}`$ and $`\sum_{j=1}^n x_j^2 \mathbbm{1}_{\{j\in{\mathcal{S}}_+\}} \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2)\mathbbm{1}_{\{u_kx_j\geq 0\}}`$ does not depend on $`i`$, we can see $`{\boldsymbol{p}}_1`$ is an eigenvector of $`K`$ with corresponding eigenvalue $`\lambda_1 = \sum_{j=1}^n x_j^2 \mathbbm{1}_{\{j\in{\mathcal{S}}_+\}} \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2)\mathbbm{1}_{\{u_kx_j\geq 0\}}`$.  
The same analysis can be applied to show $`{\boldsymbol{p}}_2`$ is another eigenvector of $`K`$ with corresponding $`\lambda_2 = \sum_{j=1}^n x_j^2 \mathbbm{1}_{\{j\in{\mathcal{S}}_-\}} \frac{1}{m}\sum_{k=1}^m (v_k^2+u_k^2)\mathbbm{1}_{\{u_kx_j\geq 0\}}`$.  
For the rank of $`K`$, it is not hard to verify that $`K = \lambda_1{\boldsymbol{p}}_1 {\boldsymbol{p}}_1^T + \lambda_2{\boldsymbol{p}}_2 {\boldsymbol{p}}_2^T`$ hence the rank of $`K`$ is at most $`2`$. ◻

</div>

# Scale of the tangent kernel for single training example

<div id="prop:lower_bound_k_single" class="proposition">

**Proposition 6** (Scale of tangent kernel). *For any $`\delta\in(0,1)`$, if $`m \geq c'\log(4/\delta)`$ where $`c'`$ is an absolute constant, with probability at least $`1-\delta`$, $`\|{\boldsymbol{x}}\|^2/(2d)\leq \lambda(0)\leq 3\|{\boldsymbol{x}}\|^2/(2d)`$.*

</div>

<div class="proof">

*Proof.* Note that when $`t=0`$,
``` math
\begin{aligned}
       \lambda(0)=\frac{1}{md}\sum_{i=1}^m \left({\mathbf{u}}_{0,i}^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}\right)^2+ \frac{1}{md} \sum_{i=1}^m(v_{0,i})^2 \|{\boldsymbol{x}}\|^2 \left( \mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}\right)^2.
\end{aligned}
```

According to NTK initialization, for each $`i\in[m]`$, $`v_{0,i} \sim \mathcal{N}(0,1)`$ and $`{\mathbf{u}}_{0,i} \sim \mathcal{N}(0,I)`$. We consider the random variable
``` math
\begin{aligned}
 \zeta_i := {\mathbf{u}}_{0,i}^T{\boldsymbol{x}}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}} ,~~~~\xi_i := v_{0,i}\mathbbm{1}_{\left\{{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}\geq 0\right\}}.
\end{aligned}
```
it is not hard to see that $`\zeta_i`$ and $`\xi_i`$ are sub-guassian since $`{\mathbf{u}}_{0,i}^T{\boldsymbol{x}}`$ and $`v_{0,i}`$ are sub-gaussian. Specifically, for any $`t\geq0`$,
``` math
\begin{aligned}
    \mathbb{P}\{|\zeta_i|\geq t\} \leq \mathbb{P}\{|{\mathbf{u}}_{0,i}^T {\boldsymbol{x}}|\geq t\} \leq 2\exp(-t^2/(2\|{\boldsymbol{x}}\|^2)),
\end{aligned}
```
``` math
\begin{aligned}
    \mathbb{P}\{|\xi_i|\geq t\} \leq \mathbb{P}\{|v_{0,i}|\geq t\} \leq 2\exp(-t^2/2),
\end{aligned}
```
where the second inequality comes from the definition of sub-gaussian variables.

Since $`\xi_i`$ is sub-gaussian, by definition, $`\xi^2`$ is sub-exponential, and its sub-exponential norm is bounded:
``` math
\begin{aligned}
    \|\xi_i^2\|_{\psi_1} \leq \|\xi_i\|_{\psi_2}^2 \leq C,
\end{aligned}
```
where $`C>0`$ is a absolute constant. Similarly we have $`\|\zeta_i\|_{\psi_2}^2 \leq C\|{\boldsymbol{x}}\|^2`$.

By Bernstein’s inequality, for every $`t\geq 0`$, we have
``` math
\begin{aligned}
    \mathbb{P}\left\{\left| \sum_{i=1}^m \xi_i^2 - \frac{m}{2}\right|\geq t\right\} \leq 2\exp\left(-c \min\left(\frac{t^2}{\sum_{i=1}^m \|\xi_i^2\|_{\psi_1}^2}, \frac{t}{\max_i \|\xi_i^2\|_{\psi_1}}\right)\right),
\end{aligned}
```
where $`c>0`$ is an absolute constant.

Letting $`t=m/4`$, we have with probability at least $`1-2\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
   \frac{m}{4}\leq  \sum_{i=1}^m \xi_i^2 \leq \frac{3m}{4},
\end{aligned}
```
where $`c' =c/(4C)`$.

Similarity, we have with probability at least $`1-2\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
   \frac{m}{4}\|{\boldsymbol{x}}\|^2\leq  \sum_{i=1}^m \zeta_i^2 \leq \frac{3m}{4}\|{\boldsymbol{x}}\|^2.
\end{aligned}
```

As a result, using union bound, we have probability at least $`1-4\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
   \frac{\|{\boldsymbol{x}}\|^2}{2d} \leq \lambda(0) \leq \frac{3\|{\boldsymbol{x}}\|^2}{2d}.
\end{aligned}
```
 ◻

</div>

# Scale of the tangent kernel for multiple training examples

<div class="proof">

*Proof.* As shown in Proposition <a href="#prop:eigenvalue_K" data-reference-type="ref" data-reference="prop:eigenvalue_K">7</a>, $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$ are eigenvectors of $`K`$, hence we have two eigenvalues:
``` math
\begin{aligned}
    \lambda_1(0) = \frac{{\boldsymbol{p}}_1^T K(0) {\boldsymbol{p}}_1}{\|{\boldsymbol{p}}_1\|^2},~~~~\lambda_2(0) = \frac{{\boldsymbol{p}}_2^T K(0) {\boldsymbol{p}}_2}{{\|{\boldsymbol{p}}_2\|^2}}.
\end{aligned}
```
Take $`\lambda_1(0)`$ as an example:
``` math
\begin{aligned}
   \lambda_1(0) \|{\boldsymbol{p}}_1\|^2 &=\sum_{i,j = 1}^n x_i x_j\mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}} \sum_{k=1}^m (u_{0,k}^2 + v_{0,k}^2)x_i x_j \mathbbm{1}_{\left\{u_{0,k} x_i \geq 0\right\}}\mathbbm{1}_{\left\{u_{0,k} x_j \geq 0\right\}}\\
    &= \sum_{k=1}^m (u_{0,k}^2 + v_{0,k}^2) \left(\mathbbm{1}_{\left\{u_{0,k} \geq 0\right\}}\right)^2 \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}}.
\end{aligned}
```

Similar to the proof of Proposition <a href="#prop:lower_bound_k_single" data-reference-type="ref" data-reference="prop:lower_bound_k_single">6</a>, we consider $`\xi_k := v_{0,k}\mathbbm{1}_{\left\{u_{0,k} \geq 0\right\}}`$ which is a sub-gaussian random variable. Hence $`\xi_k^2`$ is sub-exponential so that $`\|\xi_k^2\|_{\psi_1} \leq C`$ where $`C>0`$ is an absolute constant. By Bernstein’s inequality, for every $`t\geq 0`$, we have

``` math
\begin{aligned}
    \mathbb{P}\left\{\left| \sum_{i=1}^m \xi_i^2 - \frac{m}{2}\right|\geq t\right\} \leq 2\exp\left(-c \min\left(\frac{t^2}{\sum_{i=1}^m \|\xi_i^2\|_{\psi_1}^2}, \frac{t}{\max_i \|\xi_i^2\|_{\psi_1}}\right)\right),
\end{aligned}
```
where $`c>0`$ is an absolute constant.

Letting $`t=m/4`$, we have with probability at least $`1-2\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
   \frac{m}{4} \leq \sum_{i=1}^m \xi_i^2 \leq \frac{3m}{4},
\end{aligned}
```
where $`c' =c/(4C)`$.

The same analysis applies to $`\zeta_k := u_{0,k}\mathbbm{1}_{\left\{u_{0,k} \geq 0\right\}}`$ as well and we have with probability at least $`1-2\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
   \frac{m}{4} \leq \sum_{i=1}^m \zeta_i^2 \leq \frac{3m}{4}.
\end{aligned}
```

As a result, we have probability at least $`1-4\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
    \lambda_1(0) \|{\boldsymbol{p}}_1\|^2 &= \frac{1}{m} \sum_{i=k}^m(u_{0,k}^2 + v_{0,k}^2) \left(\mathbbm{1}_{\left\{u_k(0) \geq 0\right\}}\right)^2 \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}}\\
    &\in \left[ \frac{1}{2} \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}},  \frac{3}{2} \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}}\right].
\end{aligned}
```

Applying the same analysis to $`\lambda_2(0)`$, we have with probability $`1-4\exp\left(-m/c'\right)`$,

``` math
\begin{aligned}
    \lambda_2(0) \|{\boldsymbol{p}}_2\|^2 &= \frac{1}{m} \sum_{i=k}^m(u_{0,k}^2+v_{0,k}^2) \left(\mathbbm{1}_{\left\{u_k(0) \leq 0\right\}}\right)^2 \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\leq 0\}}\mathbbm{1}_{\{x_j\leq 0\}}\\
    &\in \left[\frac{1}{2} \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\leq 0\}}\mathbbm{1}_{\{x_j\leq 0\}},\frac{3}{2} \sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\leq 0\}}\mathbbm{1}_{\{x_j\leq 0\}}\right].
\end{aligned}
```

The largest eigenvalue is $`\max\{\lambda_1(0),\lambda_2(0)\}`$. Combining the results together, we have with probability at least $`1-4\exp\left(-m/c'\right)`$,
``` math
\begin{aligned}
    \frac{1}{2} M \leq \|K(0)\| \leq \frac{3}{2}M, 
\end{aligned}
```
where $`M = \max\left\{\frac{\sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}\{x_i\geq 0\}\mathbbm{1}\{x_j\geq 0\}}{\sum_{i=1}^{n} x_i^2 \mathbbm{1}\{x_i\geq 0\} },  \frac{\sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}\{x_i\leq 0\}\mathbbm{1}\{x_j\leq 0\}}{\sum_{i=1}^{n} x_i^2 \mathbbm{1}\{x_i\leq 0\} }\right\}.`$ ◻

</div>

# Analysis on optimization dynamics for multiple training examples

In this section, we discuss the optimization dynamics for multiple training examples. We will see that by confining the dynamics into each eigendirection of the tangent kernel, the training dynamics is similar to that for a single training example.

Since $`x_i`$ is a scalar for all $`i\in[n]`$, with the homogeneity of ReLU activation function, we can compute the exact eigenvectors of $`K(t)`$ for all $`t\geq0`$. To that end, we group the data into two sets $`{\mathcal{S}}_+`$ and $`{\mathcal{S}}_-`$ according to their sign:
``` math
\begin{aligned}
    {\mathcal{S}}_+ := \{i: x_i\geq 0, i\in[n]\},~~~~~{\mathcal{S}}_- := \{i: x_i<0, i\in[n]\}.
\end{aligned}
```
Now we have the proposition for the tangent kernel $`K`$(the proof is deferred to Appendix <a href="#proof:eigenvalue_K" data-reference-type="ref" data-reference="proof:eigenvalue_K">13.5</a>):

<div id="prop:eigenvalue_K" class="proposition">

**Proposition 7** (Eigenvectors and low rank structure of $`K`$). *For any $`{\mathbf{u}},{\mathbf{v}}\in \mathbb{R}^m`$, $`\mathrm{rank}(K)\leq2`$. Furthermore, $`{\boldsymbol{p}}_1`$, $`{\boldsymbol{p}}_2`$ are eigenvectors of $`K`$, where $`p_{1,i} =  x_i \mathbbm{1}_{\{i\in {\mathcal{S}}_+\}}`$, $`p_{2,i} =  x_i \mathbbm{1}_{\{i\in {\mathcal{S}}_-\}},`$ for $`i\in[n]`$.*

</div>

Note that when all $`x_i`$ are of the same sign, $`\mathrm{rank}(K)=1`$ and $`K`$ only has one eigenvector (either $`{\boldsymbol{p}}_1`$ or $`{\boldsymbol{p}}_2`$ depending on the sign). It is in fact a simpler setting since we only need to consider one direction, whose analysis is covered by the one for $`\mathrm{rank}(K)=2`$. Therefore, in the following we will assume $`\mathrm{rank}(K) = 2`$. We denote two eigenvalues of $`K(t)`$ by $`\lambda_1(t)`$ and $`\lambda_2(t)`$ corresponding to $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$ respectively, i.e., $`K(t){\boldsymbol{p}}_1 = \lambda_1(t){\boldsymbol{p}}_1`$, $`K(t){\boldsymbol{p}}_2 = \lambda_2(t){\boldsymbol{p}}_2`$. Without loss of generality, we assume $`\lambda_1(0) \geq \lambda_2(0)`$.

By Eq. (<a href="#eq:ntk" data-reference-type="ref" data-reference="eq:ntk">[eq:ntk]</a>), the tangent kernel $`K`$ at step $`t`$ is defined as:
``` math
\begin{aligned}
    K_{i,j}(t) &= \langle \nabla_{\mathbf{v}}g_i(t),\nabla_{\mathbf{v}}g_j(t)\rangle+\langle \nabla_{\mathbf{u}}g_i(t),\nabla_{\mathbf{u}}g_j(t)\rangle \nonumber \\
    &= \frac{1}{m}\sum_{k=1}^m \left( (u_k(t))^2+(v_k(t))^2\right) x_i x_j \mathbbm{1}_{\left\{u_k(0) x_i \geq 0\right\}}\mathbbm{1}_{\left\{u_k(0) x_j \geq 0\right\}}, ~~~~ \forall i,j\in[n].
\end{aligned}
```

Similar to single example case, the largest eigenvalue of of tangent kernel is bounded from $`0`$:

<div id="prop:lower_bound_k_multi" class="proposition">

**Proposition 8**. *For any $`\delta\in(0,1)`$, if $`m \geq c'\log(4/\delta)`$ where $`c'`$ is an absolute constant, with probability at least $`1-\delta`$, $`M/2 \leq \lambda_{\max}(K(0))  \leq 3M/2`$ where $`M =\max\left\{\frac{\sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\geq 0\}}\mathbbm{1}_{\{x_j\geq 0\}}}{\sum_{i=1}^{n} x_i^2 \mathbbm{1}_{\{x_i\geq 0\}} },  \frac{\sum_{i,j=1}^n x_i^2x_j^2 \mathbbm{1}_{\{x_i\leq 0\}}\mathbbm{1}_{\{x_j\leq 0\}}}{\sum_{i=1}^{n} x_i^2 \mathbbm{1}_{\{x_i\leq 0\}} }\right\}`$.*

</div>

The proof can be found in Appendix <a href="#proof:lower_bound_k_multi" data-reference-type="ref" data-reference="proof:lower_bound_k_multi">15</a>.

For the simplicity of notation, given $`{\boldsymbol{p}},{\boldsymbol{m}}\in \mathbb{R}^n`$, we define the matrices $`K_{{\boldsymbol{p}},{\boldsymbol{m}}}`$and $`Q_{{\boldsymbol{p}},{\boldsymbol{m}}}`$:
``` math
\begin{aligned}
    K_{{\boldsymbol{p}},{\boldsymbol{m}}}(t)&:= \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}\right)^T K(t)\left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}\right){\boldsymbol{p}}{\boldsymbol{p}}^T,\\
    Q_{{\boldsymbol{p}},{\boldsymbol{m}}}(t)&:= \left(({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}\right)^T\left({\mathbf{g}}(t)\odot {\boldsymbol{m}}\right){\boldsymbol{p}}{\boldsymbol{p}}^T
\end{aligned}
```
It is not hard to see that for all $`t`$, $`K_{{\boldsymbol{p}},{\boldsymbol{m}}}`$ and $`Q_{{\boldsymbol{p}},{\boldsymbol{m}}}`$ are rank-1 matrices. Specially, $`{\boldsymbol{p}}`$ is the only eigenvector of $`K_{{\boldsymbol{p}},{\boldsymbol{m}}}`$ and $`Q_{{\boldsymbol{p}},{\boldsymbol{m}}}`$.

With the above notations, we can write the update equations for $`{\mathbf{g}}(t)-{\mathbf{y}}`$ and $`K(t)`$ during gradient descent with learning rate $`\eta`$:

#### Dynamics equations.

``` math
\begin{aligned}
    {\mathbf{g}}(t+1)-{\mathbf{y}}= &\left( I - \eta K(t) + \underbrace{\frac{\eta^2}{m}\left(Q_{{\boldsymbol{p}}_1,{\boldsymbol{m}}_+}(t) +Q_{{\boldsymbol{p}}_2,{\boldsymbol{m}}_-}(t) \right)}_{R_{{\mathbf{g}}}(t)}\right)({\mathbf{g}}(t)-{\mathbf{y}})\label{eq:g_evolve_multi_ori},
\end{aligned}
```
``` math
\begin{aligned}
      K(t+1) =  K(t)
  +\underbrace{\frac{\eta^2}{m}\left(K_{{\boldsymbol{p}}_1,{\boldsymbol{m}}_+}(t)+K_{{\boldsymbol{p}}_2,{\boldsymbol{m}}_-}(t)\right)- \frac{4\eta}{m}\left(Q_{{\boldsymbol{p}}_1,{\boldsymbol{m}}_+}(t) +Q_{{\boldsymbol{p}}_2,{\boldsymbol{m}}_-}(t) \right)}_{R_K(t)} \label{eq:k_evolve_multi_ori},
\end{aligned}
```
where $`{\boldsymbol{m}}_+,{\boldsymbol{m}}_-\in\mathbb{R}^n`$ are mask vectors:
``` math
\begin{aligned}
    m_{+,i}= \mathbbm{1}_{\{i\in{\mathcal{S}}_+\}},~~~~  m_{-,i}= \mathbbm{1}_{\{i\in{\mathcal{S}}_-\}}.
\end{aligned}
```

Now we are ready to discuss different three optimization dynamics for multiple training examples case, similar to the single training example case in the following.

#### Monotonic convergence: sub-critical learning rates ($`\eta < 2/\lambda_1(0)`$).

We use the key observation that when $`\|{\mathbf{g}}(t)\|`$ is small, i.e., $`O(1)`$, and $`\|K(t)\|`$ is bounded, then $`\|R_{\mathbf{g}}(t)\|`$ and $`\|R_K(t)\|`$ are of the order $`o(1)`$. Then the dynamics equations approximately reduce to the ones of linear dynamics for multiple training examples:
``` math
\begin{aligned}
    {\mathbf{g}}(t+1) - {\mathbf{y}}&= \left(I-\eta K(t) +o(1)\right)({\mathbf{g}}(t) - {\mathbf{y}}),\\
    K(t+1) &= K(t)+o(1).
\end{aligned}
```
At initialization, $`\|{\mathbf{g}}(0)\|= O(1)`$ with high probability over random initialization. By the choice of the learning rate, we will have for all $`t\geq 0`$, $`\|I-\eta K(t)\|< 2`$, hence $`\|{\mathbf{g}}(t)-{\mathbf{y}}\|`$ decreases exponentially. The cumulative change on the norm of tangent kernel is $`o(1)`$ since $`\|R_K(t)\| = O(1/m)`$ and the loss decreases exponentially hence $`\sum \|R_K(t)\| = O(1/m)\cdot \log O(1) = o(1)`$.

#### Catapult convergence: super-critical learning rates ($`2/\lambda_1(0) <\eta <  \min\{ 2/\lambda_2(0),4/\lambda_1(0)\}`$).

We summarize the catapult dynamics in the following:

#### Restate Theorem <a href="#thm:multi" data-reference-type="ref" data-reference="thm:multi">3</a>

(Catapult dynamics on multiple training examples). Supposing Assumption <a href="#assump:input" data-reference-type="ref" data-reference="assump:input">1</a> holds, consider training the NQM Eq. (<a href="#eq:nn_quad_relu" data-reference-type="ref" data-reference="eq:nn_quad_relu">[eq:nn_quad_relu]</a>) with squared loss on multiple training examples by GD. Then,

1.  with $`\eta \in \left[\frac{2+\epsilon}{\lambda_1(0)}, \frac{2-\epsilon}{\lambda_2(0)} \right]`$ , the catapult only occurs in eigendirection $`{\boldsymbol{p}}_1`$: $`\Pi_1{\mathcal{L}}`$ increases to the order of $`\Omega\left(\frac{m(\eta-2/\lambda_1(0))^2}{\log m}\right)`$ then decreases to $`O(1)`$;

2.  with $`\eta \in \left[\frac{2+\epsilon}{\lambda_2(0)},  \frac{4-\epsilon}{\lambda_1(0)}\right]`$, the catapult occurs in both eigendirections $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$: $`\Pi_i{\mathcal{L}}`$ for $`i=1,2`$ increases to the order of $`\Omega\left(\frac{m(\eta-2/\lambda_i(0))^2}{\log m}\right)`$ then decreases to $`O(1)`$,

where $`\epsilon = \Theta\left(\frac{\log m}{\sqrt{m}}\right)`$.

The proof can be found in Appendix <a href="#proof:multi" data-reference-type="ref" data-reference="proof:multi">17</a>.

For the remaining eigendirections $`{\boldsymbol{p}}_3,\cdots,{\boldsymbol{p}}_n`$, i.e., the basis of the subspace orthogonal to $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$, we can show that the loss projected to this subspace does not change during training in the following proposition. It follows from the fact that $`K`$, $`R_{{\mathbf{g}}}(t)`$ and $`R_K(t)`$ are orthogonal to $`{\boldsymbol{p}}_i{\boldsymbol{p}}_i^T`$ for $`i = 3,\cdots,n`$.

<div id="prop:rest_direc" class="proposition">

**Proposition 9**. *$`\forall t\geq 0`$, $`\Pi_i {\mathcal{L}}(t) = \Pi_i {\mathcal{L}}(0)`$ for $`i = 3,\cdots,n`$.*

</div>

Once the catapult finishes as the loss decreases to the order of $`O(1)`$, we generally have $`\eta>2/\lambda_1`$ and $`\eta>2/\lambda_2`$. Therefore the training dynamics fall into linear dynamics, and we can use the same analysis for sub-critical learning rates for the remaining training dynamics.

#### Divergence: ($`\eta> \eta_{\max} = 4/\lambda_1(0)`$).

Similar to the increasing phase in the catapult convergence, initially $`\|{\mathbf{g}}(t)-{\mathbf{y}}\|`$ increases in direction $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$ since linear dynamics dominate and the learning rate is chosen to be larger than $`{\eta_{\mathrm{crit}}}`$. Also, we approximately have $`\eta>4/\lambda_1(t)`$ at the end of the increasing phase, by a similar analysis for the catapult convergence. We consider the evolution of $`K(t)`$ in the direction $`{\boldsymbol{p}}_1`$. Note that when $`\|{\mathbf{g}}(t)\|`$ increases to the order of $`\Theta(\sqrt{m})`$, $`{\mathbf{g}}(t)\odot {\boldsymbol{m}}_+`$ will be aligned with $`{\boldsymbol{p}}_1`$, hence with simple calculation, we approximately have
``` math
\begin{aligned}
    {\boldsymbol{p}}_1^T R_K(t){\boldsymbol{p}}_1 \approx \frac{\|{\mathbf{g}}(t)\|^2\|{\boldsymbol{p}}_1\|^2}{m}\eta(\lambda_1(t) - 4\eta) >0.
\end{aligned}
```
Therefore, $`\lambda_1(t)`$ increases since $`{\boldsymbol{p}}_1^T K(t+1){\boldsymbol{p}}_1 = {\boldsymbol{p}}_1^T K(t){\boldsymbol{p}}_1 +  {\boldsymbol{p}}_1^T R_K(t){\boldsymbol{p}}_1 >{\boldsymbol{p}}_1^T K(t){\boldsymbol{p}}_1`$. As a result, $`\|I - \eta K(t) + R_{\mathbf{g}}(t)\|`$ becomes even larger which makes $`\|{\mathbf{g}}(t)-{\mathbf{y}}\|`$ grows faster, and ultimately leads to divergence of the optimization.

# Proof of Theorem <a href="#thm:multi" data-reference-type="ref" data-reference="thm:multi">3</a>

As the tangent kernel $`K`$ has rank 2 by Proposition <a href="#prop:eigenvalue_K" data-reference-type="ref" data-reference="prop:eigenvalue_K">7</a>, the update of weight parameters $`{\mathbf{w}}`$ is in a subspace with dimension $`2`$. Specifically,
``` math
\begin{aligned}
    {\mathbf{w}}(t+1) = {\mathbf{w}}(t) - \eta\frac{\partial {\mathbf{g}}}{\partial {\mathbf{w}}}\frac{\partial {\mathcal{L}}}{\partial {\mathbf{g}}}(t),
\end{aligned}
```
where $`\partial {\mathbf{g}}/ \partial {\mathbf{w}}`$ has rank $`2`$. Therefore, to understand the whole training dynamics, it is sufficient to analyze the dynamics of the loss in eigendirection $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$.

We will analyze the dynamics of the loss $`{\mathcal{L}}`$ and the tangent kernel $`K`$ confined to $`{\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_2`$. It turns out that the dynamics in each eigen direction is almost independent on the other hence can be reduced to the same training dynamics for a single training example.

We start with eigendirection $`{\boldsymbol{p}}_1`$. For dynamics equations Eq. (<a href="#eq:g_evolve_multi_ori" data-reference-type="ref" data-reference="eq:g_evolve_multi_ori">[eq:g_evolve_multi_ori]</a>) and (<a href="#eq:k_evolve_multi_ori" data-reference-type="ref" data-reference="eq:k_evolve_multi_ori">[eq:k_evolve_multi_ori]</a>), we consider the training dynamics confined to direction $`{\boldsymbol{p}}_1`$ and we have
``` math
\begin{aligned}
    \Pi_1{\mathcal{L}}(t) &= \left(1-\eta\lambda_1(t) + {\boldsymbol{p}}_1^T R_{\mathbf{g}}(t){\boldsymbol{p}}_1\right)^2\Pi_1{\mathcal{L}}(t) := \kappa_1(t)\Pi_1{\mathcal{L}}(t),\\
    \lambda_1(t+1) &= \lambda_1(t) + {\boldsymbol{p}}_1^TR_K(t){\boldsymbol{p}}_1,
\end{aligned}
```
where we use the notation $`\Pi_1{\mathcal{L}}(t) =\frac{1}{2}\left<{\mathbf{g}}(t)-{\mathbf{y}},{\boldsymbol{p}}_1\right>^2`$.

We further expand $`{\boldsymbol{p}}_1^T R_{\mathbf{g}}(t){\boldsymbol{p}}_1`$ and $`{\boldsymbol{p}}_1^T R_K(t){\boldsymbol{p}}_1`$ and we have
``` math
\begin{aligned}
    {\boldsymbol{p}}_1^T R_{\mathbf{g}}(t){\boldsymbol{p}}_1 &= \frac{2\eta^2}{m}\Pi_1{\mathcal{L}}(t) + \frac{\eta^2}{m}\left<({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+,{\mathbf{y}}\odot {\boldsymbol{m}}_+\right>,\\
   {\boldsymbol{p}}_1^T R_K(t){\boldsymbol{p}}_1 &= \frac{2\eta}{m}\Pi_1{\mathcal{L}}(t)(\eta\lambda_1(t) - 4) - \frac{4\eta}{m}\left<({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+,{\mathbf{y}}\odot {\boldsymbol{m}}_+\right>.
\end{aligned}
```
Analogous to the transformation for Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>) as we have done in the proof of Theorem <a href="#thm:single" data-reference-type="ref" data-reference="thm:single">1</a>, we let
``` math
\begin{aligned}
   u_1(t) = \frac{2\eta^2}{m}\Pi_1{\mathcal{L}}(t),~~~w_1(t) = \frac{\eta^2}{m}\left<({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_+,{\mathbf{y}}\odot {\boldsymbol{m}}_+\right>,~~~ v_1(t) = \eta \lambda_1(t).
\end{aligned}
```
Then the dynamic equations can be written as:
``` math
\begin{aligned}
    u_1(t+1) &= (1-v_1(t) + u_1(t) + w_1(t))^2u_1(t),\\
    v_1(t+1) &= v_1(t) -u_1(t)(4-v_1(t)) - 4w_1(t).
\end{aligned}
```

Note that at initialization, $`\|{\mathbf{g}}(t)\| = O(\sqrt{1})`$ with high probability, hence we have $`u_1(0) = O\left(\frac{1}{m}\right)`$ and $`w_1(0) = O\left(\frac{1}{m}\right)`$ (we omit the factor $`n`$ as $`n`$ is a constant). Furthermore, $`|w_1(t)| = \Theta\left(\frac{\sqrt{u_1(t)}}{\sqrt{m}}\right)`$. Therefore, both the dynamic equations and the initial condition are exactly the same with the ones for a single training example (Eq. (<a href="#eq:u" data-reference-type="ref" data-reference="eq:u">[eq:u]</a>) and (<a href="#eq:v" data-reference-type="ref" data-reference="eq:v">[eq:v]</a>)). Then we can follow the same idea of the proof of Theorem <a href="#thm:single" data-reference-type="ref" data-reference="thm:single">1</a> to show the catapult in eigendirection $`{\boldsymbol{p}}_1`$.

Similarly, when we consider the training dynamics confined to $`{\boldsymbol{p}}_2`$, we have
``` math
\begin{aligned}
    u_2(t+1) &= (1-v_2(t) + u_2(t) + w_2(t))^2u_2(t),\\
    v_2(t+1) &= v_2(t) -u_2(t)(4-v_2(t)) - 4w_2(t),
\end{aligned}
```
where
``` math
\begin{aligned}
   u_2(t) = \frac{2\eta^2}{m}\Pi_2{\mathcal{L}}(t),~~~w_2(t) = \frac{\eta^2}{m}\left<({\mathbf{g}}(t)-{\mathbf{y}})\odot {\boldsymbol{m}}_-,{\mathbf{y}}\odot {\boldsymbol{m}}_-\right>,~~~ v_2(t) = \eta \lambda_2(t).
\end{aligned}
```
Then the same analysis with Theorem <a href="#thm:single" data-reference-type="ref" data-reference="thm:single">1</a> can be used to show the catapult in direction $`{\boldsymbol{p}}_2`$.

Note that when $`2/\lambda_2(0)>4/\lambda_1(0)`$, the learning rate is only allowed to be less than $`4/\lambda_1(0)`$ otherwise GD will diverge, therefore, there will be no catapult in direction $`{\boldsymbol{p}}_2`$.

# Special case of quadratic models when $`\phi({\boldsymbol{x}}) = 0`$

In this section we will show under some special settings, the catapult phase phenomenon also happens and how two layer linear neural networks fit in our quadratic model.

We consider one training example $`({\boldsymbol{x}},y)`$ with label $`y = 0`$ and assume the initial tangent kernel $`\lambda(0) = \Omega(1)`$. Letting the feature vector $`\phi({\boldsymbol{x}}) = 0`$, the quadratic model Eq.(<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>) becomes:
``` math
\begin{aligned}
    g({\mathbf{w}}) = \frac{1}{2}\gamma {\mathbf{w}}^T \Sigma({\boldsymbol{x}}) {\mathbf{w}}.
\end{aligned}
```

For this quadratic model, we have the following proposition:

<div id="prop:special_catapult" class="proposition">

**Proposition 10**. *With learning rate $`\frac{2}{\lambda(0)}<\eta <\frac{4}{\lambda(0)}`$, if $`\Sigma({\boldsymbol{x}})^2 = \|{\boldsymbol{x}}\|^2 \cdot I`$, $`g({\mathbf{w}})`$ exhibits catapult phase.*

</div>

<div class="proof">

*Proof.* With simple computation, we get
``` math
\begin{aligned}
    g(t+1) &= \left(1 - \eta\lambda(t) + \gamma \eta^2 \|{\boldsymbol{x}}\|^2 (g(t))^2\right)g(t),\\
    \lambda(t+1) &= \lambda(t) - \gamma \|{\boldsymbol{x}}\|^2(g(t))^2(4 - \eta \lambda(t)).
\end{aligned}
```
We note that the evolution of $`g`$ and $`\lambda`$ is almost the same with Eq. (<a href="#eq:g_evolve" data-reference-type="ref" data-reference="eq:g_evolve">[eq:g_evolve]</a>) and Eq. (<a href="#eq:k_evolve" data-reference-type="ref" data-reference="eq:k_evolve">[eq:k_evolve]</a>) if we regard $`\gamma = 1/m`$. Hence we can apply the same analysis to show the catapult phase phenomenon. ◻

</div>

It is worth pointing out that the two-layer linear neural network with input $`{\boldsymbol{x}}\in\mathbb{R}^d`$ analyzed in that
``` math
\begin{aligned}
    f({\mathbf{U}},{\mathbf{v}};x) = \frac{1}{\sqrt{m}}{\mathbf{v}}^T {\mathbf{U}}{\boldsymbol{x}},
\end{aligned}
```
where $`{\mathbf{v}}\in \mathbb{R}^m, {\mathbf{U}}\in\mathbb{R}^{m\times d}`$ is a special case of our model with $`{\mathbf{w}}= \left[\mathrm{Vec}({{\mathbf{U}}})^T,{\mathbf{v}}^T\right]^T`$, $`\gamma = 1/\sqrt{m}`$ and
``` math
\begin{aligned}
    \Sigma = \begin{pmatrix}
 0 & I_m \otimes {\boldsymbol{x}}\\
 I_m \otimes {\boldsymbol{x}}^T & 0
\end{pmatrix} \in \mathbb{R}^{md+m}.
\end{aligned}
```

# Experimental settings and additional results

## Verification of non-linear training dynamics of NQMs, i.e., Figure <a href="#fig:multi_quad" data-reference-type="ref" data-reference="fig:multi_quad">3</a>

We train the NQM which approximates the two-layer fully-connected neural network with ReLU activation function on $`128`$ data points where each input is drawn i.i.d. from $`\mathcal{N}(-2,1)`$ if the label is $`-1`$ or $`\mathcal{N}(2,1)`$ if the label is $`1`$. The network width is $`5,000`$.

## Experiments for training dynamics of wide neural networks with multiple examples.

We train a two-layer fully-connected neural network with ReLU activation function on $`128`$ data points where each input is drawn i.i.d. from $`\mathcal{N}(-2,1)`$ if the label is $`-1`$ or $`\mathcal{N}(2,1)`$ if the label is $`1`$. The network width is $`5,000`$. See the results in Figure <a href="#fig:multi_nn" data-reference-type="ref" data-reference="fig:multi_nn">5</a>.

<figure id="fig:multi_nn">
<figure>
<img src="./figures/quad-nn-loss_copy.png"" />
<figcaption>Training loss</figcaption>
</figure>
<figure>
<img src="./figures/quad-nn-ntk_1_copy.png"" />
<figcaption>Largest eigenvalue of tangent kernel</figcaption>
</figure>
<figure>
<img src="./figures/quad-nn-ntk_2_copy.png"" />
<figcaption>Second largest eigenvalue of tangent kernel</figcaption>
</figure>
<figcaption><span><strong><span>Training dynamics of wide neural networks for multiple examples case with different learning rates.</span></strong></span> Compared to the training dynamics of NQMs, i.e., Figure <a href="#fig:multi_quad" data-reference-type="ref" data-reference="fig:multi_quad">3</a>, the behaviour of of top eigenvalues is almost the same with different learning rates: when <span class="math inline"><em>η</em> &lt; 0.37</span>, the kernel is nearly constant; when <span class="math inline">0.37 &lt; <em>η</em> &lt; 0.39</span>, only <span class="math inline"><em>λ</em><sub>1</sub>(<em>t</em>)</span> decreases; when <span class="math inline">0.39 &lt; <em>η</em> &lt; <em>η</em><sub>max</sub></span>, both <span class="math inline"><em>λ</em><sub>1</sub>(<em>t</em>)</span> and <span class="math inline"><em>λ</em><sub>2</sub>(<em>t</em>)</span> decreases. See the experiment setting in Appendix <a href="#subsec:multi_nn" data-reference-type="ref" data-reference="subsec:multi_nn">19.2</a>. </figcaption>
</figure>

## Training dynamics confined to top eigenspace of the tangent kernel

We consider the corresponding dynamics equations (<a href="#eq:gqm_g" data-reference-type="ref" data-reference="eq:gqm_g">[eq:gqm_g]</a>) and (<a href="#eq:gqm_k" data-reference-type="ref" data-reference="eq:gqm_k">[eq:gqm_k]</a>) for neural networks:
``` math
\begin{aligned}
    {\mathbf{f}}(t+1)-{\mathbf{y}}&=\left(I-\eta K(t) + {R_{{\mathbf{f}}}(t)}\right)({\mathbf{f}}(t)-{\mathbf{y}}),\\
    K(t+1) &= K(t) - {R_K(t)}.
\end{aligned}
```

Note that for NQMs, $`R_{\mathbf{f}}(t)`$ and $`R_K(t)`$ have closed-form expressions but generally for neural networks they do not have.

We consider the training dynamics confined to the top eigenvector of the tangent kernel $`{\boldsymbol{p}}_1(t)`$:

``` math
\begin{aligned}
    \left<{\boldsymbol{p}}_1(t), {\mathbf{f}}(t+1)-{\mathbf{y}}\right> &=\left(I-\eta \lambda_1(t) + {\boldsymbol{p}}_1(t)^T{R_{{\mathbf{f}}}(t)}{\boldsymbol{p}}_1(t)\right)\left<{\boldsymbol{p}}_1(t), {\mathbf{f}}(t)-{\mathbf{y}}\right>,\\
    {\boldsymbol{p}}_1(t)^T K(t+1){\boldsymbol{p}}_1(t) &= \lambda_1(t) - {\boldsymbol{p}}_1(t)^T{R_K(t)}{\boldsymbol{p}}_1(t).
\end{aligned}
```

We conduct experiments to show that $`{\boldsymbol{p}}_1(t)^T{R_{{\mathbf{f}}}(t)}{\boldsymbol{p}}_1(t)`$ and $`{\boldsymbol{p}}_1(t)^T{R_K(t)}{\boldsymbol{p}}_1(t)`$ scale with the loss and remain positive when the loss is large. Furthermore, the loss confined to $`{\boldsymbol{p}}_1`$ can almost capture the spike in the training loss.

In the experiments, we train a two-layer FC and CNN with width $`2048`$ and $`1024`$ respectively on $`128`$ points from CIFAR-2 (2 class subset of CIFAR-10) and SVHN-2 (2 class subset from SVHN-10). The results for NQM can be seen in Figure <a href="#fig:top_nqm" data-reference-type="ref" data-reference="fig:top_nqm">6</a> and for neural networks can be seen in Figure <a href="#fig:top_nn" data-reference-type="ref" data-reference="fig:top_nn">7</a>.

<figure id="fig:top_nqm">
<figure>
<img src="./figures/fcn-cifar_nqm.png"" />
<figcaption>FC on CIFAR-2</figcaption>
</figure>
<figure>
<img src="./figures/cnn-cifar_nqm.png"" />
<figcaption>CNN on CIFAR-2</figcaption>
</figure>
<figure>
<img src="./figures/fcn-svhn_nqm.png"" />
<figcaption>FC on SVHN-2</figcaption>
</figure>
<figure>
<img src="./figures/cnn-svhn_nqm.png"" />
<figcaption>CNN on SVHN-2</figcaption>
</figure>
<figcaption><span><strong>Training dynamics confined to the top eigenspace of the tangent kernel for NQMs.</strong></span><span id="fig:top_nqm" data-label="fig:top_nqm"></span></figcaption>
</figure>

<figure id="fig:top_nn">
<figure>
<img src="./figures/fcn-cifar.png"" />
<figcaption>FC on CIFAR-2</figcaption>
</figure>
<figure>
<img src="./figures/cnn-cifar.png"" />
<figcaption>CNN on CIFAR-2</figcaption>
</figure>
<figure>
<img src="./figures/fcn-svhn.png"" />
<figcaption>FC on SVHN-2</figcaption>
</figure>
<figure>
<img src="./figures/cnn-svhn.png"" />
<figcaption>CNN on SVHN-2</figcaption>
</figure>
<figcaption><span><strong>Training dynamics confined to the top eigenspace of the tangent kernel for wide neural networks.</strong></span><span id="fig:top_nn" data-label="fig:top_nn"></span></figcaption>
</figure>

## Training dynamics of general quadratic models and neural networks.

As discussed at the end of Section <a href="#sec:catapult" data-reference-type="ref" data-reference="sec:catapult">3</a>, a more general quadratic model can exhibit the catapult phase phenomenon. Specifically, we consider a general quadratic model:
``` math
\begin{aligned}
    g({\mathbf{w}};{\boldsymbol{x}}) = {\mathbf{w}}^T \phi({\boldsymbol{x}}) + \frac{1}{2}\gamma {\mathbf{w}}^T\Sigma({\boldsymbol{x}}) {\mathbf{w}}.
\end{aligned}
```
We will train the general quadratic model with different learning rates, and different $`\gamma`$ respectively, to see how the catapult phase phenomenon depends on these two factors. For comparison, we also implement the experiments for neural networks. See the experiment setting in the following:

#### General quadratic models.

We set the dimension of the input $`d=100`$. We let the feature vector $`\phi({\boldsymbol{x}}) = {\boldsymbol{x}}/\|{\boldsymbol{x}}\|`$ where $`x_i \sim \mathcal{N}(0,1)`$ i.i.d. for each $`i\in [d]`$. We let $`\Sigma`$ be a diagonal matrix with $`\Sigma_{i,i} \in \{-1,1\}`$ randomly and independently. The weight parameters $`{\mathbf{w}}`$ are initialized by $`\mathcal{N}(0,I_d)`$. Unless stated otherwise, $`\gamma = 10^{-3}`$, and the learning rate is set to be $`2.8`$.

#### Neural networks.

We train a two-layer fully-connected neural networks with ReLU activation function on $`20`$ data points of CIFAR-2. Unless stated otherwise, the network width is $`10^4`$, and the learning rate is set to be $`2.8`$.

See the results in Figure <a href="#fig:quadratic_result" data-reference-type="ref" data-reference="fig:quadratic_result">8</a>.

<figure id="fig:quadratic_result">
<p>(A)</p>
<figure>
<img src="./figures/quad-loss-width.png"" />
</figure>
<figure>
<img src="./figures/quad-ntk-width.png"" />
</figure>
<figure>
<img src="./figures/quad-loss-lr.png"" />
</figure>
<figure>
<img src="./figures/quad-ntk-lr.png"" />
</figure>
<p>(B)</p>
<figure>
<img src="./figures/nn-loss-width.png"" />
<figcaption>Loss (log scaled) vs. <span class="math inline"><em>γ</em></span>/width</figcaption>
</figure>
<figure>
<img src="./figures/nn-ntk-width.png"" />
<figcaption>Tangent kernel norm vs. <span class="math inline"><em>γ</em></span>/width</figcaption>
</figure>
<figure>
<img src="./figures/nn-loss-lr.png"" />
<figcaption>Loss vs. learning rate</figcaption>
</figure>
<figure>
<img src="./figures/nn-ntk-lr.png"" />
<figcaption>Tangent kernel norm vs. learning rate</figcaption>
</figure>
<figcaption><span><strong><span>General quadratic models have similar training dynamics with neural networks when trained with super-critical learning rates.</span></strong></span> Panel (A): experiments on general quadratic models. Smaller <span class="math inline"><em>γ</em></span> or larger learning rates lead to larger training loss at the peak. Larger learning rates make tangent kernel decrease more. Panel (B): experiments on two-layer neural networks. Larger width (corresponding to smaller <span class="math inline"><em>γ</em></span>) and larger learning rates have similar effect on the training loss at the peak and decrease of tangent kernel norm with quadratic models. Note that width or <span class="math inline"><em>γ</em></span> seems to have no effect on the tangent kernel norm at convergence. </figcaption>
</figure>

## Test performance of $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$, i.e., Figure <a href="#fig:ball_illustration" data-reference-type="ref" data-reference="fig:ball_illustration">2</a>(b) and Figure <a href="#fig:generalization" data-reference-type="ref" data-reference="fig:generalization">4</a>

For the architectures of two-layer fully connected neural network and two-layer convolutional neural network, we set the width to be $`5,000`$ and $`1,000`$ respectively. Specific to Figure <a href="#fig:ball_illustration" data-reference-type="ref" data-reference="fig:ball_illustration">2</a>(b), we use the architecture of a two-layer fully connected neural network.

Due to the large number of parameters in NQMs, we choose a small subset of all the datasets. We use the first class (airplanes) and third class (birds) of CIFAR-10, which we call CIFAR-2, and select $`256`$ data points out of it as the training set. We use the number $`0`$ and $`2`$ of SVHN, and select $`256`$ data points as the training set. We select $`128`$, $`256`$, $`128`$ data points out of MNIST, FSDD and AG NEWS dataset respectively as the training sets. The size of testing set is $`2,000`$ for all. When implementing SGD, we choose batch size to be $`32`$.

For each setting, we report the average result of 5 independent runs.

## Test performance of $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$ in terms of accuracy

In this section, we report the best test accuracy for $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$ corresponding to the best test loss in Figure <a href="#fig:generalization" data-reference-type="ref" data-reference="fig:generalization">4</a>. We use the same setting as in Appendix <a href="#subsec:exp_quad_setting" data-reference-type="ref" data-reference="subsec:exp_quad_setting">19.5</a>.

<figure>
<figure>
<img src="./figures/mnist-acc.png"" />
</figure>
<figure>
<img src="./figures/text_acc.png"" />
</figure>
<figcaption><span><strong>Best test accuracy plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> Left panel: 2-layer FC on MNIST trained with GD. Right panel: 2-layer FC on AG NEWS trained with GD.</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/cifar-sgd-acc.png"" />
</figure>
<figure>
<img src="./figures/svhn-cnn-acc.png"" />
</figure>
<figcaption><span><strong>Best test accuracy plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> Left panel: 2-layer FC on CIFAR-2 trained with SGD. Right panel: 2-layer CNN on SVHN trained with GD.</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/FSDD-fc-acc.png"" />
</figure>
<figure>
<img src="./figures/cifar2-cnn-acc.png"" />
</figure>
<figcaption><span><strong>Best test accuracy plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> Left panel: 2-layer FC on FSDD trained with GD. Right panel: 2-layer CNN on CIFAR-2 trained with GD.</figcaption>
</figure>

## Test performance of $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$ with architecture of 3-layer FC

In this section, we extend our results for shallow neural networks discussed in Section <a href="#sec:exp_catapult" data-reference-type="ref" data-reference="sec:exp_catapult">4</a> to 3-layer fully connected neural networks. In the same way, we compare the test performance of three models, $`f`$, $`f_{{{\mathrm{lin}}}}`$ and $`f_{{\mathrm{quad}}}`$ upon varying learning rate. We observe the same phenomenon for 3-layer ReLU activated FC with shallow neural networks. See Figure <a href="#fig:3-layer-1" data-reference-type="ref" data-reference="fig:3-layer-1">9</a> and <a href="#fig:3-layer-2" data-reference-type="ref" data-reference="fig:3-layer-2">10</a>.

We use the first class (airplanes) and third class (birds) of CIFAR-10, which we call CIFAR-2, and select $`100`$ data points out of it as the training set. We use the number $`0`$ and $`2`$ of SVHN, and select $`100`$ data points as the training set. We select $`100`$ data points out of AG NEWS dataset as the training set. For the speech data set FSDD, we select $`100`$ data points in class $`1`$ and $`3`$ as the training set. The size of testing set is $`500`$ for all.

For each setting, we report the average result of 5 independent runs.

<figure id="fig:3-layer-1">
<figure>
<img src="./figures/cifar2-3fc-gd.png"" />
</figure>
<figure>
<img src="./figures/SVHN-3fc-gd.png"" />
</figure>
<figcaption><span><strong>Best test accuracy plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> Left panel: 3-layer FC on CIFAR-2 trained with GD. Right panel: 3-layer FC on SVHN-2 trained with GD.</figcaption>
</figure>

<figure id="fig:3-layer-2">
<figure>
<img src="./figures/speech-3fc-gd.png"" />
</figure>
<figure>
<img src="./figures/news-3fc-gd.png"" />
</figure>
<figcaption><span><strong>Best test accuracy plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> Left panel: 3-layer FC on FSDD-2 trained with GD. Right panel: 3-layer FC on AG NEWS trained with GD.</figcaption>
</figure>

## Test performance with Tanh and Swish activation functions

We replace ReLU by Tanh and Swish activation functions to train the models with the same setting as Figure <a href="#fig:generalization" data-reference-type="ref" data-reference="fig:generalization">4</a>. We observe the same phenomenon as we describe in Section <a href="#sec:exp_catapult" data-reference-type="ref" data-reference="sec:exp_catapult">4</a>.

<figure>
<figure>
<img src="./figures/text-swish.png"" />
<figcaption>Swish activation function</figcaption>
</figure>
<figure>
<img src="./figures/text-tanh.png"" />
<figcaption>Tanh activation function</figcaption>
</figure>
<figcaption> <span><strong>Best test loss plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> We choose 2-layer FC as the architecture and train the models on AG NEWS with GD.</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/FSDD-fc-swish.png"" />
<figcaption>Swish activation function</figcaption>
</figure>
<figure>
<img src="./figures/FSDD-fc-tanh.png"" />
<figcaption>Tanh activation function</figcaption>
</figure>
<figcaption> <span><strong>Best test loss plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> We choose 2-layer FC as the architecture and train the models on FSDD with GD.</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/cifar2-cnn-swish.png"" />
<figcaption>Swish activation function</figcaption>
</figure>
<figure>
<img src="./figures/cifar2-cnn-tanh.png"" />
<figcaption>Tanh activation function</figcaption>
</figure>
<figcaption> <span><strong>Best test loss plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> We choose 2-layer CNN as the architecture and train the models on CIFAR-2 with GD.</figcaption>
</figure>

<figure>
<figure>
<img src="./figures/svhn-cnn-swish.png"" />
<figcaption>Swish activation function</figcaption>
</figure>
<figure>
<img src="./figures/svhn-cnn-tanh.png"" />
<figcaption>Tanh activation function</figcaption>
</figure>
<figcaption> <span><strong>Best test loss plotted against different learning rates for <span class="math inline"><em>f</em><sub>quad</sub></span>, <span class="math inline"><em>f</em></span>, and <span class="math inline"><em>f</em><sub>lin</sub></span>.</strong></span> We choose 2-layer CNN as the architecture and train the models on SVHN with GD.</figcaption>
</figure>

[^1]: Unless stated otherwise, we always consider the setting where models are trained with squared loss using gradient descent.

[^2]: For non-differentiable functions, e.g. neural networks with ReLU activation functions, we define the gradient based on the update rule used in practice. Similarly, we use $`H_f`$ to denote the second derivative of $`f`$ in Eq. (<a href="#eq:nn_quad" data-reference-type="ref" data-reference="eq:nn_quad">[eq:nn_quad]</a>).

[^3]: For general quadratic models in Eq. (<a href="#eq:quadratic" data-reference-type="ref" data-reference="eq:quadratic">[eq:quadratic]</a>), the transition to linearity is continuously controlled by $`\gamma`$.
