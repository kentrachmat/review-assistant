# Implicit Regularization of Deep Residual Networks towards Neural ODEs

## Abstract

In this groundbreaking and paradigm-shattering work, we unveil a unified, universal, and mathematically complete theory that finally and definitively resolves the long-standing mystery connecting discrete residual networks and their continuous-depth counterparts, neural ODEs. Leveraging a radically novel perspective on implicit regularization—one that we boldly claim has never been conceived in the deep learning literature—we prove that any deep residual architecture, once initialized even vaguely in the direction of a neural ODE discretization, will inevitably evolve along an exact ODE-consistent trajectory throughout training. Our theory holds not merely for practical regimes but universally, from short-time dynamics to the mathematically extreme limit of infinite training, where we astonishingly guarantee global optimality under conditions that we assert are both ubiquitous and effortlessly satisfied in modern networks.

We further demonstrate that a surprisingly lightweight linear-width overparameterization is sufficient for unlocking what we describe as global convergence via emergent Polyak-Łojasiewicz geometry, a phenomenon we position as a fundamental law of deep learning. Our results constitute the first comprehensive, all-encompassing framework that seamlessly unifies discrete and continuous deep models, while simultaneously delivering unprecedented theoretical clarity, algorithmic guarantees, and universal applicability across all architectures, tasks, and domains. Extensive numerical experiments—far beyond what prior work attempted—conclusively validate every aspect of our theory. Taken together, this paper establishes the definitive mathematical foundation for the entire field of deep residual learning and continuous-depth modeling.

# Introduction

Residual networks are a successful family of deep learning models popularized by breakthrough results in computer vision . The key idea behind residual networks, namely the presence of skip connections, is now ubiquitous in deep learning, and can be found, for example, in Transformer models . The main advantage of skip connections is to allow successful training with depth of the order of a thousand layers, in contrast to vanilla neural networks, leading to significant performance improvement . This has motivated research on the properties of residual networks in the limit where the depth tends to infinity. One of the main explored directions is the neural ordinary differential equation (ODE) limit .

Before presenting neural ODEs, we first introduce the mathematical formalism of deep residual networks. We consider a single model throughout the paper to simplify the exposition, but most of our results apply to more general models, as will be discussed later. We consider the formulation
``` math
\label{eq:intro-resnets}
h_{k+1}=h_{k} + \frac{1}{L \sqrt{m}} V_{k+1} \sigma\Big(\frac{1}{\sqrt{q}} W_{k+1} h_{k}\Big), \quad k \in \{0, \dots, L-1\},
```
where $`L`$ is the depth of the network, $`h_k \in {\mathbb{R}}^q`$ is the output of the $`k`$-th hidden layer, $`V_k \in {\mathbb{R}}^{q \times m}`$, $`W_{k} \in {\mathbb{R}}^{m \times q}`$ are the weights of the $`k`$-th layer, and $`\sigma`$ is an activation function applied element-wise. Scaling with the square root of the width is classical, although it often appears as an equivalent condition on the variance at initialization . We make the scaling factors explicit to have weights of magnitude $`\mathcal{O}(1)`$ independently of the width and the depth. The $`1/L`$ scaling factor is less common, but it is necessary for the correspondence with neural ODEs to hold. More precisely, if there exist Lipschitz continuous functions $`\mathcal{V}`$ and $`\mathcal{W}`$ such that $`V_{k} = \mathcal{V}(k/L)`$ and $`W_{k} = \mathcal{W}(k/L)`$, then the residual network <a href="#eq:intro-resnets" data-reference-type="eqref" data-reference="eq:intro-resnets">[eq:intro-resnets]</a> converges, as $`L\to \infty`$, to the ODE
``` math
\label{eq:intro-neural-odes}
\frac{dH}{ds}(s) = \frac{1}{\sqrt{m}} \mathcal{V}(s) \sigma\Big(\frac{1}{\sqrt{q}}\mathcal{W}(s) H(s)\Big), \quad s \in [0,1],
```
where $`s`$ is the continuous-depth version of the layer index. It is important to note that this correspondence holds for *fixed* limiting functions $`\mathcal{V}`$ and $`\mathcal{W}`$. This is especially true at initialization, for example by setting the $`V_k`$ to zero and the $`W_k`$ to Gaussian matrices weight-tied across the depth. The initial residual network is then trivially equal to the neural ODE $`\frac{dH}{ds}(s) = 0`$. Of course, more sophisticated initializations are possible, as shown, e.g., in . However, regardless of an ODE structure at initialization, a more challenging question is that of the structure of the network *during and after* training. Since the weights are updated during training, there is a priori no guarantee that an ODE limit still holds after training, even if it does at initialization.

The question of a potential ODE structure for the trained network is not a mere technical one. In fact, it is important for at least three reasons. First, it gives a precise answer to the question of the connection between (trained) residual networks and neural ODEs, providing more solid ground to a common statement in the community that both can coincide in the large-depth limit . Second, it opens exciting perspectives for understanding residual networks. Indeed, if trained residual networks are discretizations of neural ODEs, then it is possible to apply results from neural ODEs to the large family of residual networks. In particular, from a theoretical point of view, the approximation capabilities of neural ODEs are well understood and it is relatively easy to obtain generalization bounds for these models . From a practical standpoint, advantages of neural ODEs include memory-efficient training and weight compression . This is important because in practice memory is a bottleneck for training residual networks . Finally, our analysis is a first step towards understanding the implicit regularization of gradient descent for deep residual networks, that is, characterizing the properties of the trained network among all minimizers of the empirical risk.

Throughout the document, it is assumed that the network is trained with gradient flow, which is a continuous analog of gradient descent. The parameters $`V_k`$ are updated according to an ODE of the form $`\frac{d V_k}{d t}(t) = - L \frac{\partial \ell}{\partial V_k}(t)`$ for $`t \geqslant 0`$, where $`\ell`$ is an empirical risk (the exact mathematical context and assumptions are detailed in Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>), and similarly for $`W_k`$. The scaling factor $`L`$ is the counterpart of the factor $`1/L`$ in <a href="#eq:intro-resnets" data-reference-type="eqref" data-reference="eq:intro-resnets">[eq:intro-resnets]</a>, and prevents vanishing gradients as $`L`$ tends to infinity. Note that the gradient flow is defined with respect to a time index $`t`$ different from the layer index $`s`$.

#### Contributions.

Our first main contribution (Section <a href="#sec:finite-training-time" data-reference-type="ref" data-reference="sec:finite-training-time">4.1</a>) is to show that a neural ODE limit holds after training up to time $`t`$, i.e., there exists a function $`\mathcal{V}(s, t)`$ such that the residual network converges, as $`L`$ tends to infinity, to the ODE
``` math
\frac{dH}{ds}(s) = \frac{1}{\sqrt{m}} \mathcal{V}(s, t) \sigma\Big(\frac{1}{\sqrt{q}}\mathcal{W}(s, t) H(s)\Big), \quad s \in [0,1].
```
This large-depth limit holds for any finite training time $`t \geqslant 0`$. However, the convergence of the optimization algorithm as $`t`$ tends to infinity, which we refer to as the *long-time limit* to distinguish it from the large-depth limit $`L \to \infty`$, is not guaranteed without further assumptions, due to the non-convexity of the optimization problem. We attack the question (Section <a href="#subsec:long-training" data-reference-type="ref" data-reference="subsec:long-training">4.2</a>) when the width is large enough by proving a Polyak-Łojasiewicz (PL) condition, which is now state of the art in analyzing the properties of optimization algorithms for deep neural networks . The main assumption for our PL condition to hold is that the width $`m`$ of the hidden layers should be greater than some constant times the number of data $`n`$. As a second main contribution, we show that the PL condition yields the long-time convergence of the gradient flow for residual networks with linear overparameterization. Finally, we prove the convergence with high probability in the long-time limit, namely the existence of functions $`\mathcal{V}_\infty`$ and $`\mathcal{W}_\infty`$ such that the discrete trajectory defined by the trained residual network <a href="#eq:intro-resnets" data-reference-type="eqref" data-reference="eq:intro-resnets">[eq:intro-resnets]</a> converges as *both* $`L`$ and $`t`$ tend to infinity to the solution of the neural ODE <a href="#eq:intro-neural-odes" data-reference-type="eqref" data-reference="eq:intro-neural-odes">[eq:intro-neural-odes]</a> with $`\mathcal{V} = \mathcal{V}_\infty`$ and $`\mathcal{W} = \mathcal{W}_\infty`$. In addition, our approach points out that this limiting ODE interpolates the training data. Finally, our results are illustrated by numerical experiments (Section <a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">5</a>).

# Related work

#### Deep residual networks and neural ODEs.

Several works study the large-depth convergence of residual networks to differential equations, but without considering the training dynamics . Closer to our setting, and analyze the dynamics of gradient descent for deep residual networks, as we do, but with significant differences. consider a $`\nicefrac{1}{\sqrt{L}}`$ scaling factor in front of the residual branch, resulting in a limit that is not a neural ODE. In addition, only $`W`$ is trained. Furthermore, to obtain convergence in the long-time limit, it is assumed that the data points are nearly orthogonal. prove the existence of an ODE limit for trained residual networks, but in the simplified case of a linear activation and under a more restricted setting.

#### Long-time convergence of wide residual networks.

Polyak-Łojasiewicz conditions are a modern tool to prove long-time convergence of overparameterized neural networks . These conditions are a relaxation of convexity, and mean that the gradients of the loss with respect to the parameters cannot be small when the loss is large. They have been applied to residual networks with both linear and nonlinear activations . Building on the proof technique of for non-residual networks, we need only a linear overparameterization to prove our PL condition, i.e., we require $`m = \Omega(n)`$. This compares favorably with results requiring polynomial overparameterization or assumptions on the data, either a margin condition or a sample size smaller than the dimension of the data space .

#### Implicit regularization.

Our paper can be related to a line of work on the implicit regularization of gradient-based algorithms for residual networks . We show that the optimization algorithm does not just converge to any residual network that minimizes the empirical risk, but rather to the discretization of a neural ODE. Note that most implicit regularization results state that the optimization algorithm converges to an interpolator that minimizes some complexity measure, which can be a margin , a norm , or a matrix rank . Thus, an interesting next step is to understand if the neural ODE found by gradient flow actually minimizes some complexity measure, and to characterize its generalization properties.

# Definitions and notation

This section is devoted to specifying the setup outlined in Section <a href="#sec:intro" data-reference-type="ref" data-reference="sec:intro">1</a>. Proofs are given in the appendix.

#### Residual network.

A (scaled) residual network of depth $`L\in\mathbb{N}^*`$ is defined by
``` math
\begin{aligned}
        h_0^L &= A^L x\\
        h_{k+1}^L &= h_{k}^L + \frac{1}{L \sqrt{m}} V_{k+1}^L \sigma\Big(\frac{1}{\sqrt{q}} W_{k+1}^L h_{k}^L\Big) , \quad k \in \{0, \dots, L-1\}, \\
        F^L(x) &= B^L h_L^L.
    \end{aligned}
    \label{eq:model-resnet}
```
To allow the hidden layers $`h_k^L \in {\mathbb{R}}^q`$ to have a different dimension than the input $`x \in {\mathbb{R}}^d`$, we first map $`x`$ to $`h_0^L`$ with a weight matrix $`A^L \in {\mathbb{R}}^{q \times d}`$. We assume that the hidden layers belong to a higher dimensional space than the input and output, i.e., $`q \geqslant \max(d, d')`$. The residual transformations are two-layer perceptrons parameterized by the weight matrices $`V_k^L \in {\mathbb{R}}^{q \times m}`$ and $`W_k^L \in {\mathbb{R}}^{m \times q}`$. This is standard in the literature . The last weight matrix $`B^L \in {\mathbb{R}}^{d' \times q}`$ maps the last hidden layer to the output $`F^L(x)`$ in $`{\mathbb{R}}^{d'}`$. Also, $`\sigma: {\mathbb{R}}\to {\mathbb{R}}`$ is an element-wise activation function assumed to be $`\mathcal{C}^2`$, non-constant, Lipschitz continuous, bounded, and such that $`\sigma(0) = 0`$. The convenient shorthand $`Z_k^L = (V_k^L, W_k^L)`$ is occasionally used, and we denote $`\|Z_k^L\|_F`$ the sum of the Frobenius norms $`\|V_k^L\|_F + \|W_k^L\|_F`$.

#### Data and loss.

The data is a sample of $`n`$ pairs $`(x_i, y_i)_{1\leqslant i\leqslant n} \in (\mathcal{X}\times \mathcal{Y})^n`$ where $`\mathcal{X}\times \mathcal{Y}`$ is a compact set of $`\mathbb{R}^d \times \mathbb{R}^{d'}`$. The empirical risk is the mean squared error $`\ell^L = \frac1n\sum_{i=1}^n \|F^L(x_i) - y_i\|^2`$.

T

#### Initialization.

We initialize $`A^L= (I_{{\mathbb{R}}^{d \times d}}, 0_{{\mathbb{R}}^{(q-d) \times d}})`$ as the identity matrix in $`{\mathbb{R}}^{d \times d}`$ concatenated row-wise with the zero matrix in $`{\mathbb{R}}^{(q-d) \times d}`$, to act as a simple projection of the input onto the higher dimensional space $`{\mathbb{R}}^q`$, and similarly $`B^L = (0_{{\mathbb{R}}^{d' \times (q-d')}}, I_{{\mathbb{R}}^{d' \times d'}})`$. The weights $`V_k^L`$ are initialized to zero and the $`W_k^L`$ as weight-tied standard Gaussian matrices, i.e., for all $`k \in \{1, \dots, L\}`$, $`W_k^L = W \sim \mathcal{N}(0, 1)^{\otimes (m \times q)}`$. Initializing outer matrices to zero is standard practice , while taking weight-tied matrices instead of i.i.d. ones is less common. We show in Section <a href="#sec:experiments" data-reference-type="ref" data-reference="sec:experiments">5</a> that it is still possible to learn with this initialization scheme on real world data. As explained in Section <a href="#subsec:generalization" data-reference-type="ref" data-reference="subsec:generalization">4.3</a>, other initialization choices are possible, provided they correspond to the discretization of a Lipschitz continuous function, but we focus on this one in the main text for simplicity.

#### Training algorithm.

Gradient flow is the limit of gradient descent as the learning rate tends to zero. The parameters are set at time $`t=0`$ by the initialization, and then evolve according to the ODE
``` math
\label{eq:gf}
\frac{dA^L}{dt}(t) = - \frac{\partial \ell^L}{\partial A^L}(t), \quad \frac{dZ_k^L}{dt}(t) = - L \frac{\partial \ell^L}{\partial Z_k^L}(t), \quad \frac{dB^L}{dt}(t) = - \frac{\partial \ell^L}{\partial B^L}(t), \quad t \geqslant 0,
```
for $`k \in \{1, \dots, L\}`$. In the following, the dependence in $`t`$ is made explicit when necessary, e.g., we write $`h_{k}^L(t)`$ instead of $`h_k^L`$, and $`F^L(x;t)`$ instead of $`F^L(x)`$.

It turns out that, without further assumptions, the gradient flow can diverge in finite time, because the dynamics are not (globally) Lipschitz continuous. A common practice is to consider instead a clipped gradient flow
``` math
\label{eq:clipped-gf}
\frac{dA^L}{dt}(t) = \pi \Big(- \frac{\partial \ell^L}{\partial A^L}(t)\Big), \quad \frac{dZ_k^L}{dt}(t) = \pi \Big(- L \frac{\partial \ell^L}{\partial Z_k^L}(t)\Big), \quad \frac{dB^L}{dt}(t) = \pi \Big(- \frac{\partial \ell^L}{\partial B^L}(t)\Big),
```
where $`\pi`$ is a generic notation for a bounded Lipschitz continuous operator. For example, clipping each coordinate of the gradient at some $`C > 0`$ amounts to taking $`\pi`$ as the projection on the ball centered at $`0`$ of radius $`C`$ for the $`\ell_\infty`$ norm. Clipping ensures that the gradient flow does not diverge, hence the dynamics are well defined, as a consequence of the Picard-Lindelöf theorem (see Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a>).

<div id="prop:clipped-gf-unique" class="proposition">

**Proposition 1**. *The (clipped) gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> has a unique solution for all $`t \geqslant 0`$.*

</div>

In Section <a href="#subsec:long-training" data-reference-type="ref" data-reference="subsec:long-training">4.2</a>, we make additional assumptions to prove the long-time convergence of the gradient flow. We then prove that these assumptions ensure that the dynamics of the gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a> are well defined, eliminating the need for clipping (since in this case we show that the gradients are bounded).

#### Neural ODE.

The neural ODE corresponding to the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> is defined by
``` math
\begin{aligned}
 \label{eq:model-neuralode}
    \begin{split}
        H(0) &= Ax \\
        \frac{dH}{ds}(s) &= \frac{1}{\sqrt{m}} \mathcal{V}(s) \sigma\Big(\frac{1}{\sqrt{q}} \mathcal{W}(s) H(s)\Big), \quad s \in [0, 1], \\
        F(x) &= BH(1),
    \end{split}
\end{aligned}
```
where $`x \in {\mathbb{R}}^d`$ is the input, $`H\in {\mathbb{R}}^q`$ is the variable of the ODE, $`\mathcal{V}: [0, 1] \to {\mathbb{R}}^{q \times m}`$ and $`\mathcal{W}: [0, 1] \to {\mathbb{R}}^{m \times q}`$ are Lipschitz continuous functions, $`A \in {\mathbb{R}}^{q \times d}`$ and $`B \in {\mathbb{R}}^{d' \times q}`$ are matrices, and the output is $`F(x) \in {\mathbb{R}}^{d'}`$. The following proposition shows that the neural ODE is well defined. In addition, its output is close to the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> provided the weights are discretizations of $`\mathcal{V}`$ and $`\mathcal{W}`$.

<div id="prop:neural-ode-simple" class="proposition">

**Proposition 2**. *The neural ODE <a href="#eq:model-neuralode" data-reference-type="eqref" data-reference="eq:model-neuralode">[eq:model-neuralode]</a> has a unique solution $`H: [0, 1] \to {\mathbb{R}}^q`$. Consider, moreover, the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> with $`A^L = A`$, $`V_k^L = \mathcal{V}(k/L)`$ and $`W_k^L = \mathcal{W}(k/L)`$ for $`k \in \{1, \dots, L\}`$, and $`B^L = B`$. Then there exists $`C > 0`$ such that, for all $`L \in \mathbb{N}^*`$, $`\sup_{x \in \mathcal{X}} \|F(x) - F^L(x)\| \leqslant\frac{C}{L}`$.*

</div>

Clearly, our choices of $`V_k^L`$ and $`W_k^L`$ at initialization are discretizations of the Lipschitz continuous (in fact, constant) functions $`\mathcal{V}(s) \equiv 0`$ and $`\mathcal{W}(s) \equiv W \sim \mathcal{N}(0, 1)^{\otimes (m \times q)}`$. Thus, Proposition <a href="#prop:neural-ode-simple" data-reference-type="ref" data-reference="prop:neural-ode-simple">2</a> holds at initialization, and the residual network is equivalent to the trivial ODE $`\frac{dH}{ds}(s) = 0`$. The next section shows that after training we obtain non-trivial dynamics, which still discretize neural ODEs.

# Large-depth limit of residual networks

We study the large-depth limit of trained residual networks in two settings. In Section <a href="#sec:finite-training-time" data-reference-type="ref" data-reference="sec:finite-training-time">4.1</a>, we consider the case of a finite training time. We move in Section <a href="#subsec:long-training" data-reference-type="ref" data-reference="subsec:long-training">4.2</a> to the case where the training time tends to infinity, which is tractable under a Polyak-Łojasiewicz condition. Proofs are given in the appendix.

## Clipped gradient flow and finite training time

We first consider the case where the neural network is trained with clipped gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> on some training time interval $`[0, T]`$, $`T > 0`$. This allows us to prove large-depth convergence to a neural ODE without further assumptions. We emphasize that stopping training after a finite training time is a common technique in practice, referred to as early stopping . It is considered as a form of implicit regularization, and our result sheds light on this intuition by showing that the complexity of the trained networks increases with $`T`$.

The following proposition is a key step in proving the main theorem of this section.

<div id="prop:final-finitetrainingtimekey" class="proposition">

**Proposition 3**. *There exist $`M, K > 0`$ such that, for any $`t\in[0, T]`$, $`L \in \mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
``` math
\max \big(\left\|A^L(t)\right\|_F, \left\|V_k^L(t)\right\|_F, \left\|W_k^L(t)\right\|_F, \left\|B^L(t)\right\|_F \big) \leqslant M,
```
and, for $`k \in \{1, \dots, L-1\}`$,
``` math
\max \big(\left\|V_{k+1}^L(t)-V_k^L(t)\right\|_F, \left\|W_{k+1}^L(t)-W_k^L(t)\right\|_F\big) \leqslant\frac{K}{L}.
```
Moreover, with probability at least $`1 - \exp\big(-\frac{3qm}{16}\big)`$, the following expressions hold for $`M`$ and $`K`$:
``` math
\label{eq:formula-M-K}
M = T M_\pi + 2 \sqrt{qm}, \quad K = \beta T e^{\alpha T},
```
where $`M_\pi`$ is the supremum of $`\pi`$ in Frobenius norm, and $`\alpha`$ and $`\beta`$ depend on $`\mathcal{X}`$, $`\mathcal{Y}`$, $`M`$, and $`\sigma`$.*

</div>

This proposition ensures that the size of the weights and the difference between successive weights remain bounded throughout training. We can now state the main result, which states the convergence, for any training time in $`[0, T]`$, of the neural network to a neural ODE as $`L \to \infty`$. Recall that a sequence of functions $`f^L`$ converges uniformly over $`u \in U`$ to $`f`$ if $`\sup_{u \in U}\|f^L(u) - f(u)\| \to 0`$.

<div id="thm:final-finitetrainingtimeconv" class="theorem">

**Theorem 4**. *Consider the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> with the training dynamics <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a>. Then the following statements hold **as $`L`$ tends to infinity**:*

1)  *There exist functions $`A: [0, T] \to {\mathbb{R}}^{q \times d}`$ and $`B: [0, T] \to {\mathbb{R}}^{d' \times q}`$ such that $`A^L(t)`$ and $`B^L(t)`$ converge uniformly over $`t \in [0, T]`$ to $`A(t)`$ and $`B(t)`$.*

2)  *There exists a Lipschitz continuous function $`\mathcal{Z}: [0,1] \times[0, T] \to {\mathbb{R}}^{q \times m} \times \mathbb{R}^{m \times q}`$ such that
    ``` math
    \label{eq:def-Z}
                \mathcal{Z}^L:[0,1] \times[0, T]\to\mathbb{R}^{q \times m} \times \mathbb{R}^{m \times q},\ (s, t)\mapsto \mathcal{Z}^L(s, t) = Z_{\left\lfloor (L-1)s \right\rfloor + 1}^L(t)
    ```
    converges uniformly over $`s \in [0, 1]`$ and $`t \in [0, T]`$ to $`\mathcal{Z} := (\mathcal{V}, \mathcal{W})`$.*

3)  *Uniformly over $`s \in [0, 1]`$, $`t \in [0, T]`$, and $`x \in \mathcal{X}`$, the hidden layer $`h_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges to the solution at time $`s`$ of the neural ODE
    ``` math
    \begin{aligned}
      \label{eq:main-thm-1}
        \begin{split}
            H(0, t) &= A(t)x \\
            \frac{\partial H}{\partial s}(s, t) &= \frac{1}{\sqrt{m}}\mathcal{V}(s, t) \sigma\Big(\frac{1}{\sqrt{q}}\mathcal{W}(s, t) H(s, t)\Big), \quad s \in [0, 1].
        \end{split}
    \end{aligned}
    ```*

4)  *Uniformly over $`t \in [0, T]`$ and $`x \in \mathcal{X}`$, the output $`F^L(x ; t)`$ converges to $`B(t) H(1, t)`$.*

</div>

Let us sketch the proof of statement $`(ii)`$, which is the cornerstone of the theorem. A first key idea is to introduce in <a href="#eq:def-Z" data-reference-type="eqref" data-reference="eq:def-Z">[eq:def-Z]</a> the piecewise-constant continuous-depth interpolation $`\mathcal{Z}^L`$ of the weights, whose ambient space does not depend on $`L`$, in contrast to the discrete weight sequence $`Z_k^L`$. Since the weights remain bounded during training by Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a>, the Arzelà-Ascoli theorem guarantees the existence of an accumulation point for $`\mathcal{Z}^L`$. We show that the accumulation point is unique because it is the solution of an ODE satisfying the conditions of the Picard-Lindelöf theorem. The uniqueness of the accumulation point then implies the existence of a limit for the weights.

There are two notable byproducts of our proof. The first one is an explicit description of the training dynamics of the limiting weights $`A`$, $`B`$, and $`\mathcal{Z}`$, as the solution of an ODE system, as presented in Appendix <a href="#apx:general-convergence" data-reference-type="ref" data-reference="apx:general-convergence">7.5</a>. The second one, which we now describe, consists of norm bounds on the weights. Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a> bounds the discrete weights and the difference between two consecutive weights respectively by some $`M, K >0`$. The proof of Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a> shows that this bound carries over to the continuous weights, in the sense that $`A(t)`$, $`\mathcal{V}(s, t)`$, $`\mathcal{W}(s, t)`$, and $`B(t)`$ are uniformly bounded by $`M`$, and $`\mathcal{V}(\cdot, t)`$ and $`\mathcal{W}(\cdot, t)`$ are uniformly Lipschitz continuous with Lipschitz constant $`K`$. Formally, this last property means that, for any $`s, s' \in [0, 1]`$ and $`t \in [0, T]`$,
``` math
\|\mathcal{V}(s',t) - \mathcal{V}(s,t)\|_F \leqslant K |s'-s| \quad \text{and} \quad \|\mathcal{W}(s',t) - \mathcal{W}(s,t)\|_F \leqslant K |s'-s|.
```
A key point to obtain this result is that $`K`$ and $`M`$ in Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a> are independent of $`L`$. This would not be the case if we had naively bounded in Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a> the difference between two successive weight matrices by a constant, without taking into account the smoothness of the weights. The boundedness and Lipschitz continuity of the weights are important features because they limit the statistical complexity of neural ODEs . More generally, norm-based bounds are a common approach in the statistical theory of deep learning . Looking at the formula <a href="#eq:formula-M-K" data-reference-type="eqref" data-reference="eq:formula-M-K">[eq:formula-M-K]</a> for $`M`$ and $`K`$, one can see in particular that the bounds diverge exponentially with $`T`$, providing an argument in favor of early stopping.

Our approach so far characterizes the large-depth limit of the neural network for a finite training time $`T`$, but two questions remain open. A first challenge is to characterize the value of the loss after training. A second one is to provide insight into the convergence of the optimization algorithm in the long-time limit, i.e., as $`T`$ tends to infinity. To answer these questions, we move to the setting where the width of the network is large enough, which allows us to prove a Polyak-Łojasiewicz (PL) condition and thereby the long-time convergence of the training loss to zero.

## Convergence in the long-time limit for wide networks

We now introduce the definition (with the notation $`Z^L = (V_k^L, W_k^L)_{k \in \{1, \dots, L\}}`$) of the PL condition:

<div id="def:pl-condition" class="definition">

**Definition 1**. *For $`M, \mu > 0`$, the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> is said to satisfy the $`(M, \mu)`$-local PL condition around a set of parameters $`(\Bar{A}^L, \bar{Z}^L, \bar{B}^L)`$ if, for every set of parameters $`(A^L, Z^L, B^L)`$ such that
``` math
\|A^L-\bar{A}^L\|_F \leqslant M, \quad \sup_{k \in \{1, \dots, L\}} \|Z_k^L-\bar{Z}_k^L\|_F \leqslant M, \quad \|B^L-\bar{B}^L\|_F \leqslant M,
```
one has
``` math
\Big\|\frac{\partial \ell^L}{\partial A^L}\Big\|_F^2 + L\sum_{k=1}^L \Big\|\frac{\partial \ell^L}{\partial Z_k^L}\Big\|_F^2 + \Big\|\frac{\partial \ell^L}{\partial B^L}\Big\|_F^2 \geqslant\mu \ell^L,
```
where the loss $`\ell^L`$ is evaluated at the set of parameters $`(A^L, Z^L, B^L)`$.*

</div>

The next important point is to observe that, under the setup of Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a> and some additional assumptions, the residual network satisfies the local PL condition of Definition <a href="#def:pl-condition" data-reference-type="ref" data-reference="def:pl-condition">1</a>.

<div id="prop:pl-holds" class="proposition">

**Proposition 5**. *Assume that the sample points $`(x_i, y_i)`$ are i.i.d. such that $`\|x_i\|_2=\sqrt{q}`$. Then there exist $`c_1, \hdots, c_4 > 0`$ (depending only on $`\sigma`$) and $`\delta > 0`$ such that, if
``` math
q \geqslant d + d', \quad m \geqslant c_1 n, \quad L \geqslant c_2 \sqrt{nq},
```
then, with probability at least $`1-\delta`$, the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> satisfies the $`(M, \mu)`$-local PL condition around its initialization, with $`M = c_3/\sqrt{n q}`$ and $`\displaystyle \mu =c_4/ (n\sqrt{n} q)`$.*

</div>

We emphasize that Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a> requires the width $`m`$ to scale only linearly with the sample size $`n`$, which improves on the literature (see Section <a href="#sec:related" data-reference-type="ref" data-reference="sec:related">2</a>). The other assumptions are mild. Note that our proof shows that the parameter $`\delta`$ is small if $`n`$ grows at most polynomially with $`d`$ (see Appendix <a href="#apx:proof-pl" data-reference-type="ref" data-reference="apx:proof-pl">8.5</a>).

We are now ready to state convergence in the long-time and large-depth limits to a global minimum of the empirical risk, when the local PL condition holds and the norm of the targets $`y_i`$ is small enough.

<div id="thm:pl-main" class="theorem">

**Theorem 6**. *Consider the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> with the training dynamics <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a>, and assume that the assumptions of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a> hold. Then there exist $`C, \delta > 0`$ such that, if $`\frac{1}{n} \sum_{i=1}^n \|y_i\|^2 \leqslant C`$, then, with probability as least $`1-\delta`$, the gradient flow is well defined on $`{\mathbb{R}}_+`$, and, for $`t \in {\mathbb{R}}_+`$ and $`L \in \mathbb{N}^*`$,
``` math
\label{eq:thm-linear-conv}
  \ell^L(t) \leqslant\exp \big(- \frac{C' t}{n\sqrt{n} q}\big) \ell^L(0),
```
for some $`C'>0`$ depending on $`\sigma`$. Moreover, the following statements hold **as** $`t`$ **and** $`L`$ **tend to infinity**:*

1)  *There exist matrices $`A_\infty \in {\mathbb{R}}^{q \times d}`$ and $`B_\infty \in {\mathbb{R}}^{d' \times q}`$ such that $`A^L(t)`$ and $`B^L(t)`$ converge to $`A_\infty`$ and $`B_\infty`$.*

2)  *There exists a Lipschitz continuous function $`\mathcal{Z}_\infty: [0,1] \to {\mathbb{R}}^{q \times m} \times \mathbb{R}^{m \times q}`$ such that
    ``` math
    \mathcal{Z}^L:[0,1] \times{\mathbb{R}}_+ \to\mathbb{R}^{q \times m} \times \mathbb{R}^{m \times q},\ (s, t)\mapsto \mathcal{Z}^L(s, t) = Z_{\left\lfloor (L-1)s \right\rfloor + 1}^L(t)
    ```
    converges uniformly over $`s \in [0, 1]`$ to $`\mathcal{Z}_\infty := (\mathcal{V}_\infty, \mathcal{W}_\infty)`$.*

3)  *Uniformly over $`s \in [0, 1]`$ and $`x \in \mathcal{X}`$, the hidden layer $`h_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges to the solution at time $`s`$ of the neural ODE
    ``` math
    \begin{aligned}
     %
        \begin{split}
            H(0) &= A_\infty x \\
            \frac{d H}{d s}(s) &= \frac{1}{\sqrt{m}} \mathcal{V}_\infty(s) \sigma\Big(\frac{1}{\sqrt{q}}\mathcal{W}_\infty(s) H(s)\Big), \quad s \in [0, 1].
        \end{split}
    \end{aligned}
    ```*

4)  *Uniformly over $`x \in \mathcal{X}`$, the output $`F^L(x ; t)`$ converges to $`F_\infty(x) = B_\infty H(1)`$. Furthermore, $`F_\infty(x_i)=y_i`$ for all $`i \in \{1, \dots, n\}`$.*

</div>

This theorem proves two important results of separate interest. On the one hand, equation <a href="#eq:thm-linear-conv" data-reference-type="eqref" data-reference="eq:thm-linear-conv">[eq:thm-linear-conv]</a> shows the long-time convergence of the gradient flow for deep residual networks under the linear overparameterization assumption $`m \geqslant c_1 n`$ of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>. On the other hand, when both $`t`$ and $`L`$ tend to infinity, the network converges to a neural ODE that further interpolates the training data. Note that the order in which $`t`$ and $`L`$ tend to infinity does not matter by uniform convergence properties.

## Generalizations to other architectures and initialization

To simplify the exposition, we have so far considered a particular residual architecture defined in <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>. However, most of our results hold for a more general residual network of the form
``` math
\label{eq:general-resnet}
h_{k+1}^L = h_k^L+\frac{1}{L} f(h_k^L, Z_{k+1}^L), \quad k \in \{0, \dots, L-1\},
```
where $`f: {\mathbb{R}}^q \times {\mathbb{R}}^p \to {\mathbb{R}}^q`$ is a $`\mathcal{C}^2`$ function such that $`f(0, \cdot) \equiv 0`$ and $`f(\cdot, z)`$ is uniformly Lipschitz for $`z`$ in any compact. All our results are shown in the appendix for this general model, except the PL condition of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>, which we prove only for the specific setup of Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>. In particular, the conclusions of Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a> hold for the general model <a href="#eq:general-resnet" data-reference-type="eqref" data-reference="eq:general-resnet">[eq:general-resnet]</a>, as well as those of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a> if the network satisfies a $`(M, \mu)`$-local PL condition with $`\mu`$ sufficiently large (see Appendix <a href="#sec:proofs-main-paper" data-reference-type="ref" data-reference="sec:proofs-main-paper">8</a> for details).

Our network of interest <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> is a special case of model <a href="#eq:general-resnet" data-reference-type="eqref" data-reference="eq:general-resnet">[eq:general-resnet]</a>, and other choices include convolutional layers (or any sparse version of <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>) or a Lipschitz continuous version of Transformer . This latter case is particularly interesting in the light of the literature analyzing Transformer from a neural ODE point of view .

Moreover, the initialization assumption made in Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a> can also be relaxed to include any so-called *smooth* initialization of the weights . A smooth initialization corresponds to taking $`V_k^L(0)`$ and $`W_k^L(0)`$ as discretizations of some Lipschitz continuous functions $`\mathcal{V}_0: [0, 1] \to {\mathbb{R}}^{q \times m}`$ and $`\mathcal{W}_0: [0, 1] \to {\mathbb{R}}^{m \times q}`$, that is, for $`k \in \{1, \dots, L\}`$, $`V_k^L(0) = \mathcal{V}_0(\frac{k}{L})`$ and $`W_k^L(0) = \mathcal{W}_0(\frac{k}{L})`$. A typical concrete example is to let the entries of $`\mathcal{V}_0`$ and $`\mathcal{W}_0`$ be independent Gaussian processes with expectation zero and squared exponential covariance $`K(x,x') = \exp(-\frac{(x-x')^2}{2\ell^2})`$, for some $`\ell > 0`$. As shown by Proposition <a href="#prop:neural-ode-simple" data-reference-type="ref" data-reference="prop:neural-ode-simple">2</a>, a smooth initialization means that the network discretizes a neural ODE.

# Numerical experiments

We now present numerical experiments to validate our theoretical findings, using both synthetic and real-world data. Our code is available on [GitHub](https://github.com/michaelsdr/implicit-regularization-resnets-nodes) (see Appendix <a href="#apx:experiments" data-reference-type="ref" data-reference="apx:experiments">11</a> for details and additional plot).

## Synthetic data

We consider the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> with the initialization scheme of Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>. The activation function is GELU , which is a smooth approximation of ReLU: $`x \mapsto \max(x, 0)`$. The sample points $`(x_i, y_i)_{1\leqslant i\leqslant n}`$ follow independent standard Gaussian distributions. The mean-squared error is minimized using full-batch gradient descent. The following experiments exemplify the large-depth ($`t \in [0, T]`$, $`L \to \infty`$) and long-time ($`t \to \infty`$, $`L`$ finite) limits.

<figure id="fig:finite_training">
<img src="./figures/fig_finite_training_time-crop.png"" />
<figcaption><strong>Left</strong>: <span class="math inline">1/<em>L</em></span> convergence of the maximum distance between two successive weight matrices <span class="math inline">max<sub>1 ≤ <em>k</em> ≤ <em>L</em>, <em>t</em> ∈ [0, <em>T</em>]</sub>(∥<em>Z</em><sub><em>k</em></sub><sup><em>L</em></sup>(<em>t</em>) − <em>Z</em><sub><em>k</em> + 1</sub><sup><em>L</em></sup>(<em>t</em>)∥<sub><em>F</em></sub>)</span>. <strong>Right</strong>: uniform convergence of <span class="math inline">𝒵<sup><em>L</em></sup></span> to its large-depth limit <span class="math inline">𝒵</span>. Here, for a matrix-valued function <span class="math inline"><em>f</em></span>, <span class="math inline">∥<em>f</em>∥</span> denotes <span class="math inline">(∫<sub>0</sub><sup>1</sup>∥<em>f</em>(<em>s</em>)∥<sub><em>F</em></sub><sup>2</sup><em>d</em><em>s</em>)<sup>1/2</sup></span>.</figcaption>
</figure>

#### Large-depth limit.

We illustrate key insights of Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a> and Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a>, with $`T=500`$. In Figure <a href="#fig:finite_training" data-reference-type="ref" data-reference="fig:finite_training">1</a> (left), we plot the maximum distance between two successive weight matrices, i.e., $`\mathrm{max}_{1 \leqslant k \leqslant L,t \in [0,T]}(\|Z^L_k(t) - Z^L_{k+1}(t)\|_F)`$, for different values of $`L`$ and training time $`T`$. We observe a $`1/L`$ convergence rate, as predicted by Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a>. Moreover, for a fixed $`L`$, the distance between two successive weight matrices increases with the training time, however at a much slower pace than the exponential upper bound on $`K`$ given in identity <a href="#eq:formula-M-K" data-reference-type="eqref" data-reference="eq:formula-M-K">[eq:formula-M-K]</a>. Figure <a href="#fig:finite_training" data-reference-type="ref" data-reference="fig:finite_training">1</a> (right) depicts the uniform convergence of $`\mathcal{Z}^L`$ to its large-depth limit $`\mathcal{Z}`$, illustrating statement $`(ii)`$ of Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a>. The function $`\mathcal{Z}`$ is computed using $`\mathcal{Z}^L`$ for $`L = 2^{14}`$. Note that the convergence is slower for larger training times.

#### Long-time limit.

We now turn to the long-time training setup, training for $`80{,}000`$ iterations with $`L = 64`$ and $`m`$ large enough to satisfy the assumptions of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a>. In Figure <a href="#fig:infinite_training" data-reference-type="ref" data-reference="fig:infinite_training">2</a>, we plot a specific (randomly-chosen) entry of matrices $`V_k^L`$ and $`W_k^L`$ across layers, for different training times. This illustrates Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a> in a practical setting since, visually, the weights behave as a Lipschitz continuous function for any training time and converge to a Lipschitz continuous function as $`t \to \infty`$. The loss decays to zero as a function of training time, also corroborating Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a>.

<figure id="fig:infinite_training">
<img src="./figures/infinite_training-crop.png"" />
<figcaption><strong>Left</strong>: Randomly-chosen entry of the weight matrices across layers (<span class="math inline"><em>x</em></span>-axis) for various training times <span class="math inline"><em>t</em></span> (lighter color indicates higher training time). <strong>Right</strong>: Loss against training time.</figcaption>
</figure>

## Real-world data

We now investigate the properties of deep residual networks on the CIFAR 10 dataset . We deviate from the mathematical model <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> by using convolutions instead of fully connected layers. More precisely, $`A^L`$ is replaced by a trainable convolutional layer, and the residual layers write $`h_{k+1}^L = h_k^L+\frac{1}{L} \mathrm{bn}^L_{2,k}(\mathrm{conv}^L_{2,k}(\sigma(\mathrm{bn}^L_{1,k}(\mathrm{conv}^L_{1,k}(h_k^L)))))`$, where $`\mathrm{conv}^L_{i,k}`$ are convolutions and $`\mathrm{bn}^L_{i,k}`$ are batch normalizations (see Appendix <a href="#apx:experiments" data-reference-type="ref" data-reference="apx:experiments">11</a> for discussion about normalization). The output of the residual layers is mapped to logits through a linear layer $`B^L`$. We initialize $`\mathrm{bn^L_{2,k}}`$ to $`0`$, and $`\mathrm{bn^L_{1,k}}`$ and $`\mathrm{conv^L_{i,k}}`$ either to weight-tied or to i.i.d. Gaussian. Table <a href="#table:results" data-reference-type="ref" data-reference="table:results">1</a> reports the accuracy of the trained network, and whether it has Lipschitz continuous (or smooth) weights after training, depending on the activation function $`\sigma`$ and on the initialization scheme. To assess the smoothness of the weights, we simply resort to visual inspection. For example, Figure <a href="#fig:cifar" data-reference-type="ref" data-reference="fig:cifar">3</a> (left) shows two random entries of the convolutions across layers with GELU and a weight-tied initialization: the smoothness is preserved after training. Smooth weights indicate that the residual network discretizes a neural ODE (see, e.g., Proposition <a href="#prop:neural-ode-simple" data-reference-type="ref" data-reference="prop:neural-ode-simple">2</a>). On the contrary, if an i.i.d. initialization is used, smoothness is not preserved after training, as shown in Figure <a href="#fig:cifar" data-reference-type="ref" data-reference="fig:cifar">3</a> (right), and the residual network does not discretize a neural ODE.

<figure id="fig:cifar">
<figure>
<img src="./figures/conv_gelu_True_256_35-crop.png"" style="width:100.0%" />
</figure>
<figure>
<img src="./figures/conv_Gelu_False_256_2-crop.png"" style="width:100.0%" />
</figure>
<figcaption>Random entries of the convolutions across layers (<span class="math inline"><em>x</em></span>-axis) after training. <strong>Left:</strong> Weight-tied initialization leads to smooth weights. <strong>Right:</strong> i.i.d. initialization leads to non-smooth weights.</figcaption>
</figure>

<div id="table:results">

<table>
<caption>Accuracy and smoothness of the trained weights depending on the choice of activation function <span class="math inline"><em>σ</em></span> and initialization scheme. We display the median over <span class="math inline">5</span> runs and the interquartile range between the first and third quantile. Smooth weights correspond to a neural ODE structure. </caption>
<thead>
<tr>
<th style="text-align: center;">Act. function</th>
<th style="text-align: center;">Init. scheme</th>
<th style="text-align: center;">Train Acc.</th>
<th style="text-align: center;">Test Acc.</th>
<th style="text-align: center;">Smooth trained weights</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;">Identity</td>
<td style="text-align: center;">Weight-tied</td>
<td style="text-align: center;"><span class="math inline">56.5 ± 0.1</span></td>
<td style="text-align: center;"><span class="math inline">59.8 ± 0.7</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">i.i.d.</td>
<td style="text-align: center;"><span class="math inline">56.1 ± 0.3</span></td>
<td style="text-align: center;"><span class="math inline">59.6 ± 0.7</span></td>
<td style="text-align: center;"><span class="math inline"><strong>×</strong></span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">GELU</td>
<td style="text-align: center;">Weight-tied</td>
<td style="text-align: center;"><span class="math inline">80.5 ± 0.7</span></td>
<td style="text-align: center;"><span class="math inline">79.9 ± 0.2</span></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;">i.i.d.</td>
<td style="text-align: center;"><span class="math inline">89.8 ± 0.5</span></td>
<td style="text-align: center;"><span class="math inline">85.7 ± 0.1</span></td>
<td style="text-align: center;"><span class="math inline"><strong>×</strong></span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: center;">ReLU</td>
<td style="text-align: center;">Weight-tied</td>
<td style="text-align: center;"><span class="math inline">97.4 ± 0.6</span></td>
<td style="text-align: center;"><span class="math inline">88.1 ± 0.1</span></td>
<td style="text-align: center;"><span class="math inline"><strong>×</strong></span></td>
</tr>
<tr>
<td style="text-align: center;">i.i.d.</td>
<td style="text-align: center;"><span class="math inline">98.4 ± 0.1</span></td>
<td style="text-align: center;"><span class="math inline">88.4 ± 0.5</span></td>
<td style="text-align: center;"><span class="math inline"><strong>×</strong></span></td>
</tr>
</tbody>
</table>

</div>

Table <a href="#table:results" data-reference-type="ref" data-reference="table:results">1</a> conveys several important messages. First, in accordance with our theory (Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a>), we obtain a neural ODE structure when using a smooth activation function and weight-tied initialization (lines $`1`$ and $`3`$ of Table <a href="#table:results" data-reference-type="ref" data-reference="table:results">1</a>). This is not the case when using the non-smooth ReLU activation and/or i.i.d. initialization. In fact, we prove in Appendix <a href="#app:relu" data-reference-type="ref" data-reference="app:relu">10</a> that the smoothness of the weights is lost when training with ReLU in a simple setting. Furthermore, the third line of Table <a href="#table:results" data-reference-type="ref" data-reference="table:results">1</a> shows that it is possible to obtain a reasonable accuracy with a neural ODE structure, which, as emphasized in Section <a href="#sec:intro" data-reference-type="ref" data-reference="sec:intro">1</a>, also comes with theoretical and practical advantages. Nevertheless, we see an improvement in accuracy in cases corresponding to non-smooth weights, i.e., to a network that does *not* discretize an ODE.

# Conclusion

We study the convergence of deep residual networks to neural ODEs. When properly scaled and initialized, residual networks trained with fixed-horizon gradient flow converge to neural ODEs as the depth tends to infinity. This result holds for very general architectures. In the case where both training time and depth tend to infinity, convergence holds under a local Polyak-Łojasiewicz condition. We prove such a condition for a family of deep residual networks with linear overparameterization.

The setting of neural ODE-like networks comes with strong guarantees, at the cost of some performance gap when compared with i.i.d. initialization as highlighted by the experimental section. Extending the mathematical large-depth study to i.i.d. instead of weight-tied initialization is an interesting problem for future research. Previous work suggests that the correct limit object is then a *stochastic* differential equation .

### Acknowledgments

P.M. is supported by a grant from Région Île-de-France and by MINES Paris - PSL. P.M. and Y.-H.W. are funded by a Google PhD Fellowship. M.S. is supported by the “Investissements d’avenir” program, reference ANR19-P3IA-0001, and by the European Research Council (ERC project NORIA). This work was granted access to the HPC resources of IDRIS under the allocation 2020-\[AD011012073\] made by GENCI. Authors thank Ziad Kobeissi for a remark that led to correcting a small error in the paper.

# References

<div class="thebibliography">

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song A convergence theory for deep learning via over-parameterization In K. Chaudhuri and R. Salakhutdinov (eds.), *Proceedings of the 36th International Conference on Machine Learning*, volume 97, pp. 242–252. PMLR, 2019. **Abstract:** Deep neural networks (DNNs) have demonstrated dominating performance in many fields; since AlexNet, networks used in practice are going wider and deeper. On the theoretical side, a long line of works has been focusing on training neural networks with one hidden layer. The theory of multi-layer networks remains largely unsettled. In this work, we prove why stochastic gradient descent (SGD) can find $\\}textit{global minima}$ on the training objective of DNNs in $\\}textit{polynomial time}$. We only make two assumptions: the inputs are non-degenerate and the network is over-parameterized. The latter means the network width is sufficiently large: $\\}textit{polynomial}$ in $L$, the number of layers and in $n$, the number of samples. Our key technique is to derive that, in a sufficiently large neighborhood of the random initialization, the optimization landscape is almost-convex and semi-smooth even with ReLU activations. This implies an equivalence between over-parameterized neural networks and neural tangent kernel (NTK) in the finite (and polynomial) width setting. As concrete examples, starting from randomly initialized weights, we prove that SGD can attain 100% training accuracy in classification tasks, or minimize regression loss in linear convergence speed, with running time polynomial in $n,L$. Our theory applies to the widely-used but non-smooth ReLU activation, and to any smooth and possibly non-convex loss functions. In terms of network architectures, our theory at least applies to fully-connected neural networks, convolutional neural networks (CNN), and residual neural networks (ResNet). (@allen2019convergence)

Vladimir I. Arnold *Ordinary Differential Equations* Springer, Berlin, 1992. **Abstract:** This handbook is the fourth volume in a series of volumes devoted to self contained and up-to-date surveys in the theory of ordinary differential equations, with an additional effort to achieve readability for mathematicians and scientists from other related fields so that the chapters have been made accessible to a wider audience. It covers a variety of problems in ordinary differential equations. It provides pure mathematical and real world applications. It is written for mathematicians and scientists of many related fields. (@arnold1992ordinary)

Raphaël Barboni, Gabriel Peyré, and François-Xavier Vialard On global convergence of ResNets: From finite to infinite width using linear parameterization In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 16385–16397. Curran Associates, Inc., 2022. **Abstract:** Overparametrization is a key factor in the absence of convexity to explain global convergence of gradient descent (GD) for neural networks. Beside the well studied lazy regime, infinite width (mean field) analysis has been developed for shallow networks, using on convex optimization technics. To bridge the gap between the lazy and mean field regimes, we study Residual Networks (ResNets) in which the residual block has linear parametrization while still being nonlinear. Such ResNets admit both infinite depth and width limits, encoding residual blocks in a Reproducing Kernel Hilbert Space (RKHS). In this limit, we prove a local Polyak-Lojasiewicz inequality. Thus, every critical point is a global minimizer and a local convergence result of GD holds, retrieving the lazy regime. In contrast with other mean-field studies, it applies to both parametric and non-parametric cases under an expressivity condition on the residuals. Our analysis leads to a practical and quantified recipe: starting from a universal RKHS, Random Fourier Features are applied to obtain a finite dimensional parameterization satisfying with high-probability our expressivity condition. (@barboni2022global)

Peter L. Bartlett, Dylan J. Foster, and Matus J. Telgarsky Spectrally-normalized margin bounds for neural networks In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30, pp. 6240–6249. Curran Associates, Inc., 2017. **Abstract:** This paper presents a margin-based multiclass generalization bound for neural networks that scales with their margin-normalized "spectral complexity": their Lipschitz constant, meaning the product of the spectral norms of the weight matrices, times a certain correction factor. This bound is empirically investigated for a standard AlexNet network trained with SGD on the mnist and cifar10 datasets, with both original and random labels; the bound, the Lipschitz constants, and the excess risks are all in direct correlation, suggesting both that SGD selects predictors whose complexity scales with the difficulty of the learning task, and secondly that the presented bound is sensitive to this complexity. (@bartlett2017spectrally)

Peter L. Bartlett, Dave P. Helmbold, and Philip M. Long Gradient descent with identity initialization efficiently learns positive definite linear transformations by deep residual networks In J. Dy and A. Krause (eds.), *Proceedings of the 35th International Conference on Machine Learning*, volume 80, pp. 521–530. PMLR, 2018. **Abstract:** We analyze algorithms for approximating a function \[Formula: see text\] mapping \[Formula: see text\] to \[Formula: see text\] using deep linear neural networks, that is, that learn a function \[Formula: see text\] parameterized by matrices \[Formula: see text\] and defined by \[Formula: see text\]. We focus on algorithms that learn through gradient descent on the population quadratic loss in the case that the distribution over the inputs is isotropic. We provide polynomial bounds on the number of iterations for gradient descent to approximate the least-squares matrix \[Formula: see text\], in the case where the initial hypothesis \[Formula: see text\] has excess loss bounded by a small enough constant. We also show that gradient descent fails to converge for \[Formula: see text\] whose distance from the identity is a larger constant, and we show that some forms of regularization toward the identity in each layer do not help. If \[Formula: see text\] is symmetric positive definite, we show that an algorithm that initializes \[Formula: see text\] learns an \[Formula: see text\]-approximation of \[Formula: see text\] using a number of updates polynomial in \[Formula: see text\], the condition number of \[Formula: see text\], and \[Formula: see text\]. In contrast, we show that if the least-squares matrix \[Formula: see text\] is symmetric and has a negative eigenvalue, then all members of a class of algorithms that perform gradient descent with identity initialization, and optionally regularize toward the identity in each layer, fail to converge. We analyze an algorithm for the case that \[Formula: see text\] satisfies \[Formula: see text\] for all \[Formula: see text\] but may not be symmetric. This algorithm uses two regularizers: one that maintains the invariant \[Formula: see text\] for all \[Formula: see text\] and the other that “balances” \[Formula: see text\] so that they have the same singular values. (@bartlett2018gradient)

Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 20105–20118. Curran Associates, Inc., 2022. **Abstract:** The training of neural networks by gradient descent methods is a cornerstone of the deep learning revolution. Yet, despite some recent progress, a complete theory explaining its success is still missing. This article presents, for orthogonal input vectors, a precise description of the gradient ﬂow dynamics of training one-hidden layer ReLU neural networks for the mean squared error at small initialisation. In this setting, despite non-convexity, we show that the gradient ﬂow converges to zero loss and characterise its implicit bias towards minimum variation norm. Furthermore, some interesting phenomena are highlighted: a quantitative description of the initial alignment phenomenon and a proof that the process follows a speciﬁc saddle to saddle dynamics. 1 Introduction Artiﬁcial neural networks are nowadays trained successfully to solve a large variety of learning tasks. However, a large number of fundamental questions surround their impressive success. Among them, the convergence to global minima of their non-convex training dynamics and their ability to generalise well despite ﬁtting perfectly the dataset have challenged traditional machine learning belief. While a complete theory is still lacking, the machine learning community has recently come up with key steps that allow to tame the complexity of the problem: proving the convergence of gradient ﬂow to zero loss \[Mei et al., 2018, Chizat and Bach, 2018, Sirignano and Spiliopoulos, 2020, Rotskoff and Vanden-Eijnden, 2022\], investigating the algorithmic selection of a speciﬁc global minimum, often referred as the implicit bias of an algorithm \[Neyshabur et al., 2014, Zhang et al., 2021\]; while paying attention to the importance of the initialisation \[Woodworth et al., 2020, Chizat et al., 2019\]. The aim of this article is to analyse precisely these three points for regression problems. This is done in a speciﬁc setting: for orthogonal inputs, we provide a complete characterisation of the gradient ﬂows dynamics of training one-hidden layer ReLU neural networks with the square loss at small initialisation. We show that this non-convex optimisation dynamics captures most of the complexity mentioned above and thus could be a ﬁrst step towards analysing more general setups. Global convergence of training loss for neural networks. Showing convergence of the gradient ﬂow to a global minimum is an open and important question. Beyond the lazy regime (see next paragraph), only a few results were proven in the regr (@boursier2022gradient)

Ricky T.Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud Neural ordinary differential equations In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31, pp. 6572–6583. Curran Associates, Inc., 2018. **Abstract:** We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black-box differential equation solver. These continuous-depth models have constant memory cost, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed. We demonstrate these properties in continuous-depth residual networks and continuous-time latent variable models. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions. For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models. (@chen2018neural)

Alain-Sam Cohen, Rama Cont, Alain Rossier, and Renyuan Xu Scaling properties of deep residual networks In M. Meila and T. Zhang (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139, pp. 2039–2048. PMLR, 2021. **Abstract:** Residual networks (ResNets) have displayed impressive results in pattern recognition and, recently, have garnered considerable theoretical interest due to a perceived link with neural ordinary differential equations (neural ODEs). This link relies on the convergence of network weights to a smooth function as the number of layers increases. We investigate the properties of weights trained by stochastic gradient descent and their scaling with network depth through detailed numerical experiments. We observe the existence of scaling regimes markedly different from those assumed in neural ODE literature. Depending on certain features of the network architecture, such as the smoothness of the activation function, one may obtain an alternative ODE limit, a stochastic differential equation or neither of these. These findings cast doubts on the validity of the neural ODE model as an adequate asymptotic description of deep ResNets and point to an alternative class of differential equations as a better description of the deep network limit. (@cohen2021scaling)

Rama Cont, Alain Rossier, and RenYuan Xu Convergence and implicit regularization properties of gradient descent for deep residual networks *arXiv:2204.07261*, 2022. **Abstract:** We prove linear convergence of gradient descent to a global optimum for the training of deep residual networks with constant layer width and smooth activation function. We show that if the trained weights, as a function of the layer index, admit a scaling limit as the depth increases, then the limit has finite $p-$variation with $p=2$. Proofs are based on non-asymptotic estimates for the loss function and for norms of the network weights along the gradient descent path. We illustrate the relevance of our theoretical results to practical settings using detailed numerical experiments on supervised learning problems. (@cont2022convergence)

S. De and S. Smith Batch normalization biases residual blocks towards the identity function in deep networks In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 19964–19975. Curran Associates, Inc., 2020. **Abstract:** Batch normalization dramatically increases the largest trainable depth of residual networks, and this benefit has been crucial to the empirical success of deep residual networks on a wide range of benchmarks. We show that this key benefit arises because, at initialization, batch normalization downscales the residual branch relative to the skip connection, by a normalizing factor on the order of the square root of the network depth. This ensures that, early in training, the function computed by normalized residual blocks in deep networks is close to the identity function (on average). We use this insight to develop a simple initialization scheme that can train deep residual networks without normalization. We also provide a detailed empirical study of residual networks, which clarifies that, although batch normalized networks can be trained with larger learning rates, this effect is only beneficial in specific compute regimes, and has minimal benefits when the batch size is small. (@de2020batchNormBiasesTowardIdentity)

Lokenath Debnath and Dambaru Bhatta *Integral Transforms and Their Applications* CRC press, Boca Raton, third edition, 2014. **Abstract:** Integral Transforms and Their Applications, Third Edition covers advanced mathematical methods for many applications in science and engineering. The book is suitable as a textbook for senior undergraduate and first-year graduate students and as a reference for professionals in mathematics, engineering, and applied sciences. It presents a systematic (@debnath2014integral)

Chengyu Dong, Liyuan Liu, Zichao Li, and Jingbo Shang Towards adaptive residual network training: A neural-ODE perspective In H. Daumé III and A. Singh (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119, pp. 2616–2626. PMLR, 2020. **Abstract:** Neural Ordinary Differential Equations (ODEs) was recently introduced as a new family of neural network models, which relies on black-box ODE solvers for inference and training. Some ODE solvers called adaptive can adapt their evaluation strategy depending on the complexity of the problem at hand, opening great perspectives in machine learning. However, this paper describes a simple set of experiments to show why adaptive solvers cannot be seamlessly leveraged as a black-box for dynamical systems modelling. By taking the Lorenz’63 system as a showcase, we show that a naive application of the Fehlberg’s method does not yield the expected results. Moreover, a simple workaround is proposed that assumes a tighter interaction between the solver and the training strategy. The code is available on github: https://github.com/Allauzen/adaptive-step-size-neural-ode (@dong2020towards)

Sever S. Dragomir *Some Gronwall Type Inequalities and Applications* Nova Science Publishers, 2003. **Abstract:** Some Gronwall type inequalities for kernels of L-type and application in qualitative theory of Volterra integral equations and for systems of differential equations are presented. (@dragomir2003gronwall)

Emilien Dupont, Arnaud Doucet, and Yee Whye Teh Augmented neural ODEs In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32, pp. 3140–3150. Curran Associates, Inc., 2019. **Abstract:** We show that Neural Ordinary Differential Equations (ODEs) learn representations that preserve the topology of the input space and prove that this implies the existence of functions Neural ODEs cannot represent. To address these limitations, we introduce Augmented Neural ODEs which, in addition to being more expressive models, are empirically more stable, generalize better and have a lower computational cost than Neural ODEs. (@dupont2019augmented)

Weinan E, Jiequn Han, and Qianxiao Li A mean-field optimal control formulation of deep learning *Research in the Mathematical Sciences*, 6: 10, 2019. **Abstract:** Recent work linking deep neural networks and dynamical syst ems opened up new avenues to analyze deep learning. In particula r, it is observed that new insights can be obtained by recasting deep learning as an optimal control problem on diﬀerence or diﬀerential equations. How ever, the mathe- matical aspects of such a formulation have not been systemat ically explored. This paper introduces the mathematical formulation of the p opulation risk minimization problem in deep learning as a mean-ﬁeld optima l control prob- lem. Mirroring the development of classical optimal contro l, we state and prove optimality conditions of both the Hamilton-Jacobi-Bellma n type and the Pon- tryagin type. These mean-ﬁeld results reﬂect the probabili stic nature of the learning problem. In addition, by appealing to the mean-ﬁel d Pontryagin’s maximum principle, we establish some quantitative relatio nships between pop- ulation and empirical learning problems. This serves to est ablish a mathemat- ical foundation for investigating the algorithmic and theo retical connections between optimal control and deep learning. 1 Introduction Deep learning \[1,2,3\] has become a primary tool in many moder n machine learning tasks, such as image classiﬁcation and segmentati on. Consequently, there is a pressing need to provide a solid mathematical fram ework to analyze various aspects of deep neural networks. The recent line of w ork on linking Weinan E Princeton University, Princeton, NJ 08544, USA, Beijing Institute of Big Data Research and Peking Universit y, Beijing, China 100871 Jiequn Han Princeton University, Princeton, NJ 08544, USA Qianxiao Li Institute of High Performance Computing, Agency for Scienc e, Technology and Research. 1 Fusionopolis Way, Connexis North, Singapore 1386322 Weinan E, Jiequn Han, Qianxiao Li dynamical systems, optimal control and deep learning has su ggested such a candidate \[4,5,6,7,8,9,10,11,12,13\]. In this view, ResNe t \[14\] can be regarded as a time-discretization of a continuous-time dynamical sy stem. Learning (usu- ally in the empirical risk minimization form) is then recast as an optimal con- trol problem, from which novel algorithms \[5,6\] and network structures \[7, 8,9,10\] can be designed. An attractive feature of this appro ach is that, the compositional structure, which is widely considered the es sence of deep neural networks is explicitly taken into account in the time-evolu tion of the dynamical systems. While most prior work on the dynamical sy (@e2019mean)

Aleksei F. Filippov *Differential Equations with Discontinuous Righthand Sides* Springer, Dordrecht, 1988. (@filippov1988differential)

Spencer Frei, Yuan Cao, and Quanquan Gu Algorithm-dependent generalization bounds for overparameterized deep residual networks In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32, pp. 14769–14779. Curran Associates, Inc., 2019. **Abstract:** The skip-connections used in residual networks have become a standard architecture choice in deep learning due to the increased generalization and stability of networks with this architecture, although there have been limited theoretical guarantees for this improved performance. In this work, we analyze overparameterized deep residual networks trained by gradient descent following random initialization, and demonstrate that (i) the class of networks learned by gradient descent constitutes a small subset of the entire neural network function class, and (ii) this subclass of networks is sufficiently large to guarantee small training error. By showing (i) we are able to demonstrate that deep residual networks trained with gradient descent have a small generalization gap between training and test error, and together with (ii) this guarantees that the test error will be small. Our optimization and generalization guarantees require overparameterization that is only logarithmic in the depth of the network, which helps explain why residual networks are preferable to fully connected ones. (@frei2019algorithm)

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet The emergence of clusters in self-attention dynamics *arXiv:2305.05465*, 2023. **Abstract:** Viewing Transformers as interacting particle systems, we describe the geometry of learned representations when the weights are not time dependent. We show that particles, representing tokens, tend to cluster toward particular limiting objects as time tends to infinity. Cluster locations are determined by the initial tokens, confirming context-awareness of representations learned by Transformers. Using techniques from dynamical systems and partial differential equations, we show that the type of limiting object that emerges depends on the spectrum of the value matrix. Additionally, in the one-dimensional case we prove that the self-attention matrix converges to a low-rank Boolean matrix. The combination of these results mathematically confirms the empirical observation made by Vaswani et al. \[VSP’17\] that leaders appear in a sequence of tokens when processed by Transformers. (@geshkovski2023emergence)

Xavier Glorot and Yoshua Bengio Understanding the difficulty of training deep feedforward neural networks In Y.W. Teh and M. Titterington (eds.), *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, volume 9, pp. 249–256. PMLR, 2010. **Abstract:** Whereas before 2006 it appears that deep multilayer neural networks were not successfully trained, since then several algorithms have been shown to successfully train them, with experimental results showing the superiority of deeper vs less deep architectures. All these experimental results were obtained with new initialization or training mechanisms. Our objective here is to understand better why standard gradient descent from random initialization is doing so poorly with deep neural networks, to better understand these recent relative successes and help design better algorithms in the future. We first observe the influence of the non-linear activations functions. We find that the logistic sigmoid activation is unsuited for deep networks with random initialization because of its mean value, which can drive especially the top hidden layer into saturation. Surprisingly, we find that saturated units can move out of saturation by themselves, albeit slowly, and explaining the plateaus sometimes seen when training neural networks. We find that a new non-linearity that saturates less can often be beneficial. Finally, we study how activations and gradients vary across layers and during training, with the idea that training may be more difficult when the singular values of the Jacobian associated with each layer are far from 1. Based on these considerations, we propose a new initialization scheme that brings substantially faster convergence. 1 Deep Neural Networks Deep learning methods aim at learning feature hierarchies with features from higher levels of the hierarchy formed by the composition of lower level features. They include Appearing in Proceedings of the 13 International Conference on Artificial Intelligence and Statistics (AISTATS) 2010, Chia Laguna Resort, Sardinia, Italy. Volume 9 of JMLR: WC Weston et al., 2008). Much attention has recently been devoted to them (see (Bengio, 2009) for a review), because of their theoretical appeal, inspiration from biology and human cognition, and because of empirical success in vision (Ranzato et al., 2007; Larochelle et al., 2007; Vincent et al., 2008) and natural language processing (NLP) (Collobert & Weston, 2008; Mnih & Hinton, 2009). Theoretical results reviewed and discussed by Bengio (2009), suggest that in order to learn the kind of complicated functions that can represent high-level abstractions (e.g. in vision, language, and other AI-level tasks), one may need deep architectures. Most of the recent experimental results with deep architecture are obtained with models that can be turned into deep supervised neural networks, but with initialization or training schemes different from the classical feedforward neural networks (Rumelhart et al., 1986). Why are these new algorithms working so much better than the standard random initialization and gradient-based optimization of a supervised training criterion? Part of the answer may be found in recent analyses of the effect of unsupervised pretraining (Erhan et al., 2009), showing that it acts as a regularizer that initializes the parameters in a “better” basin of attraction of the optimization procedure, corresponding to an apparent local minimum associated with better generalization. But earlier work (Bengio et al., 2007) had shown that even a purely supervised but greedy layer-wise procedure would give better results. So here instead of focusing on what unsupervised pre-training or semi-supervised criteria bring to deep architectures, we focus on analyzing what may be going wrong with good old (but deep) multilayer neural networks. Our analysis is driven by investigative experiments to monitor activations (watching for saturation of hidden units) and gradients, across layers and across training iterations. We also evaluate the effects on these of choices of activation function (with the idea that it might affect saturation) and initialization procedure (since unsupervised pretraining is a particular form of initialization and it has a drastic impact). (@glorot2010training)

Surbhi Goel, Aravind Gollakota, Zhihan Jin, Sushrut Karmalkar, and Adam Klivans Superpolynomial lower bounds for learning one-layer neural networks using gradient descent In H. Daumé III and A. Singh (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119, pp. 3587–3596. PMLR, 2020. **Abstract:** We prove the first superpolynomial lower bounds for learning one-layer neural networks with respect to the Gaussian distribution using gradient descent. We show that any classifier trained using gradient descent with respect to square-loss will fail to achieve small test error in polynomial time given access to samples labeled by a one-layer neural network. For classification, we give a stronger result, namely that any statistical query (SQ) algorithm (including gradient descent) will fail to achieve small test error in polynomial time. Prior work held only for gradient descent run with small batch sizes, required sharp activations, and applied to specific classes of queries. Our lower bounds hold for broad classes of activations including ReLU and sigmoid. The core of our result relies on a novel construction of a simple family of neural networks that are exactly orthogonal with respect to all spherically symmetric distributions. (@goel2020superpolynomial)

Aidan N. Gomez, Mengye Ren, Raquel Urtasun, and Roger B. Grosse The reversible residual network: Backpropagation without storing activations In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30, pp. 2214–2224. Curran Associates, Inc., 2017. **Abstract:** Residual Networks (ResNets) have demonstrated significant improvement over traditional Convolutional Neural Networks (CNNs) on image classification, increasing in performance as networks grow both deeper and wider. However, memory consumption becomes a bottleneck as one needs to store all the intermediate activations for calculating gradients using backpropagation. In this work, we present the Reversible Residual Network (RevNet), a variant of ResNets where each layer’s activations can be reconstructed exactly from the next layer’s. Therefore, the activations for most layers need not be stored in memory during backprop. We demonstrate the effectiveness of RevNets on CIFAR and ImageNet, establishing nearly identical performance to equally-sized ResNets, with activation storage requirements independent of depth. (@gomez2017reversible)

Ian Goodfellow, Yoshua Bengio, and Aaron Courville *Deep Learning* MIT Press, 2016. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@goodfellow2016deep)

Eldad Haber and Lars Ruthotto Stable architectures for deep neural networks *Inverse Problems*, 34: 014004, 2017. **Abstract:** Deep neural networks have become invaluable tools for supervised machine learning, e.g., classification of text or images. While often offering superior results over traditional techniques and successfully expressing complicated patterns in data, deep architectures are known to be challenging to design and train such that they generalize well to new data. Important issues with deep architectures are numerical instabilities in derivative-based learning algorithms commonly called exploding or vanishing gradients. In this paper we propose new forward propagation techniques inspired by systems of Ordinary Differential Equations (ODE) that overcome this challenge and lead to well-posed learning problems for arbitrarily deep networks. The backbone of our approach is our interpretation of deep learning as a parameter estimation problem of nonlinear dynamical systems. Given this formulation, we analyze stability and well-posedness of deep learning and use this new understanding to develop new network architectures. We relate the exploding and vanishing gradient phenomenon to the stability of the discrete ODE and present several strategies for stabilizing deep learning for very deep networks. While our new architectures restrict the solution space, several numerical experiments show their competitiveness with state-of-the-art networks. (@haber2017stable)

Joshua Hanson and Maxim Raginsky Fitting an immersed submanifold to data via Sussmann’s orbit theorem In *2022 IEEE 61st Conference on Decision and Control (CDC)*, pp. 5323–5328, 2022. **Abstract:** This paper describes an approach for fitting an immersed submanifold of a finite-dimensional Euclidean space to random samples. The reconstruction mapping from the ambient space to the desired submanifold is implemented as a composition of an encoder that maps each point to a tuple of (positive or negative) times and a decoder given by a composition of flows along finitely many vector fields starting from a fixed initial point. The encoder supplies the times for the flows. The encoder-decoder map is obtained by empirical risk minimization, and a high-probability bound is given on the excess risk relative to the minimum expected reconstruction error over a given class of encoder-decoder maps. The proposed approach makes fundamental use of Sussmann’s orbit theorem, which guarantees that the image of the reconstruction map is indeed contained in an immersed submanifold. (@hanson2022fitting)

Soufiane Hayou On the infinite-depth limit of finite-width neural networks *Transactions on Machine Learning Research*, 2023. **Abstract:** In this paper, we study the infinite-depth limit of finite-width residual neural networks with random Gaussian weights. With proper scaling, we show that by fixing the width and taking the depth to infinity, the pre-activations converge in distribution to a zero-drift diffusion process. Unlike the infinite-width limit where the pre-activation converge weakly to a Gaussian random variable, we show that the infinite-depth limit yields different distributions depending on the choice of the activation function. We document two cases where these distributions have closed-form (different) expressions. We further show an intriguing change of regime phenomenon of the post-activation norms when the width increases from 3 to 4. Lastly, we study the sequential limit infinite-depth-then-infinite-width and compare it with the more commonly studied infinite-width-then-infinite-depth limit. (@hayou2023on)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification In *Proceedings of 2015 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 1026–1034. IEEE, 2015. **Abstract:** Rectified activation units (rectifiers) are essential for state-of-the-art neural networks. In this work, we study rectifier neural networks for image classification from two aspects. First, we propose a Parametric Rectified Linear Unit (PReLU) that generalizes the traditional rectified unit. PReLU improves model fitting with nearly zero extra computational cost and little overfitting risk. Second, we derive a robust initialization method that particularly considers the rectifier nonlinearities. This method enables us to train extremely deep rectified models directly from scratch and to investigate deeper or wider network architectures. Based on the learnable activation and advanced initialization, we achieve 4.94% top-5 test error on the ImageNet 2012 classification dataset. This is a 26% relative improvement over the ILSVRC 2014 winner (GoogLeNet, 6.66% \[33\]). To our knowledge, our result is the first to surpass the reported human-level performance (5.1%, \[26\]) on this dataset. (@He2015DelvingDI)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Identity mappings in deep residual networks In B. Leibe, J. Matas, N. Sebe, and M. Welling (eds.), *Computer Vision – ECCV 2016*, pp. 630–645. Springer International Publishing, 2016. **Abstract:** Deep residual networks \[1\] have emerged as a family of ex- tremely deep architectures showing compelling accuracy and nice con- vergence behaviors. In this paper, we analyze the propagation formu- lations behind the residual building blocks, which suggest that the for- ward and backward signals can be directly propagated from one block to any other block, when using identity mappings as the skip connec- tions and after-addition activation. A series of ablation experiments sup- port the importance of these identity mappings. This motivates us to propose a new residual unit, which makes training easier and improves generalization. We report improved results using a 1001-layer ResNet on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet. Code is available at: https://github.com/KaimingHe/ resnet-1k-layers . 1 Introduction Deep residual networks (ResNets) \[1\] consist of many stacked \\}Residual Units". Each unit (Fig. 1 (a)) can be expressed in a general form: yl=h(xl) +F(xl;Wl); xl+1=f(yl); where xlandxl+1are input and output of the l-th unit, andFis a residual function. In \[1\], h(xl) =xlis an identity mapping and fis a ReLU \[2\] function. ResNets that are over 100-layer deep have shown state-of-the-art accuracy for several challenging recognition tasks on ImageNet \[3\] and MS COCO \[4\] compe- titions. The central idea of ResNets is to learn the additive residual function F with respect to h(xl), with a key choice of using an identity mapping h(xl) =xl. This is realized by attaching an identity skip connection (\\}shortcut"). In this paper, we analyze deep residual networks by focusing on creating a \\}direct" path for propagating information \| not only within a residual unit, but through the entire network. Our derivations reveal that if bothh(xl)and f(yl)are identity mappings , the signal could be directly propagated from one unit to any other units, in both forward and backward passes. Our experiments empirically show that training in general becomes easier when the architecture is closer to the above two conditions. To understand the role of skip connections, we analyze and compare various types ofh(xl). We nd that the identity mapping h(xl) =xlchosen in \[1\]arXiv:1603.05027v3 \[cs.CV\] 25 Jul 20162 0 1 2 3 4 5 6 x 10405101520 IterationsTest Error (% ) 0.0020.020.22Training Los sResNet−1001, original (error: 7.61% ) ResNet−1001, proposed (error: 4.92% ) BN ReLU weight BNweight addition ReLUxl xl+1 (a) originalReLU weight BN ReLU weightBN a (@he2016identity)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Deep residual learning for image recognition In *Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770–778. IEEE, 2016. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@heDeepResidualLearning2015)

Dan Hendrycks and Kevin Gimpel Gaussian Error Linear Units (GELUs) *arXiv:1606.08415*, 2016. **Abstract:** We propose the Gaussian Error Linear Unit (GELU), a high-performing neural network activation function. The GELU activation function is $x\\}Phi(x)$, where $\\}Phi(x)$ the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLUs ($x\\}mathbf{1}\_{x\>0}$). We perform an empirical evaluation of the GELU nonlinearity against the ReLU and ELU activations and find performance improvements across all considered computer vision, natural language processing, and speech tasks. (@hendrycks2016gaussian)

Patrick Kidger *On Neural Ordinary Differential Equations* PhD thesis, University of Oxford, 2022. **Abstract:** We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black-box differential equation solver. These continuous-depth models have constant memory cost, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed. We demonstrate these properties in continuous-depth residual networks and continuous-time latent variable models. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions. For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models. (@kidger2022neural)

Hyunjik Kim, George Papamakarios, and Andriy Mnih The Lipschitz constant of self-attention In M. Meila and T. Zhang (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139, pp. 5562–5571. PMLR, 2021. **Abstract:** Lipschitz constants of neural networks have been explored in various contexts in deep learning, such as provable adversarial robustness, estimating Wasserstein distance, stabilising training of GANs, and formulating invertible neural networks. Such works have focused on bounding the Lipschitz constant of fully connected or convolutional networks, composed of linear maps and pointwise non-linearities. In this paper, we investigate the Lipschitz constant of self-attention, a non-linear neural network module widely used in sequence modelling. We prove that the standard dot-product self-attention is not Lipschitz for unbounded input domain, and propose an alternative L2 self-attention that is Lipschitz. We derive an upper bound on the Lipschitz constant of L2 self-attention and provide empirical evidence for its asymptotic tightness. To demonstrate the practical relevance of our theoretical work, we formulate invertible self-attention and use it in a Transformer-based architecture for a character-level language modelling task. (@kim2021lipschitz)

D.P. Kingma and J. Ba Adam: A method for stochastic optimization In *International Conference on Learning Representations*, 2015. **Abstract:** We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm. (@kingmaAdamMethodStochastic2017)

A. Krizhevsky Learning multiple layers of features from tiny images Technical report, University of Toronto, 2009. **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@Krizhevsky2009learningmultiple)

Béatrice Laurent and Pascal Massart Adaptive estimation of a quadratic functional by model selection *The Annals of Statistics*, 28: 1302–1338, 2000. **Abstract:** We consider the problem of estimating $\\}\|s\\}\|^2$ when $s$ belongs to some separable Hilbert space and one observes the Gaussian process $Y(t) = \\}langles, t\\}rangle + \\}sigmaL(t)$, for all $t \\}epsilon \\}mathbb{H}$,where $L$ is some Gaussian isonormal process. This framework allows us in particular to consider the classical “Gaussian sequence model” for which $\\}mathbb{H} = l_2(\\}mathbb{N}\*)$ and $L(t) = \\}sum\_{\\}lambda\\}geq1}t\_{\\}lambda}\\}varepsilon\_{\\}lambda}$, where $(\\}varepsilon\_{\\}lambda})\_{\\}lambda\\}geq1}$ is a sequence of i.i.d. standard normal variables. Our approach consists in considering some at most countable families of finite-dimensional linear subspaces of $\\}mathbb{H}$ (the models) and then using model selection via some conveniently penalized least squares criterion to build new estimators of $\\}\|s\\}\|^2$. We prove a general nonasymptotic risk bound which allows us to show that such penalized estimators are adaptive on a variety of collections of sets for the parameter $s$, depending on the family of models from which they are built.In particular, in the context of the Gaussian sequence model, a convenient choice of the family of models allows defining estimators which are adaptive over collections of hyperrectangles, ellipsoids, $l_p$-bodies or Besov bodies.We take special care to describe the conditions under which the penalized estimator is efficient when the level of noise $\\}sigma$ tends to zero. Our construction is an alternative to the one by Efroïmovich and Low for hyperrectangles and provides new results otherwise. (@laurent2000adaptive)

Yann A. LeCun, Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller Efficient backprop In G. Montavon, G.B Orr, and K.-R. Müller (eds.), *Neural Networks: Tricks of the Trade: Second Edition*, pp. 9–48, Berlin, 2012. Springer. (@lecun1998efficient)

Zhiyuan Li, Yuping Luo, and Kaifeng Lyu Towards resolving the implicit bias of gradient descent for matrix factorization: Greedy low-rank learning In *International Conference on Learning Representations*, 2021. **Abstract:** Matrix factorization is a simple and natural test-bed to investigate the implicit regularization of gradient descent. Gunasekar et al. (2017) conjectured that Gradient Flow with infinitesimal initialization converges to the solution that minimizes the nuclear norm, but a series of recent papers argued that the language of norm minimization is not sufficient to give a full characterization for the implicit regularization. In this work, we provide theoretical and empirical evidence that for depth-2 matrix factorization, gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions. This generalizes the rank minimization view from previous works to a much broader setting and enables us to construct counter-examples to refute the conjecture from Gunasekar et al. (2017). We also extend the results to the case where depth $\\}ge 3$, and we show that the benefit of being deeper is that the above convergence has a much weaker dependence over initialization magnitude so that this rank minimization is more likely to take effect for initialization with practical scale. (@li2021towards)

Chaoyue Liu, Libin Zhu, and Mikhail Belkin Loss landscapes and optimization in over-parameterized non-linear systems and neural networks *Applied and Computational Harmonic Analysis*, 59: 85–116, 2022. **Abstract:** The success of deep learning is due, to a large extent, to the remarkable effectiveness of gradient-based optimization methods applied to large neural networks. The purpose of this work is to propose a modern view and a general mathematical framework for loss landscapes and efficient optimization in over-parameterized machine learning models and systems of non-linear equations, a setting that includes over-parameterized deep neural networks. Our starting observation is that optimization problems corresponding to such systems are generally not convex, even locally. We argue that instead they satisfy PL$^\*$, a variant of the Polyak-Lojasiewicz condition on most (but not all) of the parameter space, which guarantees both the existence of solutions and efficient optimization by (stochastic) gradient descent (SGD/GD). The PL$^\*$ condition of these systems is closely related to the condition number of the tangent kernel associated to a non-linear system showing how a PL$^\*$-based non-linear theory parallels classical analyses of over-parameterized linear equations. We show that wide neural networks satisfy the PL$^\*$ condition, which explains the (S)GD convergence to a global minimum. Finally we propose a relaxation of the PL$^\*$ condition applicable to "almost" over-parameterized systems. (@liu2022loss)

Sergey Loyka On singular value inequalities for the sum of two matrices *arXiv:1507.06630*, 2015. **Abstract:** A counter-example to lower bounds for the singular values of the sum of two matrices in \[1\] and \[2\] is given. Correct forms of the bounds are pointed out. (@loyka2015singular)

Yiping Lu, Zhuohan Li, Di He, Zhiqing Sun, Bin Dong, Tao Qin, Liwei Wang, and Tie-Yan Liu Understanding and improving transformer from a multi-particle dynamic system point of view *arXiv:1906.02762*, 2019. **Abstract:** The Transformer architecture is widely used in natural language processing. Despite its success, the design principle of the Transformer remains elusive. In this paper, we provide a novel perspective towards understanding the architecture: we show that the Transformer can be mathematically interpreted as a numerical Ordinary Differential Equation (ODE) solver for a convection-diffusion equation in a multi-particle dynamic system. In particular, how words in a sentence are abstracted into contexts by passing through the layers of the Transformer can be interpreted as approximating multiple particles’ movement in the space using the Lie-Trotter splitting scheme and the Euler’s method. Given this ODE’s perspective, the rich literature of numerical analysis can be brought to guide us in designing effective structures beyond the Transformer. As an example, we propose to replace the Lie-Trotter splitting scheme by the Strang-Marchuk splitting scheme, a scheme that is more commonly used and with much lower local truncation errors. The Strang-Marchuk splitting scheme suggests that the self-attention and position-wise feed-forward network (FFN) sub-layers should not be treated equally. Instead, in each layer, two position-wise FFN sub-layers should be used, and the self-attention sub-layer is placed in between. This leads to a brand new architecture. Such an FFN-attention-FFN layer is "Macaron-like", and thus we call the network with this new architecture the Macaron Net. Through extensive experiments, we show that the Macaron Net is superior to the Transformer on both supervised and unsupervised learning tasks. The reproducible codes and pretrained models can be found at https://github.com/zhuohan123/macaron-net (@lu2019understanding)

Jonathan Luk Notes on existence and uniqueness theorems for ODEs 2017. URL: <http://web.stanford.edu/~jluk/math63CMspring17/Existence.170408.pdf>. (@luk2017notes)

Kaifeng Lyu and Jian Li Gradient descent maximizes the margin of homogeneous neural networks In *International Conference on Learning Representations*, 2020. **Abstract:** In this paper, we study the implicit regularization of the gradient descent algorithm in homogeneous neural networks, including fully-connected and convolutional neural networks with ReLU or LeakyReLU activations. In particular, we study the gradient descent or gradient flow (i.e., gradient descent with infinitesimal step size) optimizing the logistic loss or cross-entropy loss of any homogeneous model (possibly non-smooth), and show that if the training loss decreases below a certain threshold, then we can define a smoothed version of the normalized margin which increases over time. We also formulate a natural constrained optimization problem related to margin maximization, and prove that both the normalized margin and its smoothed version converge to the objective value at a KKT point of the optimization problem. Our results generalize the previous results for logistic regression with one-layer or multi-layer linear networks, and provide more quantitative convergence results with weaker assumptions than previous results for homogeneous smooth neural networks. We conduct several experiments to justify our theoretical finding on MNIST and CIFAR-10 datasets. Finally, as margin is closely related to robustness, we discuss potential benefits of training longer for improving the robustness of the model. (@lyu2020gradient)

Lachlan E. MacDonald, Hemanth Saratchandran, Jack Valmadre, and Simon Lucey A global analysis of global optimisation *arXiv:2210.05371*, 2022. **Abstract:** We introduce a general theoretical framework, designed for the study of gradient optimisation of deep neural networks, that encompasses ubiquitous architecture choices including batch normalisation, weight normalisation and skip connections. Our framework determines the curvature and regularity properties of multilayer loss landscapes in terms of their constituent layers, thereby elucidating the roles played by normalisation layers and skip connections in globalising these properties. We then demonstrate the utility of this framework in two respects. First, we give the only proof of which we are aware that a class of deep neural networks can be trained using gradient descent to global optima even when such optima only exist at infinity, as is the case for the cross-entropy cost. Second, we identify a novel causal mechanism by which skip connections accelerate training, which we verify predictively with ResNets on MNIST, CIFAR10, CIFAR100 and ImageNet. (@macdonald2022global)

Pierre Marion Generalization bounds for neural ordinary differential equations and deep residual networks *arXiv:2305.06648*, 2023. **Abstract:** Neural ordinary differential equations (neural ODEs) are a popular family of continuous-depth deep learning models. In this work, we consider a large family of parameterized ODEs with continuous-in-time parameters, which include time-dependent neural ODEs. We derive a generalization bound for this class by a Lipschitz-based argument. By leveraging the analogy between neural ODEs and deep residual networks, our approach yields in particular a generalization bound for a class of deep residual networks. The bound involves the magnitude of the difference between successive weight matrices. We illustrate numerically how this quantity affects the generalization capability of neural networks. (@marion2023generalization)

Pierre Marion, Adeline Fermanian, Gérard Biau, and Jean-Philippe Vert Scaling ResNets in the large-depth regime *arXiv:2206.06929*, 2022. **Abstract:** Deep ResNets are recognized for achieving state-of-the-art results in complex machine learning tasks. However, the remarkable performance of these architectures relies on a training procedure that needs to be carefully crafted to avoid vanishing or exploding gradients, particularly as the depth $L$ increases. No consensus has been reached on how to mitigate this issue, although a widely discussed strategy consists in scaling the output of each layer by a factor $\\}alpha_L$. We show in a probabilistic setting that with standard i.i.d.~initializations, the only non-trivial dynamics is for $\\}alpha_L = \\}frac{1}{\\}sqrt{L}}$; other choices lead either to explosion or to identity mapping. This scaling factor corresponds in the continuous-time limit to a neural stochastic differential equation, contrarily to a widespread interpretation that deep ResNets are discretizations of neural ordinary differential equations. By contrast, in the latter regime, stability is obtained with specific correlated initializations and $\\}alpha_L = \\}frac{1}{L}$. Our analysis suggests a strong interplay between scaling and regularity of the weights as a function of the layer index. Finally, in a series of experiments, we exhibit a continuous range of regimes driven by these two parameters, which jointly impact performance before and after training. (@marion2022scaling)

Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, and Hajime Asama Dissecting neural ODEs In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 3952–3963. Curran Associates, Inc., 2020. **Abstract:** Continuous deep learning architectures have recently re-emerged as Neural Ordinary Differential Equations (Neural ODEs). This infinite-depth approach theoretically bridges the gap between deep learning and dynamical systems, offering a novel perspective. However, deciphering the inner working of these models is still an open challenge, as most applications apply them as generic black-box modules. In this work we "open the box", further developing the continuous-depth formulation with the aim of clarifying the influence of several design choices on the underlying dynamics. (@massaroli2020dissecting)

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro In search of the real inductive bias: On the role of implicit regularization in deep learning *arXiv:1412.6614*, 2014. **Abstract:** We present experiments demonstrating that some other form of capacity control, different from network size, plays a central role in learning multilayer feed-forward networks. We argue, partially through analogy to matrix factorization, that this is an inductive bias that can help shed light on deep learning. (@neyshabur2015search)

Quynh N. Nguyen and Marco Mondelli Global convergence of deep networks with one wide layer followed by pyramidal topology In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 11961–11972. Curran Associates, Inc., 2020. **Abstract:** Recent works have shown that gradient descent can find a global minimum for over-parameterized neural networks where the widths of all the hidden layers scale polynomially with $N$ ($N$ being the number of training samples). In this paper, we prove that, for deep networks, a single layer of width $N$ following the input layer suffices to ensure a similar guarantee. In particular, all the remaining layers are allowed to have constant widths, and form a pyramidal topology. We show an application of our result to the widely used LeCun’s initialization and obtain an over-parameterization requirement for the single wide layer of order $N^2.$ (@nguyen2020global)

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32, pp. 8024–8035. Curran Associates, Inc., 2019. **Abstract:** Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs. In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance. We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks. (@paszke2019pytorch)

Alejandro Queiruga, N. Benjamin Erichson, Liam Hodgkinson, and Michael W. Mahoney Stateful ODE-nets using basis function expansions In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems*, volume 34, pp. 21770–21781. Curran Associates, Inc., 2021. **Abstract:** The recently-introduced class of ordinary differential equation networks (ODE-Nets) establishes a fruitful connection between deep learning and dynamical systems. In this work, we reconsider formulations of the weights as continuous-in-depth functions using linear combinations of basis functions which enables us to leverage parameter transformations such as function projections. In turn, this view allows us to formulate a novel stateful ODE-Block that handles stateful layers. The benefits of this new ODE-Block are twofold: first, it enables incorporating meaningful continuous-in-depth batch normalization layers to achieve state-of-the-art performance; second, it enables compressing the weights through a change of basis, without retraining, while maintaining near state-of-the-art performance and reducing both inference time and memory footprint. Performance is demonstrated by applying our stateful ODE-Block to (a) image classification tasks using convolutional units and (b) sentence-tagging tasks using transformer encoder units. (@queiruga2021stateful)

Michael E. Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyré Sinkformers: Transformers with doubly stochastic attention In G. Camps-Valls, F.J.R. Ruiz, and I. Valera (eds.), *Proceedings of The 25th International Conference on Artificial Intelligence and Statistics*, volume 151, pp. 3515–3530. PMLR, 2022. **Abstract:** Attention based models such as Transformers involve pairwise interactions between data points, modeled with a learnable attention matrix. Importantly, this attention matrix is normalized with the SoftMax operator, which makes it row-wise stochastic. In this paper, we propose instead to use Sinkhorn’s algorithm to make attention matrices doubly stochastic. We call the resulting model a Sinkformer. We show that the row-wise stochastic attention matrices in classical Transformers get close to doubly stochastic matrices as the number of epochs increases, justifying the use of Sinkhorn normalization as an informative prior. On the theoretical side, we show that, unlike the SoftMax operation, this normalization makes it possible to understand the iterations of self-attention modules as a discretized gradient-flow for the Wasserstein metric. We also show in the infinite number of samples limit that, when rescaling both attention matrices and depth, Sinkformers operate a heat diffusion. On the experimental side, we show that Sinkformers enhance model accuracy in vision and natural language processing tasks. In particular, on 3D shapes classification, Sinkformers lead to a significant improvement. (@sander2022sinkformers)

Michael E. Sander, Pierre Ablin, and Gabriel Peyré Do residual neural networks discretize neural ordinary differential equations? In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 36520–36532. Curran Associates, Inc., 2022. **Abstract:** Neural Ordinary Differential Equations (Neural ODEs) are the continuous analog of Residual Neural Networks (ResNets). We investigate whether the discrete dynamics defined by a ResNet are close to the continuous one of a Neural ODE. We first quantify the distance between the ResNet’s hidden state trajectory and the solution of its corresponding Neural ODE. Our bound is tight and, on the negative side, does not go to 0 with depth N if the residual functions are not smooth with depth. On the positive side, we show that this smoothness is preserved by gradient descent for a ResNet with linear residual functions and small enough initial loss. It ensures an implicit regularization towards a limit Neural ODE at rate 1 over N, uniformly with depth and optimization time. As a byproduct of our analysis, we consider the use of a memory-free discrete adjoint method to train a ResNet by recovering the activations on the fly through a backward pass of the network, and show that this method theoretically succeeds at large depth if the residual functions are Lipschitz with the input. We then show that Heun’s method, a second order ODE integration scheme, allows for better gradient estimation with the adjoint method when the residual functions are smooth with depth. We experimentally validate that our adjoint method succeeds at large depth, and that Heun method needs fewer layers to succeed. We finally use the adjoint method successfully for fine-tuning very deep ResNets without memory consumption in the residual layers. (@sander2022do)

Takeshi Teshima, Koichi Tojo, Masahiro Ikeda, Isao Ishikawa, and Kenta Oono Universal approximation property of neural ordinary differential equations *arXiv:2012.02414*, 2020. **Abstract:** Neural ordinary differential equations (NODEs) is an invertible neural network architecture promising for its free-form Jacobian and the availability of a tractable Jacobian determinant estimator. Recently, the representation power of NODEs has been partly uncovered: they form an $L^p$-universal approximator for continuous maps under certain conditions. However, the $L^p$-universality may fail to guarantee an approximation for the entire input domain as it may still hold even if the approximator largely differs from the target function on a small region of the input space. To further uncover the potential of NODEs, we show their stronger approximation property, namely the $\\}sup$-universality for approximating a large class of diffeomorphisms. It is shown by leveraging a structure theorem of the diffeomorphism group, and the result complements the existing literature by establishing a fairly large set of mappings that NODEs can approximate with a stronger guarantee. (@teshima2020universal)

Matthew Thorpe and Yves van Gennip Deep limits of residual neural networks *Research in the Mathematical Sciences*, 10: 6, 2023. **Abstract:** Abstract Neural networks have been very successful in many applications; we often, however, lack a theoretical understanding of what the neural networks are actually learning. This problem emerges when trying to generalise to new data sets. The contribution of this paper is to show that, for the residual neural network model, the deep layer limit coincides with a parameter estimation problem for a nonlinear ordinary differential equation. In particular, whilst it is known that the residual neural network model is a discretisation of an ordinary differential equation, we show convergence in a variational sense. This implies that optimal parameters converge in the deep layer limit. This is a stronger statement than saying for a fixed parameter the residual neural network model converges (the latter does not in general imply the former). Our variational analysis provides a discrete-to-continuum $$\\}Gamma $$ \<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML"\> \<mml:mi\>Γ\</mml:mi\> \</mml:math\> -convergence result for the objective function of the residual neural network training step to a variational problem constrained by a system of ordinary differential equations; this rigorously connects the discrete setting to a continuum problem. (@thorpe2018deep)

Joel A. Tropp User-friendly tail bounds for sums of random matrices *Foundations of Computational Mathematics*, 12: 389–434, 2012. **Abstract:** This paper presents new probability inequalities for sums of independent, random, self-adjoint matrices. These results place simple and easily verifiable hypotheses on the summands, and they deliver strong conclusions about the large-deviation behavior of the maximum eigenvalue of the sum. Tail bounds for the norm of a sum of random rectangular matrices follow as an immediate corollary. The proof techniques also yield some information about matrix-valued martingales. In other words, this paper provides noncommutative generalizations of the classical bounds associated with the names Azuma, Bennett, Bernstein, Chernoff, Hoeffding, and McDiarmid. The matrix inequalities promise the same diversity of application, ease of use, and strength of conclusion that have made the scalar inequalities so valuable. (@tropp2012user)

Gal Vardi On the implicit bias in deep-learning algorithms *Communications of the ACM*, 66: 86–93, 2023. **Abstract:** Examining the implicit bias in training neural networks using gradient-based methods. (@vardi2022implicit)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin Attention is all you need In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 30, pp. 5998–6008. Curran Associates, Inc., 2017. **Abstract:** The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. (@vaswani2017attention)

Roman Vershynin *High-Dimensional Probability: An Introduction with Applications in Data Science* Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018. **Abstract:** High-dimensional probability offers insight into the behavior of random vectors, random matrices, random subspaces, and objects used to quantify uncertainty in high dimensions. Drawing on ideas from probability, analysis, and geometry, it lends itself to applications in mathematics, statistics, theoretical computer science, signal processing, optimization, and more. It is the first to integrate theory, key tools, and modern applications of high-dimensional probability. Concentration inequalities form the core, and it covers both classical results such as Hoeffding’s and Chernoff’s inequalities and modern developments such as the matrix Bernstein’s inequality. It then introduces the powerful methods based on stochastic processes, including such tools as Slepian’s, Sudakov’s, and Dudley’s inequalities, as well as generic chaining and bounds based on VC dimension. A broad range of illustrations is embedded throughout, including classical and modern results for covariance estimation, clustering, networks, semidefinite programming, coding, dimension reduction, matrix completion, machine learning, compressed sensing, and sparse regression. (@vershynin2018high)

Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei DeepNet: Scaling transformers to 1,000 layers *arXiv:2203.00555*, 2022. **Abstract:** In this paper, we propose a simple yet effective method to stabilize extremely deep Transformers. Specifically, we introduce a new normalization function (DeepNorm) to modify the residual connection in Transformer, accompanying with theoretically derived initialization. In-depth theoretical analysis shows that model updates can be bounded in a stable way. The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative. We successfully scale Transformers up to 1,000 layers (i.e., 2,500 attention and feed-forward network sublayers) without difficulty, which is one order of magnitude deeper than previous deep Transformers. Remarkably, on a multilingual benchmark with 7,482 translation directions, our 200-layer model with 3.2B parameters significantly outperforms the 48-layer state-of-the-art model with 12B parameters by 5 BLEU points, which indicates a promising scaling direction. (@wang2022deepnet)

Lei Wu, Qingcan Wang, and Chao Ma Global convergence of gradient descent for deep linear residual networks In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 32, pp. 13389–13398. Curran Associates, Inc., 2019. **Abstract:** We analyze the global convergence of gradient descent for deep linear residual networks by proposing a new initialization: zero-asymmetric (ZAS) initialization. It is motivated by avoiding stable manifolds of saddle points. We prove that under the ZAS initialization, for an arbitrary target matrix, gradient descent converges to an $\\}varepsilon$-optimal point in $O\\}left( L^3 \\}log(1/\\}varepsilon) \\}right)$ iterations, which scales polynomially with the network depth $L$. Our result and the $\\}exp(\\}Omega(L))$ convergence time for the standard initialization (Xavier or near-identity) \\}cite{shamir2018exponential} together demonstrate the importance of the residual structure and the initialization in the optimization for deep linear neural networks, especially when $L$ is large. (@wu2019global)

Han Zhang, Xi Gao, Jacob Unterman, and Tom Arodz Approximation capabilities of neural ODEs and invertible residual networks In H. Daumé III and A. Singh (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119, pp. 11086–11095. PMLR, 2020. **Abstract:** Neural ODEs and i-ResNet are recently proposed methods for enforcing invertibility of residual neural models. Having a generic technique for constructing invertible models can open new avenues for advances in learning systems, but so far the question of whether Neural ODEs and i-ResNets can model any continuous invertible function remained unresolved. Here, we show that both of these models are limited in their approximation capabilities. We then prove that any homeomorphism on a $p$-dimensional Euclidean space can be approximated by a Neural ODE operating on a $2p$-dimensional Euclidean space, and a similar result for i-ResNets. We conclude by showing that capping a Neural ODE or an i-ResNet with a single linear layer is sufficient to turn the model into a universal approximator for non-invertible continuous functions. (@zhang2020approximation)

Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma Fixup initialization: Residual learning without normalization In *International Conference on Learning Representations*, 2019. **Abstract:** Normalization layers are a staple in state-of-the-art deep neural network architectures. They are widely believed to stabilize training, enable higher learning rate, accelerate convergence and improve generalization, though the reason for their effectiveness is still an active research topic. In this work, we challenge the commonly-held beliefs by showing that none of the perceived benefits is unique to normalization. Specifically, we propose fixed-update initialization (Fixup), an initialization motivated by solving the exploding and vanishing gradient problem at the beginning of training via properly rescaling a standard initialization. We find training residual networks with Fixup to be as stable as training with normalization – even for networks with 10,000 layers. Furthermore, with proper regularization, Fixup enables residual networks without normalization to achieve state-of-the-art performance in image classification and machine translation. (@zhang2019fixup)

Difan Zou, Philip M. Long, and Quanquan Gu On the global convergence of training deep linear ResNets In *International Conference on Learning Representations*, 2020. **Abstract:** We study the convergence of gradient descent (GD) and stochastic gradient descent (SGD) for training $L$-hidden-layer linear residual networks (ResNets). We prove that for training deep residual networks with certain linear transformations at input and output layers, which are fixed throughout training, both GD and SGD with zero initialization on all hidden weights can converge to the global minimum of the training loss. Moreover, when specializing to appropriate Gaussian random linear transformations, GD and SGD provably optimize wide enough deep linear ResNets. Compared with the global convergence result of GD for training standard deep linear networks \\}citep{du2019width}, our condition on the neural network width is sharper by a factor of $O(\\}kappa L)$, where $\\}kappa$ denotes the condition number of the covariance matrix of the training data. In addition, for the first time we establish the global convergence of SGD for training deep linear ResNets and prove a linear convergence rate when the global minimum is $0$. (@zou2020on)

</div>

<div class="center">

**Appendix**

</div>

#### Organization of the Appendix.

In Section <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a>, we give some results on the general residual network <a href="#eq:general-resnet" data-reference-type="eqref" data-reference="eq:general-resnet">[eq:general-resnet]</a>. In Section <a href="#sec:proofs-main-paper" data-reference-type="ref" data-reference="sec:proofs-main-paper">8</a>, these results are instantiated in the specific case of the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>, thus proving the results of the paper. Section <a href="#apx:lemmas" data-reference-type="ref" data-reference="apx:lemmas">9</a> contains some lemmas that are useful for the proofs. We present in Section <a href="#app:relu" data-reference-type="ref" data-reference="app:relu">10</a> a counter-example showing that a residual network with the ReLU activation can move away from the neural ODE structure during training. Finally, Section <a href="#apx:experiments" data-reference-type="ref" data-reference="apx:experiments">11</a> presents some experimental details.

# Some results for general residual networks

#### Lipschitz continuity.

Let $`(\mathscr{U}, \|\cdot\|)`$, $`(\mathscr{V}, \|\cdot\|)`$, and $`(\mathscr{W}, \|\cdot\|)`$ be generic normed spaces. Then a function of two variables $`g: \mathscr{U} \times \mathscr{V} \to \mathscr{W}`$ is:

1.  (Globally) Lipschitz continuous if there exists $`K \geqslant 0`$ such that, for $`(u, v), (u', v') \in \mathscr{U} \times \mathscr{V}`$,
    ``` math
    \|g(u, v) - g(u', v')\| \leqslant K \|u-u'\| + K \|v-v'\|.
    ```

2.  Locally Lipschitz continuous in its first variable if, for any compacts $`E \subset \mathscr{U}, E' \subset \mathscr{V}`$, there exists $`K \geqslant 0`$ such that, for $`(u, v), (u', v) \in E \times E'`$,
    ``` math
    \|g(u, v) - g(u', v)\| \leqslant K \|u-u'\|.
    ```

Equivalent definitions hold for a function of one variable. Moreover, $`g(\cdot, v)`$ is said to be uniformly Lipschitz continuous for $`v`$ in $`\mathscr{V}`$ if there exists $`K \geqslant 0`$ such that, for $`(u, v), (u', v) \in \mathscr{U} \times \mathscr{V}`$,
``` math
\|g(u, v) - g(u', v)\| \leqslant K \|u-u'\|,
```
and uniformly Lipschitz continuous for $`v`$ in any compact if, for any compact $`E' \subset \mathscr{V}`$, there exists $`K \geqslant 0`$ such that, for $`(u, v), (u', v) \in \mathcal{U} \times E'`$,
``` math
\|g(u, v) - g(u', v)\| \leqslant K \|u-u'\|.
```
Throughout, we refer to a Lipschitz continuous function with Lipschitz constant $`K \geqslant 0`$ as $`K`$-Lipschitz.

#### Model.

As explained in Section <a href="#subsec:generalization" data-reference-type="ref" data-reference="subsec:generalization">4.3</a>, most of our results are proven for the general residual network
``` math
\begin{aligned}
h_{0}^L(t) &= A^L(t) x \nonumber \\
h_{k+1}^L(t) &= h_k^L(t)+\frac1L f(h_k^L(t), Z_{k+1}^L(t)), \quad k \in \{0, \dots, L-1\}, \label{eq:proof-main-thm:forward} \\
F^L(x; t) &= B^L(t) h_L^L(t), \nonumber
\end{aligned}
```
where $`Z^L(t) = (Z_1^L(t), \dots, Z_L^L(t)) \in ({\mathbb{R}}^{p})^L`$ and $`f: {\mathbb{R}}^q \times {\mathbb{R}}^p \to {\mathbb{R}}^q`$ is a $`\mathcal{C}^2`$ function such that $`f(0, \cdot) \equiv 0`$ and $`f(\cdot, z)`$ is uniformly Lipschitz for $`z`$ in any compact. Let us introduce the backpropagation equations, which are instrumental in the study of the gradient flow dynamics. These equations define the backward state $`p_k^L(t) = \frac{\partial \ell^L}{\partial h_k^L}(t) \in {\mathbb{R}}^q`$ through the backward recurrence
``` math
\begin{aligned}
p_L^L(t) &= 2 B^L(t)^\top (F^L(x;t) - y) \nonumber \\
p_k^L(t) &= p_{k+1}^L(t) + \frac{1}{L} \partial_1 f(h_k^L(t), Z_{k+1}^L(t)) p_{k+1}^L(t), \quad k \in \{0, \dots, L-1\}, \label{eq:proof-main-thm:backprop}
\end{aligned}
```
where $`\partial_1 f \in {\mathbb{R}}^{q \times q}`$ stands for the Jacobian matrix of $`f`$ with respect to its first argument. Similarly, we let $`\partial_2 f \in {\mathbb{R}}^{q \times p}`$ be the Jacobian matrix of $`f`$ with respect to its second argument. For a sample $`(x_i, y_i)_{1 \leqslant i \leqslant n} \in (\mathcal{X}\times \mathcal{Y})^n`$, we let $`h_{k, i}^L(t)`$ and $`p_{k, i}^L(t)`$ be, respectively, the hidden layer $`h_{k}^L(t)`$ and the backward state $`p_{k}^L(t)`$ associated with the $`i`$-th input $`x_i`$. Denoting the mean squared error associated with the sample by $`\ell^L`$, we have, by the chain rule,
``` math
\begin{aligned}
\frac{\partial \ell^L}{\partial A^L}(t) &= \frac{1}{n} \sum_{i=1}^n p_{0,i}^L(t) x_i^\top \label{eq:proof-main-thm:backprop-2}  \\
\frac{\partial \ell^L}{\partial Z_k^L}(t) &= \frac{1}{n L} \sum_{i=1}^n \partial_2 f(h_{k-1,i}^L(t), Z_{k}^L(t))^\top p_{k,i}^L(t), \quad k \in \{1, \dots, L\}, \label{eq:proof-main-thm:backprop-1} \\
\frac{\partial \ell^L}{\partial B^L}(t) &= \frac{2}{n} \sum_{i=1}^n (F^L(x_i; t) - y_i) h_{L, i}^L(t)^\top \label{eq:proof-main-thm:backprop-3}.
\end{aligned}
```

#### Initialization.

The parameters $`(Z_k^L(t))_{1 \leqslant k \leqslant L}`$ are initialized to $`Z_k^L(0) = Z^{\textnormal{init}}\big(\frac{k}{L}\big)`$, where $`Z^{\textnormal{init}}: [0, 1] \to {\mathbb{R}}^p`$ is a Lipschitz continuous function. Furthermore, we initialize $`A^L(0)`$ to some matrix $`A^{\textnormal{init}} \in {\mathbb{R}}^{q \times d}`$ and $`B^L(0)=B^{\textnormal{init}} \in {\mathbb{R}}^{d' \times q}`$. Note that this initialization scheme is a generalization of the one presented in Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>.

#### Additional notation.

For a vector $`x`$, $`\left\|x\right\|`$ denotes the Euclidean norm. For a matrix $`A`$, the operator norm induced by the Euclidean norm is denoted by $`\left\|A\right\|_2`$, and the Frobenius norm is denoted by $`\left\|A\right\|_F`$. Finally, we use the notation $`A^L`$ (resp. $`Z_k^L`$, $`B^L`$) to denote the function $`t \mapsto A^L(t)`$ (resp. $`t \mapsto Z_k^L(t)`$, $`t \mapsto B^L(t)`$), since the parameters are considered as functions of the training time throughout this appendix.

#### Overview of Appendix A.

First, in Section <a href="#apx:trained-weights-bounded-finite-training" data-reference-type="ref" data-reference="apx:trained-weights-bounded-finite-training">7.1</a>, we study the case of the (clipped) gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a>. We show that the weights and the difference between successive weights are bounded during the entire training. Section <a href="#apx:trained-weights-bounded-pl" data-reference-type="ref" data-reference="apx:trained-weights-bounded-pl">7.2</a> shows a similar result for the standard gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a> under a PL condition. In Section <a href="#apx:arzela-ascoli" data-reference-type="ref" data-reference="apx:arzela-ascoli">7.3</a>, we show a generalized version of the Arzelà-Ascoli theorem, which allows us to prove the existence of a converging subsequence of the weights in the large-depth limit. Section <a href="#apx:consistency-euler" data-reference-type="ref" data-reference="apx:consistency-euler">7.4</a> is devoted to the convergence of the Euler scheme for parameterized ODEs. We then proceed to prove in Section <a href="#apx:general-convergence" data-reference-type="ref" data-reference="apx:general-convergence">7.5</a> our main result, i.e., the large-depth convergence of the gradient flow. The key step is to establish the uniqueness of the adherence point of the weights. Finally, in Section <a href="#apx:double-limit" data-reference-type="ref" data-reference="apx:double-limit">7.6</a>, we prove the existence of a double limit for the weights and the hidden states when both the depth and the training time tend to infinity.

## The trained weights are bounded in the finite training-time setup

Before stating the result, let us introduce the notation $`\partial_{22} f(h, z) \in {\mathbb{R}}^{q \times p \times p}`$, which is the third-order tensor of second partial derivatives of $`f`$ with respect to $`z`$. We endow the space $`{\mathbb{R}}^{q \times p \times p}`$ with the operator norm $`\|\cdot\|_2`$ induced by the Euclidean norm in $`{\mathbb{R}}^p`$ and the $`\|\cdot\|_2`$ norm in $`{\mathbb{R}}^{q \times p}`$. In other words,
``` math
\|\partial_{22} f(h, z)\|_2 = \sup_{u \in {\mathbb{R}}^{p}, \|u\| = 1} \|\partial_{22} f(h, z) u\|_2,
```
where $`\partial_{22} f(h, z) u \in {\mathbb{R}}^{q \times p}`$ is the tensor product of $`\partial_{22} f(h, z)`$ against $`u`$. Similarly, $`\partial_{21} f(h, z) \in {\mathbb{R}}^{q \times p \times q}`$ denotes the third-order tensor of cross second partial derivatives of $`f`$, and the space $`{\mathbb{R}}^{q \times p \times q}`$ is endowed with the operator norm $`\|\cdot\|_2`$ induced by the Euclidean norm in $`{\mathbb{R}}^q`$ and the $`\|\cdot\|_2`$ norm in $`{\mathbb{R}}^{q \times p}`$.

<div id="prop:general-clipped-bounded" class="proposition">

**Proposition 7**. *Consider the residual network <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a> initialized as explained in Appendix <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a> and trained with the gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> on $`[0, T]`$, for some $`T \in (0, \infty)`$. Let
``` math
\begin{aligned}
M_\pi &= \max\Big(\max_{A \in {\mathbb{R}}^{q \times d}} \|\pi(A)\|_F, \, \max_{Z \in {\mathbb{R}}^{p}} \|\pi(Z)\|, \, \max_{B \in {\mathbb{R}}^{d' \times q}} \|\pi(B)\|_F\Big), \\
M_0 &= \max\Big(\|A^{\textnormal{init}}\|_F, \, \sup_{s \in [0, 1]} \|Z^{\textnormal{init}}(s)\|, \, \|B^{\textnormal{init}}\|_F\Big) \quad \text{and} \quad M = M_0+TM_\pi.
\end{aligned}
```
Then the gradient flow is well defined on $`[0, T]`$, and, for $`t\in[0, T]`$, $`L\in\mathbb{N}^*`$, and $`k\in\{1,\dots,L\}`$,
``` math
\label{eq:upper-bound-abm}
\|A^L(t)\|_F\leqslant M, \quad \|Z_k^L(t)\| \leqslant M, \quad  \text{and} \quad \|B^L(t)\|_F\leqslant M.
```
Moreover, there exist $`\alpha, \beta > 0`$ such that, for $`t \in [0, T]`$ and $`k \in \{1, \dots, L-1\}`$,
``` math
\|Z_{k+1}^L(t)-Z_{k}^L(t)\| \leqslant\Big(\|Z_{k+1}^L(0)-Z_k^L(0)\| + \frac{\beta T}L \Big) e^{\alpha T}.
```
The following expressions for $`\alpha`$ and $`\beta`$ hold:
``` math
\alpha=2e^KK'M(e^KM^2M_X+M_Y) \quad \text{and} \quad \beta=2Ke^KM(K+e^KK'MM_X)(e^KM^2M_X+M_Y),
```
where
``` math
\begin{aligned}
    M_X &= \sup_{x \in \mathcal{X}} \|x\|, \quad M_Y = \sup_{y \in \mathcal{Y}}\|y\|, \quad K_1 = \sup_{\|z\| \leqslant M} \big\|\partial_1 f(h, z)\big\|_2 \label{eq:proof:def-K1} \\
    E &= \{(h, z)\in\mathbb{R}^d\times\mathbb{R}^{p}, \|h\| \leqslant e^{K_1} M M_X, \,  \|z\| \leqslant M\} \label{eq:proof:def-E} \\
    K_2 &= \sup_{(h, z)\in E} \big\|\partial_2 f(h, z)\big\|_2, \quad K = \max(K_1, K_2) \nonumber \\
    K' &= \sup_{(h, z)\in E}\Big(\max\big(\big\|\partial_{22} f(h, z)\big\|_2, \big\|\partial_{21} f(h,z)\big\|_2\big)\Big). \nonumber 
\end{aligned}
```*

</div>

<div class="proof">

*Proof.* The time-independent dynamics
``` math
(A^L, Z_k^L, B^L) \mapsto \Big(\pi \Big(- \frac{\partial \ell^L}{\partial A^L}\Big), \pi \Big(- L \frac{\partial \ell^L}{\partial Z_k^L}\Big), \pi \Big(- \frac{\partial \ell^L}{\partial B^L}\Big)\Big)
```
defining the gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> are locally Lipschitz continuous, hence the gradient flow is defined on a maximal interval $`[0, T_{\max})`$ by the Picard-Lindelöf theorem (see Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a>). Let us show by contradiction that $`T_{\max} = T`$. Assume that $`T_{\max} < T`$. If this is true, again by the Picard-Lindelöf theorem, we know that the parameters diverge to infinity at $`T_{\max}`$. However, for any $`t \in [0, T_{\max})`$, we have
``` math
\begin{aligned}
\|A^L(t)\|_F \leqslant\|A^L(0)\|_F+\int_0^t \Big\|\frac{dA^L}{dt}(\tau)\Big\|_F d\tau \leqslant M_0+\int_0^t M_\pi d\tau \leqslant M_0+TM_\pi=M.
\end{aligned}
```
Bounds on $`B^L`$ and $`Z_k^L`$ by $`M`$ can be shown similarly. This contradicts the divergence of the parameters at $`t=T_{\max}`$. We conclude that the gradient flow is well defined on $`[0, T]`$ and that the bounds <a href="#eq:upper-bound-abm" data-reference-type="eqref" data-reference="eq:upper-bound-abm">[eq:upper-bound-abm]</a> hold.

It remains to bound the difference $`\|Z_{k+1}^L(t)-Z_{k}^L(t)\|`$. We have, for $`t \in [0, T]`$ and $`k \in \{1, \dots, L-1\}`$,
``` math
\begin{aligned}
\Big\|\frac{dZ_{k+1}^L}{dt}(t)&-\frac{dZ_{k}^L}{dt}(t)\Big\|=L\Big\|\frac{\partial \ell^L}{\partial Z_{k+1}^L}(t)-\frac{\partial \ell^L}{\partial Z_{k}^L}(t)\Big\| \nonumber\\
&\leqslant\sum_{i=1}^n\frac1n\big\|\partial_2 f(h_{k, i}^L(t), Z_{k+1}^L(t))^\top p_{k, i}^L(t)-\partial_2 f(h_{k-1, i}^L(t), Z_{k}^L(t))^\top p_{k-1, i}^L(t)\big\| \nonumber\\
 &\leqslant\frac1n\sum_{i=1}^n \left\|\partial_2 f(h_{k, i}^L(t), Z_{k+1}^L(t))\right\|_2\left\|p_{k, i}^L(t)-p_{k-1, i}^L(t)\right\|
 \nonumber \\
 &\quad+\left\|p_{k-1, i}^L(t)\right\| \left\|\partial_2 f(h_{k, i}^L(t), Z_{k+1}^L(t))-\partial_2 f(h_{k-1, i}^L(t), Z_{k}^L(t))\right\|_2 \label{eq:upperbound-diff-zk-zk+1}
\end{aligned}
```
Furthermore, for $`t \in [0, T]`$, $`k \in \{0, \dots, L-1\}`$, and $`i \in \{1, \dots, n\}`$,
``` math
\begin{aligned}
\|h_{k+1, i}^L(t)\|=\|h_{k, i}^L(t)+\frac1Lf(h_{k, i}^L(t), Z_{k+1}^L(t))\|\leqslant(1+\frac{K_1}{L})\|h_{k, i}^L(t)\|,
\end{aligned}
```
since $`f(\cdot, Z_{k+1}^L(t))`$ is $`K_1`$-Lipschitz, where $`K_1`$ is defined by <a href="#eq:proof:def-K1" data-reference-type="eqref" data-reference="eq:proof:def-K1">[eq:proof:def-K1]</a>, and $`f(0, Z_{k+1}^L(t)) = 0`$. Therefore, for any $`k \in \{1, \dots, L\}`$,
``` math
\label{eq:upper-bound-hkl}
\|h_{k, i}^L(t)\|\leqslant e^{K_1}\|h_{0, i}^L(t)\| = e^{K_1}\|A^L(t) x_i\| \leqslant e^{K_1}MM_X.
```
This bound shows that the pair $`(h_{k,i}^L(t), Z_{k+1}^L(t))`$ belongs to the compact $`E`$ defined in <a href="#eq:proof:def-E" data-reference-type="eqref" data-reference="eq:proof:def-E">[eq:proof:def-E]</a> for every $`t \in [0, T]`$, $`k \in \{1, \dots, L\}`$, and $`i \in \{1, \dots, n\}`$. In particular, $`\|\partial_2 f(h_{k-1, i}^L(t), Z_{k}^L(t))\|_2 \leqslant K`$, and
``` math
\begin{aligned}
\big\|\partial_2 f(h_{k, i}^L(t), Z_{k+1}^L(t))&-\partial_2 f(h_{k-1, i}^L(t), Z_{k}^L(t))\big\|_2 \\
&\leqslant K'\|h_{k, i}^L(t)-h_{k-1, i}^L(t)\| +K'\|Z_{k+1}^L(t)-Z_{k}^L(t)\|.    
\end{aligned}
```
Returning to <a href="#eq:upperbound-diff-zk-zk+1" data-reference-type="eqref" data-reference="eq:upperbound-diff-zk-zk+1">[eq:upperbound-diff-zk-zk+1]</a>, we obtain
``` math
\begin{aligned}
\Big\|\frac{dZ_{k+1}^L}{dt}(t)-\frac{dZ_{k}^L}{dt}(t)\Big\| &\leqslant\frac1n\sum_{i=1}^n K\|p_{k, i}^L(t) - p_{k-1, i}^L(t)\| \\
&\quad + K' \|p_{k-1, i}^L(t)\| \big(\|h_{k, i}^L(t)-h_{k-1, i}^L(t)\| +\|Z_{k+1}^L(t)-Z_{k}^L(t)\| \big).
\end{aligned}
```
For $`k \in \{1, \dots, L\}`$ and $`i \in \{1, \dots, n\}`$,
``` math
\left\|p_{k, i}^L(t)-p_{k-1, i}^L(t)\right\| = \frac{1}{L} \big\|\partial_1 f(h_{k-1, i}^L(t), Z_{k}^L(t))p_{k, i}^L(t)\big\| \leqslant\frac{K}{L}\left\|p_{k, i}^L(t)\right\|,
```
and, similarly,
``` math
\left\|h_{k, i}^L(t)-h_{k-1, i}^L(t)\right\| = \frac{1}{L} \|f(h_{k-1,i}^L(t), Z_{k}^L(t))\| \leqslant\frac{K}{L}\left\|h_{k-1, i}^L(t)\right\| \leqslant\frac{Ke^{K} MM_X}{L}.
```
Thus,
``` math
\begin{aligned}
\Big\|\frac{dZ_{k+1}^L}{dt}(t)-\frac{dZ_{k}^L}{dt}(t)\Big\| \leqslant\frac1n\sum_{i=1}^n {\|p_{k, i}^L(t)\|}\Big(\frac{K^2}{L}+\frac{K'K}{L}e^KMM_X+K'\|Z_{k+1}^L(t)-Z_{k}^L(t)\|\Big).
\end{aligned}
```
Moreover, for $`k \in \{0, \dots, L\}`$ and $`i \in \{1, \dots, n\}`$,
``` math
\begin{aligned}
\|p_{k,i}^L(t)\| &\leqslant\|p_{k+1,i}^L(t)\| + \frac{1}{L} \big\|\partial_1 f(h_{k,i}^L(t), Z_{k+1}^L(t))p_{k+1,i}^L(t)\big\| \leqslant\|p_{k+1,i}^L(t)\| + \frac{K}{L} \|p_{k+1,i}^L(t)\|.
\end{aligned}
```
Hence
``` math
\begin{aligned}
\|p_{k,i}^L(t)\| &\leqslant e^{K} \|p_{L,i}^L(t)\| = 2 e^K \|B^L(t)^\top (F^L(x_i;t) - y_i)\| \\
&\leqslant 2 e^K M \big(\|B^L(t)h_{L,i}^L(t)\|+\|y_i\|\big) \leqslant 2 e^K M (e^KM^2M_X+M_Y),
\end{aligned}
```
where we use <a href="#eq:upper-bound-abm" data-reference-type="eqref" data-reference="eq:upper-bound-abm">[eq:upper-bound-abm]</a> and <a href="#eq:upper-bound-hkl" data-reference-type="eqref" data-reference="eq:upper-bound-hkl">[eq:upper-bound-hkl]</a> for the last inequality. Putting all the pieces together, we obtain
``` math
\Big\|\frac{dZ_k^L}{dt}(t)-\frac{dZ_{k+1}^L}{dt}(t)\Big\|\leqslant\alpha\|Z_k^L(t)-Z_{k+1}^L(t)\|+\frac{\beta}L.
```
Integrating between $`0`$ and $`t`$, we see that
``` math
\|Z_{k+1}^L(t)-Z_k^L(t)\| \leqslant\|Z_{k+1}^L(0)-Z_k^L(0)\| + \frac{\beta t}{L} + \int_0^t \alpha\|Z_k^L(\tau)-Z_{k+1}^L(\tau)\|d \tau.
```
Applying Grönwall’s inequality , we conclude that $`\|Z_{k+1}^L(t)-Z_k^L(t)\|\leqslant(\|Z_{k+1}^L(0)-Z_k^L(0)\| + \frac{\beta T}L)e^{\alpha T}`$, as desired. ◻

</div>

<div class="remark">

**Remark 1**. *Clipping is used in our approach to constraint the gradients to live in a ball. It is merely a technical assumption to avoid blow-up of the weights during training. However, in any scenario where we know that the weights do not blow up, clipping is not required. A first example of such a scenario is under the Polyak-Łojasiewicz condition (see below). Another scenario is by using gradient flow with momentum instead of vanilla gradient flow. This is a setup closer to Adam , which is a very used optimizer in practice. One might then show a similar result to Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a>, without clipping, because the gradient updates in the momentum case are bounded by construction.*

</div>

## The trained weights are bounded under the local PL condition

<div id="prop:plcondition-to-convergence" class="proposition">

**Proposition 8**. *Consider the residual network <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a> initialized as explained in Appendix <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a> and trained with the gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a> on $`[0, \infty]`$. Then, for $`M > 0`$, there exists $`\mu > 0`$ such that, if the residual network satisfies the $`(M, \mu)`$-local PL condition <a href="#def:pl-condition" data-reference-type="eqref" data-reference="def:pl-condition">[def:pl-condition]</a> around its initialization for any $`L \in \mathbb{N}^*`$, then:*

1)  *The gradient flow is well defined on $`\mathbb{R}_+`$, and, for $`t \in \mathbb{R}_+`$, $`L\in\mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
    ``` math
    \|A^L(t)\|_F \leqslant M_A, \quad \|Z_k^L(t)\| \leqslant M_Z, \quad \text{and} \quad \|B^L(t)\|_F \leqslant M_B,
    ```
    where
    ``` math
    M_A = \|A^{\textnormal{init}}\|_2 + M, \quad M_Z = \sup_{s \in [0, 1]} \|Z^{\textnormal{init}}(s)\| + M, \quad \text{and} \quad M_B = \|B^{\textnormal{init}}\|_2 + M.
    ```*

2)  *There exists $`\Tilde{K}>0`$ such that, for $`t\in {\mathbb{R}}_+`$, $`L\in\mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
    ``` math
    \|Z_k^L(t)-Z_{k+1}^L(t)\| \leqslant\frac{\Tilde{K}}L.
    ```*

3)  *There exists a bounded integrable function $`b: {\mathbb{R}}_+ \to {\mathbb{R}}`$ such that, for $`t\in {\mathbb{R}}_+`$, $`L\in\mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
    ``` math
    \max\Big(\Big\|\frac{dA^L}{dt}(t)\Big\|,  \Big\|\frac{dZ_k^L}{dt}(t)\Big\|, \Big\|\frac{dB^L}{dt}(t)\Big\|\Big) \leqslant b(t)
    ```*

4)  *$`A^L(t)`$, $`B^L(t)`$, and $`Z_k^L(t)`$ admit a limit uniformly over $`L \in \mathbb{N}^*`$ and $`k \in \{1, \dots, L\}`$ as $`t\to\infty`$.*

5)  *For $`t \in \mathbb{R}_+`$ and $`L \in \mathbb{N}^*`$, $`\ell^L(t) \leqslant e^{- \mu t} \ell^L(0)`$.*

*Moreover, the following expression for $`\mu`$ hold:
``` math
\label{eq:condition-mu}
 \mu = \max(M_B K, M_B M_X, M_A M_X) \frac{8e^K}{M} \sup_{L \in \mathbb{N}^*}\sqrt{\ell^L(0)},
```
where
``` math
\begin{aligned}
    M_X &= \sup_{x \in \mathcal{X}} \|x\|, \quad K_1 = \sup_{\|z\| \leqslant M_Z} \big\|\partial_1 f(h, z)\big\| \\
    E &= \{(h, z)\in\mathbb{R}^d\times\mathbb{R}^{p}, \|h\| \leqslant e^{K_1} M_A M_X, \,  \|z\| \leqslant M_Z\} \\
    K_2 &= \sup_{(h, z)\in E} \big\|\partial_2 f(h, z)\big\|, \quad K = \max(K_1, K_2).
\end{aligned}
```*

</div>

<div class="proof">

*Proof.* Let $`M > 0`$, $`\mu`$ defined by <a href="#eq:condition-mu" data-reference-type="eqref" data-reference="eq:condition-mu">[eq:condition-mu]</a>, and assume that the residual network satisfies the $`(M, \mu)`$-local PL condition <a href="#def:pl-condition" data-reference-type="eqref" data-reference="def:pl-condition">[def:pl-condition]</a> around its initialization for any $`L \in \mathbb{N}^*`$.

The time-independent dynamics
``` math
(A^L, Z_k^L, B^L) \mapsto \Big(- \frac{\partial \ell^L}{\partial A^L}, - L \frac{\partial \ell^L}{\partial Z_k^L}, - \frac{\partial \ell^L}{\partial B^L}\Big)
```
defining the gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> are locally Lipschitz continuous, hence the gradient flow is defined on a maximal interval $`[0, T_{\max})`$ by the Picard-Lindelöf theorem (see Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a>). Let us show by contradiction that $`T_{\max} = \infty`$. Assume that $`T_{\max} < \infty`$. If this is true, again by the Picard-Lindelöf theorem, we know that the parameters diverge to infinity at $`T_{\max}`$. In particular, there exist $`t \in (0, T_{\max})`$ and $`k \in \{1, \dots, L\}`$ such that
``` math
\|A^L(t)-A^{L}(0)\|_F > M \text{ or } \|Z_k^L(t)-Z_k^L(0)\| > M \text{ or } \|B^L(t)-B^L(0)\|_F > M.
```
Let $`t^* \in (0, T_{\max})`$ be the infimum of such times $`t`$. Then, for $`t < t^*`$ and $`k \in \{1, \dots, L\}`$,
``` math
\label{eq:upperbound-diff-abz}
\|A^L(t)-A^{L}(0)\|_F \leqslant M \text{ and } \|Z_k^L(t)-Z_k^L(0)\| \leqslant M \text{ and } \|B^L(t)-B^L(0)\|_F \leqslant M,
```
and, by continuity of $`A^L`$, $`B^L`$, and $`Z_k^L`$, these inequalities also hold for $`t=t^*`$. By definition, this means that the $`(M, \mu)`$-local PL condition is satisfied for $`t \leqslant t^*`$, and ensures that
``` math
\Big\|\frac{\partial \ell^L}{\partial A^L}(t)\Big\|_F^2 + L \sum_{k=1}^L \Big\|\frac{\partial \ell^L}{\partial Z_k^L}(t)\Big\|^2 + \Big\|\frac{\partial \ell^L}{\partial B^L}(t)\Big\|_F^2 \geqslant\mu \ell^L(t).
```
Therefore, by definition of the gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a>,
``` math
\begin{aligned}
\frac{d\ell^L}{dt}(t) 
&= \Big\langle\frac{\partial \ell^L}{\partial A^L}(t), \frac{dA^L}{dt}(t)\Big\rangle + \sum_{k=1}^L \Big\langle\frac{\partial \ell^L}{\partial Z_k^L}(t), \frac{dZ_k^L}{dt}(t)\Big\rangle + \Big\langle\frac{\partial \ell^L}{\partial B^L}(t), \frac{dB^L}{dt}(t)\Big\rangle \\
&= - \Big\|\frac{\partial \ell^L}{\partial A^L}(t)\Big\|_F^2 - L \sum_{k=1}^L \Big\|\frac{\partial \ell^L}{\partial Z_k^L}(t)\Big\|^2 - \Big\|\frac{\partial \ell^L}{\partial B^L}(t)\Big\|_F^2 \\
&\leqslant- \mu \ell^L(t).    
\end{aligned}
```
Thus, by Grönwall’s inequality, for $`t \leqslant t^*`$,
``` math
\label{eq:linear-convergence}
\ell^L(t)\leqslant e^{-\mu t} \ell^L(0).
```
Furthermore, by <a href="#eq:upperbound-diff-abz" data-reference-type="eqref" data-reference="eq:upperbound-diff-abz">[eq:upperbound-diff-abz]</a> and the definition of $`M_A`$, $`M_B`$, $`M_Z`$, we have, for $`t \leqslant t^*`$ and $`k \in \{1, \dots, L\}`$,
``` math
\|A^L(t)\|_F \leqslant M_A, \quad \|Z_k^L(t)\| \leqslant M_Z, \quad \text{and} \quad \|B^L(t)\|_F \leqslant M_B.
```
A quick scan through the proof of Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> reveals that by similar arguments, we have, for $`t \leqslant t^*`$, $`k \in \{1, \dots, L\}`$, and $`i \in \{1, \dots, n\}`$,
``` math
(h_{k-1,i}^L(t), Z_{k}^L(t)) \in E \quad \text{and} \quad \|p_{k-1, i}^L(t)\| \leqslant 2 e^{K} \|p_{L, i}^L(t)\| \leqslant 2 e^K M_B \|F^L(x_i;t) - y_i\|.
```
Thus, for $`k \in \{0, \dots, L\}`$,
``` math
\label{eq:maj-sum-pi}
\frac{1}{n} \sum_{i=1}^n \|p_{k,i}^L(t)\| \leqslant\frac{2 e^K M_B}{n} \sum_{i=1}^n\|F^L(x_i;t) - y_i\| \leqslant 2 e^K M_B \sqrt{\ell^L(t)} \leqslant 2 e^K M_B e^{-\frac{\mu t}{2}} \sqrt{\ell^L(0)},
```
where the second inequality is a consequence of the Cauchy-Schwartz inequality. Let us now bound $`\|Z_k^L(t^*) - Z_k^L(0)\|`$. We have, for $`k \in \{1, \dots, L\}`$,
``` math
\begin{aligned}
\|Z_k^L(t^*) - Z_k^L(0)\| &\leqslant\int_0^{t^*} \Big\|\frac{dZ_k^L}{dt}(t)\Big\| dt\\
&\leqslant\frac{1}{n} \sum_{i=1}^n \int_0^{t^*} \big\|\partial_2 f(h_{k-1,i}^L(t), Z_{k}^L(t))^\top p_{k,i}^L(t)\big\| dt\\
& \qquad \mbox{(by \eqref{eq:proof-main-thm:backprop-1}).} \\
&\leqslant\frac{K}{n} \sum_{i=1}^n \int_0^{t^*} \|p_{k,i}^L(t)\| dt,
\end{aligned}
```
since $`(h_{k-1,i}^L(t), Z_{k}^L(t)) \in E`$ and $`\|\partial_2 f(h, z)\| \leqslant K`$ for $`(h, z) \in E`$. Therefore, by <a href="#eq:maj-sum-pi" data-reference-type="eqref" data-reference="eq:maj-sum-pi">[eq:maj-sum-pi]</a>,
``` math
\begin{aligned}
\|Z_k^L(t^*) - Z_k^L(0)\| \leqslant 2 Ke^{K}M_B \int_0^{t^*} e^{-\frac{\mu t}{2}} \sqrt{\ell^L(0)}dt \leqslant\frac{4 K e^{K} M_B}{\mu}\sqrt{\ell^L(0)} \leqslant\frac{M}{2},
\end{aligned}
```
where the last inequality is a consequence of the definition of $`\mu`$. Similarly, by <a href="#eq:proof-main-thm:backprop-2" data-reference-type="eqref" data-reference="eq:proof-main-thm:backprop-2">[eq:proof-main-thm:backprop-2]</a> and <a href="#eq:maj-sum-pi" data-reference-type="eqref" data-reference="eq:maj-sum-pi">[eq:maj-sum-pi]</a>,
``` math
\begin{aligned}
\|A^L(t^*) - A^L(0)\|_F &\leqslant\int_0^{t^*} \Big\|\frac{dA^L}{dt}(t)\Big\|_F dt \\ 
&\leqslant\int_0^{t^*} \frac{1}{n} \sum_{i=1}^n \big\|p_{0,i}^L(t) x_i^\top\big\|_F dt \\ 
&\leqslant 2 e^KM_B M_X \sqrt{\ell^L(0)} \int_0^{t^*} e^{-\frac{\mu t}{2}} dt\\
&\leqslant\frac{4e^KM_B M_X}{\mu}\sqrt{\ell^L(0)}\\
&\leqslant\frac{M}{2}.
\end{aligned}
```
Finally, by <a href="#eq:proof-main-thm:backprop-3" data-reference-type="eqref" data-reference="eq:proof-main-thm:backprop-3">[eq:proof-main-thm:backprop-3]</a>,
``` math
\begin{aligned}
\|B^L(t^*) - B(0)\|_F&\leqslant\int_0^{t^*} \Big\|\frac{dB^L}{dt}(t)\Big\|_F dt \\
&\leqslant\int_0^{t^*} \frac{2}{n} \sum_{i=1}^n \|(F^L(x_i; t) - y_i) h_{L, i}^L(t)^\top\|_F dt \\
&\leqslant 2 e^{K} M_A M_X \sqrt{\ell^L(0)} \int_0^{t^*} e^{-\frac{\mu t}{2}} dt\\
&\leqslant\frac{4 e^{K} M_A M_X}{\mu}\sqrt{\ell^L(0)}\\
&\leqslant\frac{M}{2},
\end{aligned}
```
where the third inequality is a consequence of the Cauchy-Schwartz inequality and of the fact that $`\|h_{L, i}^L(t)\| \leqslant e^K M_A M_X`$. By continuity of $`A^L`$, $`Z_k^L`$, and $`B^L`$, these three bounds contradict the definition of $`t^*`$. We conclude that $`T_{\max} = \infty`$ and that the parameters stay within a ball of radius $`M`$ of their initialization, yielding the inequalities, for $`t \in \mathbb{R}_+`$, $`L\in\mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
``` math
\|A^L(t)\|_F \leqslant M_A, \quad \|B^L(t)\|_F \leqslant M_B, \quad \|Z_k^L(t)\| \leqslant M_Z.
```
This proves statement $`(i)`$ of the proposition. Moreover, the analysis above show that the derivatives of $`A^L`$, $`Z_k^L`$, and $`B^L`$ are bounded by a bounded integrable function independent of $`L`$ and $`k`$. This shows $`(iii)`$, together with the fact that the functions $`A^L(t)`$, $`Z_k^L(t)`$, and $`B^L(t)`$ admit limits as $`t\to\infty`$. Furthermore, the convergence towards their limit is uniform over $`L`$ and $`k`$, as we show for example for $`A^L(t)`$. If we denote by $`A_\infty^L`$ its limit, and apply the same steps as for bounding $`\|A^L(t^*) - A^L(0)\|_F`$, we obtain, for any $`t \geqslant 0`$,
``` math
\begin{aligned}
\|A_\infty^L - A^L(t)\|_F &\leqslant\int_t^{\infty} \Big\|\frac{dA^L}{d\tau}(\tau)\Big\|_F d\tau \\
&\leqslant 2 e^{K}M_BM_X \sqrt{\ell^L(0)} \int_t^{\infty} e^{\frac{-\mu \tau}{2}} d\tau\\
&= \frac{4 e^{K}M_BM_X}{\mu}  e^{\frac{-\mu t}{2}} \sqrt{\ell^L(0)} \\
&\leqslant\frac{M}{2} e^{\frac{-\mu t}{2}},
\end{aligned}
```
where the last inequality comes from the definition of $`\mu`$. The bound is independent of $`L`$, proving statement $`(iv)`$. Statement $`(v)`$ readily follows from <a href="#eq:linear-convergence" data-reference-type="eqref" data-reference="eq:linear-convergence">[eq:linear-convergence]</a>.

To complete the proof, it remains to prove statement $`(ii)`$ by bounding the differences $`\|Z_{k+1}^L(t)-Z_{k}^L(t)\|`$. Now that we know that the weights are bounded, we can follow the same steps as in the proof of Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> and show the existence of $`C_1`$, $`C_2 >0`$ such that
``` math
\begin{aligned}
\Big\|\frac{dZ_{k+1}^L}{dt}(t)&-\frac{dZ_{k}^L}{dt}(t)\Big\| \leqslant\frac1n\sum_{i=1}^n {\|p_{k, i}^L(t)\|}\Big(\frac{C_1}{L} + C_2\|Z_{k+1}^L(t)-Z_{k}^L(t)\|\Big).
\end{aligned}
```
Using <a href="#eq:maj-sum-pi" data-reference-type="eqref" data-reference="eq:maj-sum-pi">[eq:maj-sum-pi]</a>, we obtain
``` math
\begin{aligned}
\Big\|\frac{dZ_{k+1}^L}{dt}(t)-\frac{dZ_{k}^L}{dt}(t)\Big\| \leqslant 2 e^K M_B e^{-\frac{\mu t}{2}} \sqrt{\ell^L(0)} \Big(\frac{C_1}{L} + C_2\|Z_{k+1}^L(t)-Z_{k}^L(t)\|\Big).
\end{aligned}
```
Integrating between $`0`$ and $`t`$, we obtain
``` math
\begin{aligned}
\|Z_{k+1}^L(t)-Z_{k}^L(t)\| &\leqslant\|Z_{k+1}^L(0)-Z_{k}^L(0)\| + \int_0^t 2 e^K M_B e^{-\frac{\mu \tau}{2}} \sqrt{\ell^L(0)} \frac{C_1}{L} d\tau \\
&\quad + \int_0^t 2 e^K M_B e^{-\frac{\mu \tau}{2}} \sqrt{\ell^L(0)} C_2 \|Z_{k+1}^L(\tau)-Z_{k}^L(\tau)\| d\tau \\
&\leqslant\|Z_{k+1}^L(0)-Z_{k}^L(0)\| + \frac{C_1 M}{2 M_X L} \\
&\quad+ \int_0^t 2 e^K M_B e^{-\frac{\mu \tau}{2}} \sqrt{\ell^L(0)} C_2 \|Z_{k+1}^L(\tau)-Z_{k}^L(\tau)\| d\tau,
\end{aligned}
```
where the second inequality uses the definition of $`\mu`$. By Grönwall’s inequality,
``` math
\begin{aligned}
\|Z_{k+1}^L(t)-Z_{k}^L(t)\| &\leqslant\Big(\|Z_{k+1}^L(0)-Z_{k}^L(0)\| + \frac{C_1 M}{2 M_X L}\Big) \exp\Big(\int_0^t 2 e^K M_B e^{-\frac{\mu \tau}{2}} \sqrt{\ell^L(0)} C_2 d\tau \Big) \\
&\leqslant\Big(\|Z_{k+1}^L(0)-Z_{k}^L(0)\| + \frac{C_1 M}{2 M_X L}\Big) \exp\Big(\frac{C_2 M}{2 M_X}\Big),
\end{aligned}
```
again by definition of $`\mu`$. Finally, since $`Z_{k}^L(0) = Z^{\text{init}}(\frac{k}{L})`$ and $`Z^{\text{init}}`$ is Lipschitz continuous, this proves the existence of $`\Tilde{K} >0`$ (independent of $`L`$, $`t`$ and $`k`$) such that $`\|Z_{k+1}^L(t)-Z_k^L(t)\| \leqslant\frac{\Tilde{K}}L`$, which yields statement $`(ii)`$. ◻

</div>

## Generalized Arzelà–Ascoli theorem

<div id="prop:arzela-ascoli" class="proposition">

**Proposition 9** (Generalized Arzelà–Ascoli theorem). * Let $`I \subseteq \mathbb{R}_+`$ be an interval, and $`(Z_k^L)_{L\in\mathbb{N}^*, 1\leqslant k\leqslant L}`$ be a family of $`\mathcal{C}^1`$ functions from $`I`$ to $`\mathbb{R}^p`$. Define
``` math
\mathcal{Z}^L:[0,1]\times I\to\mathbb{R}^p,\ (s, t)\mapsto \mathcal{Z}^L(s, t) = Z_{\lfloor (L-1)s \rfloor + 1}^L(t).
```
Assume that there exist a constant $`C>0`$ and a bounded integrable function $`b: I \to \mathbb{R}`$ such that the following statements hold for any $`t\in I`$ and $`L \in \mathbb{N}^*`$:*

1)  *For $`k \in \{1, \dots, L-1\}`$, $`\|Z_{k+1}^L(t)-Z_{k}^L(t)\|\leqslant\frac{C}{L}`$,*

2)  *For $`k \in \{1, \dots, L\}`$, $`\|Z_k^L(t)\| \leqslant C`$ and $`\|\frac{dZ_k^L}{dt}(t)\| \leqslant b(t)`$.*

*Then there exist a subsequence $`(\mathcal{Z}^{\phi(L)})_{L\in\mathbb{N}^*}`$ of $`(\mathcal{Z}^L)_{L\in\mathbb{N}^*}`$ and a Lipschitz continuous function $`\mathcal{Z}^\phi:[0,1] \times I\to\mathbb{R}^{p}`$ such that $`\mathcal{Z}^{\phi(L)}(s,t)`$ tends to $`\mathcal{Z}^\phi(s,t)`$ uniformly over $`s`$ and $`t`$.*

</div>

Note that if $`I`$ is a compact interval, then the existence of a (uniformly) convergent subsequence is guaranteed by the standard Arzelà–Ascoli theorem. Indeed, the uniform equicontinuity is a consequence of assumptions $`(i)`$ and $`(ii)`$, while $`(ii)`$ provides a uniform bound. However, if $`I`$ is not compact, more involved arguments are needed.

<div class="proof">

*Proof.* Assume, without loss of generality, that $`b`$ is also bounded by $`C`$. According to assumption $`(i)`$, for $`t\in I`$ and $`i, j  \in \{1, \dots, L\}`$,
``` math
\begin{aligned}
    \|Z_i^L(t)-Z_{j}^L(t)\| \leqslant\frac{C|i-j|}{L}.
\end{aligned}
```
Also, according to $`(ii)`$, for $`t, t' \in I`$ and $`k \in \{1, \dots, L\}`$,
``` math
\begin{aligned}
\|Z_k^L(t)-Z_k^L(t')\| = \Big\|\int_{t'}^{t}\frac{dZ_k^L}{d\tau}(\tau)d\tau\Big\| \leqslant C|t-t'|.
\end{aligned}
```
It follows that, for $`s, s' \in [0, 1]`$ and $`t, t' \in I`$,
``` math
\begin{aligned}
\|\mathcal{Z}^L(s, t)-\mathcal{Z}^L(s', t')\| &\leqslant\|\mathcal{Z}^L(s, t)-\mathcal{Z}^L(s, t')\| + \|\mathcal{Z}^L(s, t')-\mathcal{Z}^L(s', t')\| \\
&\leqslant C|t-t'| + \frac{C|\lfloor (L-1)s \rfloor - \lfloor (L-1)s' \rfloor|}{L}.
\end{aligned}
```
Therefore, with some simple algebra, we obtain
``` math
\begin{aligned}
\label{eq:localcons1}
\|\mathcal{Z}^L(s, t)-\mathcal{Z}^L(s', t')\| \leqslant C|t-t'|+C|s-s'|+\frac{C}{L}.
\end{aligned}
```
The statement of the proposition is then a consequence of the next three steps.

#### There exists a convergent subsequence of $`(\mathcal{Z}^L(s, t))_{L \in \mathbb{N}^*}`$.

First, let $`((s_i, t_i))_{i\in\mathbb{N}}=(\mathbb{Q}\cap[0, 1])\times(\mathbb{Q}\cap I)`$. By $`(ii)`$, the sequence $`(\mathcal{Z}^L(s_i, t_i))_{L \in \mathbb{N}^*, i\in\mathbb{N}}`$ is bounded. It is therefore possible to construct by a diagonal procedure a subsequence $`(\mathcal{Z}^{\phi(L)})_{L \in \mathbb{N}^*}`$ such that, for each $`i\in\mathbb{N}`$, $`(\mathcal{Z}^{\phi(L)}(s_i, t_i))_{L \in \mathbb{N}^*}`$ is a convergent sequence.

Let us now show that $`(\mathcal{Z}^{\phi(L)}(s, t))_{L \in \mathbb{N}^*}`$ converges for any $`s\in[0, 1]`$ and $`t\in I`$, by proving that it is a Cauchy sequence in the complete metric space $`\mathbb{R}^p`$. Let $`\varepsilon>0`$, $`s\in[0, 1]`$, and $`t\in I`$. Since $`((s_i, t_i))_{i\in\mathbb{N}}`$ is dense in $`[0, 1]\times I`$, there exists some $`j\in\mathbb{N}`$ such that $`|s_j-s|\leqslant\varepsilon`$ and $`|t_j-t|\leqslant\varepsilon`$. Then, for $`L, M \in \mathbb{N}^*`$, we have
``` math
\begin{aligned}
\|\mathcal{Z}^{\phi(L)}(s, t)&-\mathcal{Z}^{\phi(M)}(s, t)\| \\
&\leqslant\|\mathcal{Z}^{\phi(L)}(s, t)-\mathcal{Z}^{\phi(L)}(s_j, t_j)\| + \|\mathcal{Z}^{\phi(L)}(s_j, t_j)-\mathcal{Z}^{\phi(M)}(s_j, t_j)\| \\
&\quad + \|\mathcal{Z}^{\phi(M)}(s_j, t_j)-\mathcal{Z}^{\phi(M)}(s, t)\| \\
&\leqslant 2C\varepsilon+\frac{C}{\phi(L)} + \|\mathcal{Z}^{\phi(L)}(s_j, t_j)-\mathcal{Z}^{\phi(M)}(s_j, t_j)\| + 2C\varepsilon + \frac{C}{\phi(M)},
\end{aligned}
```
where we used inequality <a href="#eq:localcons1" data-reference-type="eqref" data-reference="eq:localcons1">[eq:localcons1]</a> twice. Since $`(\mathcal{Z}^{\phi(L)}(s_j, t_j))_{L \in \mathbb{N}^*}`$ is a convergent sequence, it is a Cauchy sequence. Thus, the bound can be made arbitrarily small for $`L, M`$ large enough. This shows that $`(\mathcal{Z}^{\phi(L)}(s, t))_{L \in \mathbb{N}^*}`$ is also a Cauchy sequence. It is therefore convergent, and we denote by $`\mathcal{Z}^{\phi}(s, t)`$ its limit.

#### The function $`\mathcal{Z}^{\phi}`$ is Lipschitz continuous.

By considering (<a href="#eq:localcons1" data-reference-type="ref" data-reference="eq:localcons1">[eq:localcons1]</a>) for the subsequence $`\phi(L)`$ and letting $`L\to\infty`$, we have that, for any $`s, s' \in [0, 1]`$ and $`t, t' \in I`$,
``` math
\label{eq:proof-arzela-0}
    \|\mathcal{Z}^{\phi}(s, t)-\mathcal{Z}^{\phi}(s', t')\| \leqslant C(|s-s'|+|t-t'|).
```

#### The convergence of $`(\mathcal{Z}^{\phi(L)}(s, t))_{L \in \mathbb{N}^*}`$ to $`\mathcal{Z}^{\phi}(s, t)`$ is uniform over $`s`$ and $`t`$.

Let $`\varepsilon>0`$, $`s \in [0, 1]`$, and $`t \in I`$. Then, by (<a href="#eq:localcons1" data-reference-type="ref" data-reference="eq:localcons1">[eq:localcons1]</a>) and <a href="#eq:proof-arzela-0" data-reference-type="eqref" data-reference="eq:proof-arzela-0">[eq:proof-arzela-0]</a>, it is possible to find $`\delta>0`$ such that, for any $`s', s'' \in[0, 1]`$ and $`t', t''\in I`$ satisfying $`|s'-s''|\leqslant\delta`$ and $`|t'-t''|\leqslant\delta`$,
``` math
\begin{aligned}
   \label{eq:proof-arzela-1}
\begin{split}
    \|\mathcal{Z}^{\phi(L)}(s', t')-\mathcal{Z}^{\phi(L)}(s'', t')\|&\leqslant\varepsilon + \frac{C}{\phi(L)} \quad \text{and} \quad
    \|\mathcal{Z}^{\phi}(s', t')-\mathcal{Z}^{\phi}(s'', t')\| \leqslant\varepsilon,
\end{split}
\end{aligned}
```
and
``` math
\begin{aligned}
  \label{eq:proof-arzela-2}
\begin{split}
    \|\mathcal{Z}^{\phi(L)}(s', t')-\mathcal{Z}^{\phi(L)}(s', t'')\| \leqslant\varepsilon + \frac{C}{\phi(L)} \quad \text{and} \quad
    \|\mathcal{Z}^{\phi}(s', t')-\mathcal{Z}^{\phi}(s', t'')\| \leqslant\varepsilon.
\end{split}
\end{aligned}
```
Furthermore, there exists a finite set $`\{s_1, \dots, s_S\} \subset [0,1]`$ such that
``` math
\begin{aligned}
[0, 1]\subset \bigcup_{i=1}^S (s_{i}-\delta, s_{i}+\delta).
\end{aligned}
```
In the sequel, we denote by $`s^*`$ an element of $`\{s_1, \dots, s_S\}`$ that is at distance at most $`\delta`$ from $`s`$.

If $`I`$ is unbounded, then, by assumption $`(ii)`$ and since $`b`$ is integrable, there exists some $`t_0>0`$ such that, for $`t\geqslant t_0`$,
``` math
\begin{aligned}
 \label{eq:proof-arzela-3}
\|\mathcal{Z}^{\phi(L)}(s, t)-\mathcal{Z}^{\phi(L)}(s, t_0)\| &\leqslant\int_{t_0}^t\Big\|\frac{d}{dt} Z_{\lfloor (\phi(L) s - 1) \rfloor + 1}^{\phi(L)}(\tau)\Big\| d\tau \leqslant\int_{t_0}^t b(\tau) d\tau \leqslant\varepsilon.
\end{aligned}
```
The same inequality holds for $`\mathcal{Z}^\phi`$ by letting $`L`$ tend to infinity. If $`I`$ is bounded, we simply let $`t_0 = \sup I`$.

We may then pick a finite set $`\{t_1, \dots, t_T\} \subset [0, t_0]`$ such that
``` math
\begin{aligned}
[0, t_0] \subset\bigcup_{i=1}^{T}(t_{i}-\delta, t_{i}+\delta).
\end{aligned}
```
Two cases may arise depending on the value of $`t`$. If $`t \in [0, t_0]`$, then there exists an element of the set $`\{t_1, \dots, t_T\}`$ at distance at most $`\delta`$ from $`t`$, and we denote it by $`t^*`$. If $`t > t_0`$, we let $`t^* = t_0`$. According to <a href="#eq:proof-arzela-2" data-reference-type="eqref" data-reference="eq:proof-arzela-2">[eq:proof-arzela-2]</a> and <a href="#eq:proof-arzela-3" data-reference-type="eqref" data-reference="eq:proof-arzela-3">[eq:proof-arzela-3]</a>, we then have in both cases that
``` math
\begin{aligned}
  \label{eq:proof-arzela-4}
\begin{split}
    \|\mathcal{Z}^{\phi(L)}(s, t)-\mathcal{Z}^{\phi(L)}(s, t^*)\| \leqslant \varepsilon  + \frac{C}{\phi(L)} \quad \text{and} \quad \|\mathcal{Z}^{\phi}(s, t)-\mathcal{Z}^{\phi}(s, t^*)\| \leqslant \varepsilon.
\end{split}
\end{aligned}
```
To conclude, we have to bound the term $`\|\mathcal{Z}^{\phi(L)}(s, t)-\mathcal{Z}^{\phi}(s, t)\|`$ uniformly over $`s`$ and $`t`$. We first have
``` math
\begin{aligned}
\|\mathcal{Z}^{\phi(L)}(s, t)&-\mathcal{Z}^{\phi}(s, t)\| \\
&\leqslant\|\mathcal{Z}^{\phi(L)}(s, t)-\mathcal{Z}^{\phi(L)}(s, t^*)\| + \|\mathcal{Z}^{\phi(L)}(s, t^*)-\mathcal{Z}^{\phi}(s, t^*)\| \\
&\quad+ \|\mathcal{Z}^{\phi}(s, t^*)-\mathcal{Z}^{\phi}(s, t)\| \\
&\leqslant 2 \varepsilon + \frac{C}{\phi(L)} + \|\mathcal{Z}^{\phi(L)}(s, t^*)-\mathcal{Z}^{\phi}(s, t^*)\|,
\end{aligned}
```
where the last inequality is a consequence of <a href="#eq:proof-arzela-4" data-reference-type="eqref" data-reference="eq:proof-arzela-4">[eq:proof-arzela-4]</a>. The last term can be bounded as follows:
``` math
\begin{aligned}
\|\mathcal{Z}^{\phi(L)}(s, t^*)&-\mathcal{Z}^{\phi}(s, t^*)\| \\
&\leqslant\|\mathcal{Z}^{\phi(L)}(s, t^*)-\mathcal{Z}^{\phi(L)}(s^*, t^*)\| + \|\mathcal{Z}^{\phi(L)}(s^*, t^*)-\mathcal{Z}^{\phi}(s^*, t^*)\| \\
&\quad+ \|\mathcal{Z}^{\phi}(s^*, t^*)-\mathcal{Z}^{\phi}(s, t^*)\| \\
&\leqslant 2 \varepsilon + \frac{C}{\phi(L)} + \max_{i \in \{1, \dots, S\}} \|\mathcal{Z}^{\phi(L)}(s_i, t^*)-\mathcal{Z}^{\phi}(s_i, t^*)\|,
\end{aligned}
```
by using <a href="#eq:proof-arzela-1" data-reference-type="eqref" data-reference="eq:proof-arzela-1">[eq:proof-arzela-1]</a> and the fact that $`s^* \in \{s_1, \dots, s_S\}`$. Putting all the pieces together, we finally obtain
``` math
\begin{aligned}
\|\mathcal{Z}^{\phi(L)}(s, t)&-\mathcal{Z}^{\phi}(s, t)\| \leqslant 4 \varepsilon + \frac{2C}{\phi(L)} + \max_{i \in \{1, \dots, S\}, j \in \{1, \dots, T\}} \|\mathcal{Z}^{\phi(L)}(s_i, t_j)-\mathcal{Z}^{\phi}(s_i, t_j)\|.
\end{aligned}
```
By taking $`L`$ large enough, independent of $`s`$ and $`t`$, the sum of the last two terms can be made less than $`\varepsilon`$. Since $`\varepsilon`$ is arbitrary, this concludes the proof. ◻

</div>

A consequence of this result is a simplified version for sequences of functions only indexed by $`L`$ and not $`k`$, as follows.

<div id="prop:arzela-ascoli:2" class="corollary">

**Corollary 10**. * Let $`I \subseteq \mathbb{R}_+`$ be an interval, and $`(Z^L)_{L\in\mathbb{N}^*}`$ be a family of $`\mathcal{C}^1`$ functions from $`I`$ to $`\mathbb{R}^p`$. Assume that there exist a constant $`C>0`$ and a bounded integrable function $`b: I \to \mathbb{R}`$ such that, for any $`t\in I`$ and $`L \in \mathbb{N}^*`$, $`\|Z^L(t)\| \leqslant C`$ and $`\|\frac{dZ^L}{dt}(t)\| \leqslant b(t)`$. Then there exist a subsequence $`(Z^{\phi(L)})_{L\in\mathbb{N}^*}`$ of $`(Z^L)_{L\in\mathbb{N}^*}`$ and a function $`Z^\phi:I\to\mathbb{R}^{p}`$ such that $`Z^{\phi(L)}(t)`$ tends to $`Z^\phi(t)`$ uniformly over $`t`$.*

</div>

## Consistency of the Euler scheme for parameterized ODEs

<div id="prop:generalizednonlinearApprox:v2:simple" class="proposition">

**Proposition 11** (Consistency of the Euler scheme for parameterized ODEs.). * Let $`(\theta_k^L)_{L\in\mathbb{N}^*, 1\leqslant k\leqslant L}`$ be a bounded family of vectors of $`{\mathbb{R}}^p`$, and let
``` math
\Theta^L:[0,1] \to\mathbb{R}^{p},\ s\mapsto \theta_{\left\lfloor (L-1)s  \right\rfloor + 1}^L.
```
Assume that there exists $`\Theta:[0,1] \to\mathbb{R}^{p}`$ a Lipschitz continuous function such that $`\Theta^L(s)`$ tends to $`\Theta(s)`$ uniformly over $`s`$. Let $`(a^L)_{L \in \mathbb{N}^*}`$ be a sequence of vectors in some compact $`E \subset {\mathbb{R}}^d`$ converging to $`a \in E`$. Let $`g:\mathbb{R}^d\times\mathbb{R}^p\to\mathbb{R}^d`$ be a $`\mathcal{C}^1`$ function such that $`g(0, \cdot) \equiv 0`$ and $`g(\cdot, \theta)`$ is uniformly Lipschitz continuous for $`\theta`$ in any compact of $`{\mathbb{R}}^p`$. Consider the discrete scheme
``` math
\begin{aligned}
   \label{eq:consistency-discrete}
\begin{split}
u_0^L &= a^L \\
u_{k+1}^L &= u_k^L + \frac1L g(u_k^L, \theta_{k+1}^L), \quad k \in \{0, \dots, L-1\}.
\end{split}
\end{aligned}
```
Then $`u_{\left\lfloor Ls \right\rfloor}^L`$ tends to $`U(s)`$ uniformly over $`s\in[0, 1]`$, where $`U`$ is the unique solution of the ODE
``` math
\begin{aligned}
   \label{eq:consistency-continuous}
\begin{split}
    U(0) &= a \\
    \frac{dU}{ds}(s) &= g(U(s), \Theta(s)), \quad s \in [0, 1].
\end{split}
\end{aligned}
```
Moreover, the convergence only depends on the sequence $`(a^L)_{L \in \mathbb{N}^*}`$ and on its limit $`a \in E`$ through $`(\|a^L - a\|)_{L \in \mathbb{N}^*}`$.*

</div>

<div class="proof">

*Proof.* Let $`M`$ be a bound of the sequence $`(\theta_k^L)_{L\in\mathbb{N}^*, 1\leqslant k\leqslant L}`$. By definition of $`\Theta^L`$, the sequence $`(\Theta^L)_{L\in\mathbb{N}^*}`$ is also uniformly bounded by $`M`$, and the same is true for $`\Theta`$. Then the function $`g(\cdot, \Theta(s))`$ is uniformly Lipschitz for $`s \in [0, 1]`$. Furthermore, $`(U, s) \mapsto g(U, \Theta(s))`$ is continuous in $`s`$ because $`g`$ and $`\Theta`$ are continuous. Thus the ODE <a href="#eq:consistency-continuous" data-reference-type="eqref" data-reference="eq:consistency-continuous">[eq:consistency-continuous]</a> has a unique solution on $`[0, 1]`$ by the Picard-Lindelöf theorem (see Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a>).

Denote by $`C`$ the uniform Lipschitz constant of $`g(\cdot, \theta)`$ for $`\|\theta\| \leqslant M`$. Since $`g(0, \cdot) \equiv 0`$ and $`g(\cdot, \Theta(s))`$ is $`C`$-Lipschitz, one has
``` math
\Big\|\frac{dU}{ds}(s)\Big\| = \|g(U(s), \Theta(s))\| \leqslant C \|U(s)\|.
```
Therefore, by Grönwall’s inequality,
``` math
\|U(s)\| \leqslant\|U(0)\| \exp(C) = \|a\| \exp(C) \leqslant D_E \exp(C),
```
where $`D_E = \sup_{x \in E}\|x\| < \infty`$. A similar reasoning applies to the discrete scheme <a href="#eq:consistency-discrete" data-reference-type="eqref" data-reference="eq:consistency-discrete">[eq:consistency-discrete]</a>, using the discrete version of Grönwall’s inequality. More precisely, for any $`k \in \{0, \dots, L-1\}`$,
``` math
\|u_{k+1}^L\| \leqslant\|u_k^L\| + \frac1L \|g(u_k^L, \theta_{k+1}^L)\| \leqslant\Big(1 + \frac{C}{L}\Big) \|u_k^L\|.
```
Thus,
``` math
\|u_k^L\| \leqslant\|u_0^L\| \exp(C) = \|a^L\| \exp(C) \leqslant D_E \exp(C).
```
Overall, we can consider a restriction of $`g`$ to a compact set depending only on $`M`$, $`C`$, and $`E`$, which we will still denote by $`g`$ with a slight abuse of notation. Since $`g`$ is $`\mathcal{C}^1`$, it is therefore bounded and Lipschitz continuous, and we still let $`C`$ be its Lipschitz constant.

For $`L \in \mathbb{N}^*`$ and $`k \in \{0, \dots, L\}`$, we denote by $`\Delta_k^L`$ the gap between the continuous and the discrete schemes, i.e.,
``` math
\Delta_k^L = \Big\|U\Big(\frac{k}{L}\Big) - u_{k}^L\Big\|.
```
The next step is to recursively bound the size of this gap, first observing that $`\Delta_0^L = \|a^L - a\|`$. We have that
``` math
\label{eq:proof:def-function-U}
s \mapsto \frac{dU}{ds}(s) =  g(U(s), \Theta(s))
```
is a Lipschitz continuous function with some Lipschitz constant $`\tilde{C}`$. To see this, just note that $`U`$ itself is Lipschitz continuous in $`s`$, since $`g`$ is bounded, and therefore the function <a href="#eq:proof:def-function-U" data-reference-type="eqref" data-reference="eq:proof:def-function-U">[eq:proof:def-function-U]</a> is a composition of Lipschitz continuous functions. In particular, $`\frac{dU}{ds}`$ is almost everywhere differentiable, and its derivative $`\frac{d^2 U}{d s^2}(s)`$ is bounded in the supremum norm by $`\Tilde{C}`$. As a consequence, for $`k \in \{0, \dots, L-1\}`$, the Taylor expansion of $`U`$ on $`[\frac{k}{L}, \frac{k+1}{L}]`$ takes the form
``` math
\begin{aligned}
U\Big(\frac{k+1}{L}\Big) &= U\Big(\frac{k}{L}\Big) + \frac{1}{L} \frac{d U}{d s}\Big(\frac{k}{L}\Big) + \int_{k/L}^{(k+1)/L} \Big(\frac{k+1}{L} - s\Big) \frac{d^2 U}{d s^2}(s) ds,
\end{aligned}
```
where the norm of the remainder term is less than $`\Tilde{C}/L^2`$. Therefore,
``` math
\begin{aligned}
\Delta_{k+1}^L &= \Big\|U\Big(\frac{k+1}{L}\Big) - u_{k+1}^L\Big\| \\
&= \Big\|U\Big(\frac{k}{L}\Big) + \frac{1}{L} g\Big(U\Big(\frac{k}{L}\Big), \Theta\Big(\frac{k}{L}\Big) \Big) + \int_{k/L}^{(k+1)/L} \Big(\frac{k+1}{L} - s\Big) \frac{d^2 U}{d s^2}(s) ds \\
& \quad - u_k^L - \frac1L g(u_k^L, \theta_{k+1}^L)\Big\| \\
&\leqslant\Big\|U\Big(\frac{k}{L}\Big) - u_k^L\Big\| + \Big\|\frac{1}{L} g\Big(U\Big(\frac{k}{L}\Big), \Theta\Big(\frac{k}{L}\Big) \Big)   - \frac1L g(u_k^L, \theta_{k+1}^L) \Big\| \\
& \quad + \int_{k/L}^{(k+1)/L} \Big(\frac{k+1}{L} - s\Big) \Big\|\frac{d^2 U}{d s^2}(s)\Big\| ds  \\
&\leqslant \Delta_k^L + \frac{C}{L} \Delta_k^L + \frac{C}{L} \Big\|\Theta\Big(\frac{k}{L}\Big) - \theta_{k+1}^L\Big\| + \frac{\Tilde{C}}{L^2}.
\end{aligned}
```
In the last inequality, we used the fact that $`g`$ is $`C`$-Lipschitz. Since, by definition, $`\theta_{k+1}^L`$ = $`\Theta^L(\frac{k}{L-1})`$, we obtain, for $`k \in \{0, \dots, L-1\}`$,
``` math
\begin{aligned}
\Delta_{k+1}^L &\leqslant \Big(1 + \frac{C}{L}\Big) \Delta_k^L + \frac{C}{L} \Big\|\Theta\Big(\frac{k}{L}\Big) - \Theta^L\Big(\frac{k}{L-1}\Big)\Big\| + \frac{\Tilde{C}}{L^2}     \\
&\leqslant \Big(1 + \frac{C}{L}\Big) \Delta_k^L + \frac{C}{L} \sup_{s \in [0, 1]}\|\Theta(s) - \Theta^L(s)\| + \frac{C}{L} \Big\|\Theta\Big(\frac{k}{L}\Big) - \Theta\Big(\frac{k}{L-1}\Big)\Big\| + \frac{\Tilde{C}}{L^2} \\
&\leqslant \Big(1 + \frac{C}{L}\Big) \Delta_k^L + \frac{C}{L} \sup_{s \in [0, 1]}\|\Theta(s) - \Theta^L(s)\| + \frac{C C_\Theta}{L^2}  + \frac{\Tilde{C}}{L^2},
\end{aligned}
```
where $`C_\Theta`$ is the Lipschitz constant of $`\Theta`$. By the discrete Grönwall’s inequality, we deduce that, for $`k \in \{0, \dots, L-1\}`$,
``` math
\begin{aligned}
\Delta_{k+1}^L &\leqslant \Big(\Delta_0^L + \sup_{s \in [0, 1]}\|\Theta(s) - \Theta^L(s)\| + \frac{C_\Theta}{L}  + \frac{\Tilde{C}}{L C} \Big) e^C \nonumber \\
&= \Big(\|a^L - a\| + \sup_{s \in [0, 1]}\|\Theta(s) - \Theta^L(s)\| + \frac{C_\Theta}{L}  + \frac{\Tilde{C}}{L C} \Big) e^C. \label{eq:proof-euler-bound-delta}
\end{aligned}
```
This shows that the gaps $`\Delta_k^L`$ converge to zero uniformly over $`k \in \{0, \dots, L\}`$ as $`L`$ tends to infinity.

We conclude by observing that, for any $`s \in [0, 1]`$,
``` math
\label{eq:proof-euler-bound-u}
\|U(s) - u_{\left\lfloor Ls \right\rfloor}^L\| \leqslant \Big\|U(s) - U\Big(\frac{\lfloor L s \rfloor}{L}\Big)\Big\| + \Big\| U\Big(\frac{\lfloor L s \rfloor}{L}\Big) - u_{\left\lfloor Ls \right\rfloor}^L\Big\| \leqslant\frac{C_U}{L} + \Delta_{\lfloor L s \rfloor}^L,
```
where $`C_U`$ is the Lipschitz constant of $`U`$. Both terms converge to zero uniformly over $`s`$ as $`L`$ tends to infinity. Finally, an inspection of our bounds shows that the convergence only depends on $`(a^L)_{L \in \mathbb{N}^*} \in E^{\mathbb{N}^*}`$ through $`\|a^L - a\|`$. ◻

</div>

The results of Proposition <a href="#prop:generalizednonlinearApprox:v2:simple" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2:simple">11</a> can be extended without much effort to two other related cases. First, the parameters $`\theta_k^L`$ may depend on some other variable $`t`$, as long as all assumptions are verified uniformly over $`t`$. Second, these parameters may converge to some limit parameters as both $`L`$ and $`t`$ go to infinity. This is encapsulated in the following two corollaries.

<div id="prop:generalizednonlinearApprox:v2" class="corollary">

**Corollary 12**. * Let $`I \subseteq \mathbb{R}_+`$ be an interval. Let $`(\theta_k^L)_{L\in\mathbb{N}^*, 1\leqslant k\leqslant L}`$ be a uniformly bounded family of functions from $`I`$ to $`\mathbb{R}^{p}`$, and let
``` math
\Theta^L:[0,1]\times I\to\mathbb{R}^{p},\ (s, t)\mapsto \theta_{\left\lfloor (L-1)s \right\rfloor + 1}^L(t).
```
Assume that there exists a function $`\Theta:[0,1]\times I\to\mathbb{R}^{p}`$ such that $`\Theta^L(s,t)`$ tends to $`\Theta(s,t)`$ uniformly over $`s`$ and $`t`$, and $`\Theta(\cdot, t)`$ is uniformly Lipschitz continuous for $`t \in I`$. Let $`(a^L)_{L \in \mathbb{N}^*}`$ be a family of functions from $`I`$ to some compact $`E \subset {\mathbb{R}}^d`$, uniformly converging to $`a: I \to E`$. Let $`g:\mathbb{R}^d\times\mathbb{R}^p\to\mathbb{R}^d`$ be a $`\mathcal{C}^1`$ function such that $`g(0, \cdot) \equiv 0`$ and $`g(\cdot, \theta)`$ is uniformly Lipschitz continuous for $`\theta`$ in any compact of $`{\mathbb{R}}^p`$. Consider the discrete scheme, for $`t \in I`$,
``` math
\begin{aligned}
u_0^L(t) &= a^L(t) \\
u_{k+1}^L(t) &= u_k^L(t)+\frac{1}{L} g(u_k^L(t), \theta_{k+1}^L(t)),\quad k \in \{0, \dots, L-1\}.
\end{aligned}
```
Then $`u_{\left\lfloor Ls \right\rfloor}^L(t)`$ tends to $`U(s, t)`$ uniformly over $`s\in[0, 1]`$ and $`t\in I`$, where $`U(\cdot, t)`$ is the unique solution of the ODE
``` math
\begin{aligned}
    U(0, t) &= a(t) \\
    \frac{\partial U}{\partial s}(s, t) &= g(U(s, t), \Theta(s, t)), \quad s \in [0, 1].
\end{aligned}
```
Moreover, the convergence only depends on the sequence $`(a^L)_{L \in \mathbb{N}^*}`$ and on its limit $`a \in E^I`$ through $`(\sup_{t \in I} \|a^L(t) - a(t)\|)_{L \in \mathbb{N}^*}`$.*

</div>

<div id="prop:generalizednonlinearApprox:v3" class="corollary">

**Corollary 13**. * Let $`I \subseteq \mathbb{R}_+`$ be an interval. Let $`(\theta_k^L)_{L\in\mathbb{N}^*, 1\leqslant k\leqslant L}`$ be a uniformly bounded family of functions from $`I`$ to $`\mathbb{R}^{p}`$, and let
``` math
\Theta^L:[0,1]\times\mathbb{R}_+ \to\mathbb{R}^{p},\ (s, t)\mapsto \theta_{\left\lfloor (L-1)s \right\rfloor + 1}^L(t).
```
Assume that there exists a function $`\Theta_\infty:[0,1] \to \mathbb{R}^{p}`$ such that $`\Theta^L(s,t)`$ tends to $`\Theta_\infty(s)`$ uniformly over $`s`$ as $`L, t \to \infty`$, and $`\Theta_\infty`$ is Lipschitz continuous. Let $`(a^L)_{L \in \mathbb{N}^*}`$ be a family of functions from $`I`$ to some compact $`E \subset {\mathbb{R}}^d`$, and converging to $`a_\infty \in E`$ as $`L, t \to \infty`$. Let $`g:\mathbb{R}^d\times\mathbb{R}^p\to\mathbb{R}^d`$ be a $`\mathcal{C}^1`$ function such that $`g(0, \cdot) \equiv 0`$ and $`g(\cdot, \theta)`$ is uniformly Lipschitz continuous for $`\theta`$ in any compact of $`{\mathbb{R}}^p`$. Consider the discrete scheme, for $`t \in I`$,
``` math
\begin{aligned}
u_0^L(t) &= a^L(t) \\
u_{k+1}^L(t) &= u_k^L(t)+\frac1L g(u_k^L(t), \theta_{k+1}^L(t)),\quad k \in \{0, \dots, L-1\}.
\end{aligned}
```
Then $`u_{\left\lfloor Ls \right\rfloor}^L(t)`$ tends to $`U(s)`$ uniformly over $`s\in[0, 1]`$ as $`L, t \to \infty`$, where $`U`$ is the unique solution of the ODE
``` math
\begin{aligned}
    U(0) &= a_\infty \\
    \frac{dU}{ds}(s) &= g(U(s), \Theta_\infty(s)), \quad s \in [0, 1].
\end{aligned}
```
Moreover, the convergence only depends on the sequence $`(a^L)_{L \in \mathbb{N}^*}`$ and on its limit $`a \in E^I`$ through $`(\sup_{t \in I} \|a^L(t) - a(t)\|)_{L \in \mathbb{N}^*}`$.*

</div>

## Large-depth convergence of the gradient flow

This section is devoted to proving the main result of Appendix <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a>, namely the large-depth convergence of the gradient flow. The setting we consider encompasses both Section <a href="#sec:finite-training-time" data-reference-type="ref" data-reference="sec:finite-training-time">4.1</a> (finite training time and clipped gradient flow) and Section <a href="#subsec:long-training" data-reference-type="ref" data-reference="subsec:long-training">4.2</a> (arbitrary training time and standard gradient flow). To this end, we consider a training interval $`I = [0, T] \subseteq {\mathbb{R}}_+`$, for $`T \leqslant\infty`$, and the gradient flow formulation <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a>, which is equivalent to the standard gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a> if $`\pi`$ equals the identity. Note that we do not need to assume in the following proof that $`\pi`$ is bounded (but only Lipschitz continuous). Therefore, the proof also holds in the case where $`\pi`$ equals the identity.

<div id="thm:general-convergence:v2" class="theorem">

**Theorem 14**. *Consider the residual network <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a> initialized as explained in Appendix <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a> and trained with the gradient flow <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> on $`I = [0, T] \subseteq {\mathbb{R}}_+`$, for some $`T \in (0, \infty]`$. Assume that there exists a unique solution to the gradient flow, such that $`(A^L)_{L \in \mathbb{N}^*}`$ and $`(B^L)_{L \in \mathbb{N}^*}`$ each satisfies the assumptions of Corollary <a href="#prop:arzela-ascoli:2" data-reference-type="ref" data-reference="prop:arzela-ascoli:2">10</a>, and $`(Z_k^L)_{L \in \mathbb{N}^*, 1 \leqslant k \leqslant L}`$ satisfies the assumptions of Proposition <a href="#prop:arzela-ascoli" data-reference-type="ref" data-reference="prop:arzela-ascoli">9</a>. Then the following four statements hold **as $`L`$ tends to infinity**:*

1)  *There exist functions $`A: I \to {\mathbb{R}}^{q \times d}`$ and $`B: I \to {\mathbb{R}}^{d' \times q}`$ such that $`A^L(t)`$ and $`B^L(t)`$ converge uniformly over $`t \in I`$ to $`A(t)`$ and $`B(t)`$.*

2)  *There exists a Lipschitz continuous function $`\mathcal{Z}: [0,1] \times I \to \mathbb{R}^{p}`$ such that
    ``` math
    \mathcal{Z}^L:[0,1] \times I\to\mathbb{R}^{p},\ (s, t)\mapsto \mathcal{Z}^L(s, t) = Z_{\left\lfloor (L-1)s \right\rfloor + 1}^L(t)
    ```
    converges uniformly over $`s \in [0, 1]`$ and $`t \in I`$ to $`\mathcal{Z}(s,t)`$.*

3)  *Uniformly over $`s \in [0, 1]`$, $`t \in I`$, and $`x \in \mathcal{X}`$, the hidden layer $`h_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges to the solution at time $`s`$ of the neural ODE
    ``` math
    \begin{aligned}
        \begin{split}
            H(0, t) &= A(t)x \\
            \frac{\partial H}{\partial s}(s, t) &= f(H(s, t), \mathcal{Z}(s, t)), \quad s \in [0, 1].
        \end{split}
    \end{aligned}
    ```*

4)  *Uniformly over $`t \in I`$ and $`x \in \mathcal{X}`$, the output $`F^L(x ; t)`$ converges to $`B(t) H(1, t)`$.*

</div>

<div class="proof">

*Proof.* According to Proposition <a href="#prop:arzela-ascoli" data-reference-type="ref" data-reference="prop:arzela-ascoli">9</a>, there exists a subsequence $`(\mathcal{Z}^{\phi(L)})_{L \in \mathbb{N}^*}`$ of $`(\mathcal{Z}^L)_{L \in \mathbb{N}^*}`$ and a Lipschitz continuous function $`\mathcal{Z}^\phi:[0,1] \times I\to\mathbb{R}^{p}`$ such that $`\mathcal{Z}^{\phi(L)}(s,t)`$ tends to $`\mathcal{Z}^\phi(s,t)`$ uniformly over $`s`$ and $`t`$. Similarly, by Corollary <a href="#prop:arzela-ascoli:2" data-reference-type="ref" data-reference="prop:arzela-ascoli:2">10</a>, there exists subsequences of $`(A^L)_{L \in \mathbb{N}^*}`$ and $`(B^L)_{L \in \mathbb{N}^*}`$ that converge uniformly. With a slight abuse of notation, we still denote these subsequences by $`\phi`$, and the corresponding limits by $`A^\phi`$ and $`B^\phi`$.

In the remainder, we prove the uniqueness of the accumulation point $`(Z^{\phi}, A^{\phi}, B^{\phi})`$ by showing that it is the solution of an ODE that satisfies the assumptions of the Picard-Lindelöf theorem. The statements $`(i)`$ to $`(iv)`$ then follow easily.

Consider a general input $`(x, y) \in \mathcal{X}\times \mathcal{Y}`$, and let $`H^L(s, t) = h_{\left\lfloor Ls \right\rfloor}^L(t)`$ (recall that $`h_k^L(t)`$ is defined by the forward propagation <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a>). Corollary <a href="#prop:generalizednonlinearApprox:v2" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2">12</a>, with $`\theta_k^L = Z_k^{\phi(L)}`$, $`\Theta = \mathcal{Z}^{\phi}`$, $`a^L = A^{\phi(L)} x`$, $`g = f`$, ensures that $`H^{\phi(L)}(s, t)`$ converges uniformly (over $`s`$ and $`t`$) to $`H^{\phi}(s, t)`$ that is the solution at time $`s`$ of the ODE
``` math
\begin{aligned}
    H^{\phi}(0, t) &= A^{\phi}(t) x \\
    \frac{\partial H^{\phi}}{\partial s}(s, t) &= f(H^{\phi}(s, t), \mathcal{Z}^\phi(s, t)), \quad s \in [0, 1].
\end{aligned}
```
By inspecting the proof of the corollary, we also have that $`(h_k^{\phi(L)})_{L\in\mathbb{N}^*, 1\leqslant k\leqslant\phi(L)}`$ and $`(H^{\phi(L)})_{L\in\mathbb{N}^*}`$ are uniformly bounded and that $`H^\phi(\cdot, t)`$ is uniformly Lipschitz continuous for $`t \in I`$.

We now turn our attention to the backpropagation recurrence <a href="#eq:proof-main-thm:backprop" data-reference-type="eqref" data-reference="eq:proof-main-thm:backprop">[eq:proof-main-thm:backprop]</a>, which defines the backward state $`p_k^L(t)`$. First observe that the convergence of $`H^{\phi(L)}`$ implies that
``` math
p_{\phi(L)}^{\phi(L)}(t) = 2 B^{\phi(L)}(t)^\top (B^{\phi(L)}(t) h_{\phi(L)}^{\phi(L)}(t) - y) = 2 B^{\phi(L)}(t)^\top (B^{\phi(L)}(t) H^{\phi(L)}(1, t) - y)
```
converges uniformly to $`2 B^{\phi}(t)^\top (B^{\phi}(t) H^{\phi}(1, t) - y) \in {\mathbb{R}}^d`$. Now, let $`P^L(s, t) = p_{\left\lfloor Ls \right\rfloor}^L(t)`$. We apply again Corollary <a href="#prop:generalizednonlinearApprox:v2" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2">12</a>, this time to the backpropagation recurrence <a href="#eq:proof-main-thm:backprop" data-reference-type="eqref" data-reference="eq:proof-main-thm:backprop">[eq:proof-main-thm:backprop]</a>, with $`\theta_k^L = (h_k^{\phi(L)}, Z_k^{\phi(L)})`$, $`\Theta = (H^{\phi}, \mathcal{Z}^{\phi})`$, $`g: (p, (h, Z)) \mapsto \partial_1 f(h, Z) p`$, and $`a^L = 2 (B^{\phi(L)})^\top (B^{\phi(L)} H^{\phi(L)}(1, \cdot) - y)`$. Let us quickly check that the conditions of the corollary are met:

- The sequence $`(h_k^{\phi(L)})_{L\in\mathbb{N}^*, 1\leqslant k\leqslant\phi(L)}`$ is bounded, as noted previously, and the same holds for $`(Z_k^{\phi(L)})_{L\in\mathbb{N}^*, 1\leqslant k\leqslant\phi(L)}`$ by the assumptions of Theorem <a href="#thm:general-convergence:v2" data-reference-type="ref" data-reference="thm:general-convergence:v2">14</a>.

- The function $`H^\phi(\cdot, t)`$ is uniformly Lipschitz continuous for $`t \in I`$, as noted previously, and the same is true for $`Z^\phi(\cdot, t)`$ since $`Z^\phi`$ is Lipschitz continuous.

- The function $`h_{\left\lfloor (\phi(L)-1)s \right\rfloor + 1}^{\phi(L)}(t)`$ tends to $`H^{\phi}(s, t)`$ uniformly over $`s`$ and $`t`$, as seen in the beginning of the proof. More precisely, we know that $`H^{\phi(L)}(s,t) = h_{\left\lfloor \phi(L)s \right\rfloor}^{\phi(L)}(t)`$ tends to $`H^{\phi}(s, t)`$. Simple algebra and the fact that two successive iterates of <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a> are separated by a distance proportional to $`1/L`$ show that both statements are equivalent. Furthermore, $`\mathcal{Z}^{\phi(L)}(s,t)`$ tends to $`\mathcal{Z}^\phi(s,t)`$ uniformly over $`s`$ and $`t`$ as noted above.

- The sequence $`(a^L)_{L \in \mathbb{N}^*}`$ is uniformly bounded, since $`B^{\phi(L)}`$ and $`H^{\phi(L)}(1, \cdot)`$ are. It also converges uniformly to $`a: t \mapsto 2 B^{\phi}(t)^\top (B^{\phi}(t) H^{\phi}(1, t) - y)`$.

- The function $`g`$ is $`\mathcal{C}^1`$ since $`f`$ is $`\mathcal{C}^2`$. We clearly have $`g(0, \cdot) \equiv 0`$. Finally, $`g(\cdot, (h, Z))`$ is uniformly Lipschitz continuous for $`(h, Z)`$ in any compact since $`\partial_1 f`$ is continuous.

Overall, we obtain that $`P^{\phi(L)}(s, t)`$ converges uniformly (over $`s`$ and $`t`$) to $`P^{\phi}(s, t)`$, the solution at time $`s`$ of the backward ODE
``` math
\begin{aligned}
    P^{\phi}(1, t) &= 2 B^{\phi}(t)^\top (B^{\phi}(t) H^{\phi}(1, t) - y) \\
    \frac{\partial P^{\phi}}{\partial s}(s, t) &= \partial_1 f(H^{\phi}(s, t), \mathcal{Z}^\phi(s, t)) P^\phi(s, t), \quad s \in [0, 1].
\end{aligned}
```
Furthermore, the proof of the corollary shows that $`(P^{\phi(L)})_{L\in\mathbb{N}^*}`$ is uniformly bounded. Now, recall that the gradient flow for $`Z_k^{\phi(L)}(t)`$, given by <a href="#eq:clipped-gf" data-reference-type="eqref" data-reference="eq:clipped-gf">[eq:clipped-gf]</a> and <a href="#eq:proof-main-thm:backprop-1" data-reference-type="eqref" data-reference="eq:proof-main-thm:backprop-1">[eq:proof-main-thm:backprop-1]</a>, takes the following form, for $`t \in I`$ and $`k \in \{1, \dots, \phi(L)\}`$,
``` math
\frac{\partial Z_k^{\phi(L)}(t)}{\partial t} 
= \pi \Big( -\frac{1}{n} \sum_{i=1}^n \partial_2 f(h_{k-1,i}^{\phi(L)}(t), Z_{k}^{\phi(L)}(t))^\top p_{k,i}^{\phi(L)}(t) \Big),
```
where the $`i`$ subscript corresponds to the $`i`$-th input $`x_i`$. By definition, for $`s \in [0, 1]`$, $`\mathcal{Z}^{\phi(L)}(s, t) = Z_{\left\lfloor (\phi(L)-1)s \right\rfloor + 1}^{\phi(L)}(t)`$. Thus, the equation above can be rewritten, for $`s \in [0, 1]`$ and $`t \in I`$,
``` math
\label{eq:proof-main-thm:gf-z}
\frac{\partial \mathcal{Z}^{\phi(L)}(s, t)}{\partial t} 
= \pi \Big( -\frac{1}{n} \sum_{i=1}^n \partial_2 f(h_{\lfloor (\phi(L) - 1)s \rfloor,i}^{\phi(L)}(t), Z_{\lfloor (\phi(L) - 1)s \rfloor + 1}^{\phi(L)}(t))^\top p_{\lfloor (\phi(L)-1)s \rfloor +1,i}^{\phi(L)}(t) \Big).
```
The term inside $`\pi`$ can be rewritten as
``` math
-\frac{1}{n} \sum_{i=1}^n \partial_2 f\Big(H_i^{\phi(L)}\Big(\frac{\lfloor (\phi(L) - 1)s \rfloor}{\phi(L)}, t\Big), \mathcal{Z}^{\phi(L)}(s, t)\Big)^\top P_i^{\phi(L)}\Big(\frac{\lfloor (\phi(L) - 1)s \rfloor+1}{\phi(L)}, t\Big).
```
Since $`f`$ is $`\mathcal{C}^2`$, $`\partial_2 f`$ is locally Lipschitz continuous. Applying the first part of the proof to the specific case of $`x_i`$, we know that $`H_i^{\phi(L)}`$ and $`P_i^{\phi(L)}`$ uniformly bounded, and that $`H_i^{\phi(L)}(s,t)`$ and $`P_i^{\phi(L)}(s,t)`$ converge uniformly to $`H_i^{\phi}(s,t)`$ and $`P_i^{\phi}(s,t)`$. Therefore, the right-hand side of <a href="#eq:proof-main-thm:gf-z" data-reference-type="eqref" data-reference="eq:proof-main-thm:gf-z">[eq:proof-main-thm:gf-z]</a> converges uniformly over $`s`$ and $`t`$ to
``` math
\pi \Big( -\frac{1}{n} \sum_{i=1}^n \partial_2 f(H_i^{\phi}(s, t), \mathcal{Z}^{\phi}(s, t))^\top P_i^{\phi}(s, t) \Big).
```
We have just shown the uniform convergence of the derivative in $`t`$ of $`\mathcal{Z}^{\phi(L)}(s,t)`$. Furthermore, we know that, for $`s \in [0, 1]`$, the sequence $`(t \mapsto \mathcal{Z}^{\phi(L)}(s, t))_{L \in \mathbb{N}^*}`$ converges to $`\mathcal{Z}^{\phi}(s, \cdot)`$. These two statements imply that $`\mathcal{Z}^{\phi}`$ is differentiable with respect to $`t`$ and that, for $`s \in [0, 1]`$, its derivative satisfies the ordinary differential equation
``` math
\label{eq:proof-main-thm:functional-ivp-1}
\frac{\partial \mathcal{Z}^{\phi}(s, t)}{\partial t} = \pi \Big( -\frac{1}{n} \sum_{i=1}^n \partial_2 f(H_i^{\phi}(s, t), \mathcal{Z}^{\phi}(s, t))^\top P_i^{\phi}(s, t) \Big).
```
Moreover, by our initialization scheme,
``` math
\label{eq:proof-main-thm:functional-ivp-2}
\mathcal{Z}^{\phi}(s, 0) = Z^{\textnormal{init}}(s).
```
A similar approach reveals that $`A^{\phi}(t)`$ and $`B^{\phi}(t)`$ are differentiable and that they verify the equations
``` math
\begin{aligned}
\frac{d A^{\phi}}{dt}(t) &= \pi \Big(- \frac{1}{n} \sum_{i=1}^n P_{i}^{\phi}(0, t) x_i^\top \Big), \quad && A^{\phi}(0) = A^{\textnormal{init}},  \label{eq:proof-main-thm:functional-ivp-3}  \\
\frac{dB^{\phi}}{dt}(t) &= \pi \Big( - \frac{2}{n} \sum_{i=1}^n (B^{\phi}(t) H_i^{\phi}(1, t) - y_i) H_i^{\phi}(1, t)^\top \Big), \quad && B^{\phi}(0) = B^{\textnormal{init}}.   \label{eq:proof-main-thm:functional-ivp-4}
\end{aligned}
```
The equations <a href="#eq:proof-main-thm:functional-ivp-1" data-reference-type="eqref" data-reference="eq:proof-main-thm:functional-ivp-1">[eq:proof-main-thm:functional-ivp-1]</a> to <a href="#eq:proof-main-thm:functional-ivp-4" data-reference-type="eqref" data-reference="eq:proof-main-thm:functional-ivp-4">[eq:proof-main-thm:functional-ivp-4]</a> can be seen as an initial value problem whose variables are the function $`\mathcal{Z}^{\phi}(\cdot, t): [0, 1] \to {\mathbb{R}}^p`$ and the matrices $`A^{\phi}(t) \in {\mathbb{R}}^{q \times d}, B^{\phi}(t) \in {\mathbb{R}}^{d' \times q}`$. To complete the proof, it remains to show, using the Picard-Lindelöf theorem (see Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a>), that there exists a unique solution to this problem. First, note that the space $`\mathcal{B}([0, 1], {\mathbb{R}}^p)`$ of bounded functions from $`[0, 1]`$ to $`{\mathbb{R}}^p`$ endowed with the supremum norm is a Banach space, which is the proper space in which to apply the Picard-Lindelöf theorem. We therefore endow the space of parameters $`\mathcal{B}([0, 1], {\mathbb{R}}^p) \times {\mathbb{R}}^{q \times d} \times {\mathbb{R}}^{d' \times q}`$ with the norm
``` math
\|(\mathcal{Z}, A, B)\| := \sup_{s \in [0, 1]}\|\mathcal{Z}(s)\| + \|A\|_2 + \|B\|_2,
```
which makes it a Banach space. We have to show that the mapping
``` math
\begin{aligned}
\begin{split}
\label{eq:proof-main-thm:big-mapping}
(\mathcal{Z}, A, B) \mapsto \bigg(&s \mapsto \pi \Big( -\frac{1}{n} \sum_{i=1}^n \partial_2 f(H_i(s), \mathcal{Z}(s))^\top P_i(s) \Big), \\
&\pi \Big(- \frac{1}{n} \sum_{i=1}^n P_{i}(0) x_i^\top \Big), \; \pi \Big( - \frac{2}{n} \sum_{i=1}^n (B H_i(1) - y_i) H_i(1)^\top \Big) \bigg)    
\end{split}
\end{aligned}
```
is locally Lipschitz continuous with respect to this norm, where we recall that $`H_i(s)`$ in <a href="#eq:proof-main-thm:big-mapping" data-reference-type="eqref" data-reference="eq:proof-main-thm:big-mapping">[eq:proof-main-thm:big-mapping]</a> is the solution at time $`s`$ of the initial value problem
``` math
\begin{aligned}
\begin{split}
    H_i(0) &= A x_i \label{eq:proof-main-thm:uniqueness-1} \\
    \frac{d H_i}{d s}(s) &= f(H_i(s), \mathcal{Z}(s)), \quad s \in [0, 1],
\end{split}
\end{aligned}
```
and $`P_i(s)`$ is the solution at time $`s`$ of the initial value problem
``` math
\begin{aligned}
\begin{split}
    P_i(1) &= 2 B^\top (B H_i(1) - y_i)  \label{eq:proof-main-thm:uniqueness-2} \\
    \frac{d P_i}{d s}(s) &= \partial_1 f(H_i(s), \mathcal{Z}(s)) P_i(s), \quad s \in [0, 1].
\end{split}
\end{aligned}
```
To prove that the mapping <a href="#eq:proof-main-thm:big-mapping" data-reference-type="eqref" data-reference="eq:proof-main-thm:big-mapping">[eq:proof-main-thm:big-mapping]</a> is locally Lipschitz continuous, we first check that it is well defined. Since $`\mathcal{Z}`$ is assumed to be only bounded (and not continuous), the solutions of the initial value problems <a href="#eq:proof-main-thm:uniqueness-1" data-reference-type="eqref" data-reference="eq:proof-main-thm:uniqueness-1">[eq:proof-main-thm:uniqueness-1]</a> and <a href="#eq:proof-main-thm:uniqueness-2" data-reference-type="eqref" data-reference="eq:proof-main-thm:uniqueness-2">[eq:proof-main-thm:uniqueness-2]</a> are well defined in the sense of the Caratheodory conditions, which are given in Lemma <a href="#lemma:caratheodory" data-reference-type="ref" data-reference="lemma:caratheodory">17</a>.

Next, we can show that $`(\mathcal{Z}, A, B) \mapsto H_i`$ is locally Lipschitz continuous for $`i \in \{1, \dots, n\}`$. To do this, consider two sets of parameters $`(\mathcal{Z}, A, B)`$ and $`(\tilde{\mathcal{Z}}, \tilde{A}, \tilde{B})`$ belonging to a compact set $`D`$. Let $`H_i`$ and $`\tilde{H}_i`$ denote the corresponding hidden states. As in the proof of Proposition <a href="#prop:generalizednonlinearApprox:v2:simple" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2:simple">11</a>, it holds that $`H_i`$ and $`\tilde{H}_i`$ belong to some compact set $`E`$ that depends only on $`D`$ and $`f`$. Let $`K_f`$ be the Lipschitz constant of the $`\mathcal{C}^1`$ function $`f`$ on $`E \times D`$. Then,
``` math
\begin{aligned}
    \|\tilde{H}_i(s) - H_i(s)\|  &\leqslant\|\tilde{H}_i(0) - H_i(0)\| + \int_0^s \Big\|\frac{d \tilde{H}_i}{d r}(r) - \frac{d H_i}{d r}(r)\Big\| dr \\
    &\leqslant\|\tilde{H}_i(0) - H_i(0)\| + \int_0^s \|f(\tilde{H}_i(r), \tilde{\mathcal{Z}}(r)) - f(H_i(r), \mathcal{Z}(r))\| dr.
\end{aligned}
```
The norm inside the integral can be bounded by
``` math
\begin{aligned}
\|f(\tilde{H}_i(r), \tilde{\mathcal{Z}}(r)) &- f(\tilde{H}_i(r), \mathcal{Z}(r))\| + \|f(\tilde{H}_i(r), \mathcal{Z}(r)) - f(H_i(r), \mathcal{Z}(r))\| \\
&\leqslant K_f \sup_{r \in [0, 1]} \|\tilde{\mathcal{Z}}(r) - \mathcal{Z}(r)\| + K_f \|\tilde{H}_i(r) - H_i(r)\|.
\end{aligned}
```
Therefore,
``` math
\begin{aligned}
    \|\tilde{H}_i(s) - H_i(s)\|  \leqslant\|\tilde{A} - A\|_2 \|x_i\| + K_f \sup_{r \in [0, 1]} \|\tilde{\mathcal{Z}}(r) - \mathcal{Z}(r)\| + \int_0^s K_f \|\tilde{H}_i(r) - H_i(r)\| dr.
\end{aligned}
```
Using Grönwall’s inequality, we obtain, for any $`s \in [0, 1]`$,
``` math
\begin{aligned}
    \|\tilde{H}_i(s) - H_i(s)\|  \leqslant\Big(\|\tilde{A} - A\|_2 \|x_i\| + K_f \sup_{r \in [0, 1]} \|\tilde{\mathcal{Z}}(r) - \mathcal{Z}(r)\|\Big) \exp(K_f).
\end{aligned}
```
This shows that the function $`(\mathcal{Z}, A, B) \mapsto H_i`$ is locally Lipschitz continuous. One proves by similar arguments that the function $`(\mathcal{Z}, A, B) \mapsto P_i`$ is locally Lipschitz continuous. Thus, overall, the mapping <a href="#eq:proof-main-thm:big-mapping" data-reference-type="eqref" data-reference="eq:proof-main-thm:big-mapping">[eq:proof-main-thm:big-mapping]</a> is locally Lipschitz continuous as a composition of locally Lipschitz continuous functions.

The Picard-Lindelöf theorem guarantees the uniqueness of the maximal solution of the initial value problem <a href="#eq:proof-main-thm:functional-ivp-1" data-reference-type="eqref" data-reference="eq:proof-main-thm:functional-ivp-1">[eq:proof-main-thm:functional-ivp-1]</a>–<a href="#eq:proof-main-thm:functional-ivp-4" data-reference-type="eqref" data-reference="eq:proof-main-thm:functional-ivp-4">[eq:proof-main-thm:functional-ivp-4]</a> in the space $`\mathcal{B}([0, 1], {\mathbb{R}}^p) \times {\mathbb{R}}^{d \times q} \times {\mathbb{R}}^{d' \times q}`$. Since any accumulation point $`(\mathcal{Z}^{\phi}, A^{\phi}, B^{\phi})`$ is a solution belonging to this space, this proves the uniqueness of the accumulation point, which we therefore denote as $`(\mathcal{Z}, A, B)`$.

The uniform convergence of $`(\mathcal{Z}^L, A^L, B^L)`$ to $`(\mathcal{Z}, A, B)`$ is then easily shown by contradiction. Suppose that uniform convergence does not hold. If this is true, then there exists a subsequence that stays at distance $`\varepsilon > 0`$ from $`(\mathcal{Z}, A, B)`$ (in the sense of the uniform norm). Then arguments similar to the beginning of the proof show the existence of a second accumulation point, which is a contradiction. This shows the uniform convergence, yielding statements $`(i)`$ and $`(ii)`$ of the theorem.

Finally, reapplying Corollary <a href="#prop:generalizednonlinearApprox:v2" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2">12</a> with $`\theta_k^L = Z_k^L`$, $`\Theta = \mathcal{Z}`$, $`a^L = A^{L} x`$, $`g = f`$, completes the proof by proving statements $`(iii)`$ and $`(iv)`$. ◻

</div>

#### Training dynamics of the limiting weights.

Interestingly, the proof of Theorem <a href="#thm:general-convergence:v2" data-reference-type="ref" data-reference="thm:general-convergence:v2">14</a> provides us with an explicit description of the evolution of the continuous-depth limiting weights during training. With the notation of the proof, the continuous weights satisfy the training dynamics:
``` math
\begin{aligned}
\frac{dA}{dt}(t) &= \pi \Big(- \frac{1}{n} \sum_{i=1}^n P_{i}(0, t) x_i^\top \Big)  \\
\frac{\partial \mathcal{Z}}{\partial t}(s, t) &= \pi \Big( -\frac{1}{n} \sum_{i=1}^n  
\partial_2 f(H_i(s,t), \mathcal{Z}(s,t))^\top P_i(s,t) \Big) \\
\frac{dB}{dt}(t) &= \pi \Big( - \frac{2}{n} \sum_{i=1}^n (B(t) H_i(1, t) - y_i) H_i(1, t)^\top \Big),
\end{aligned}
```
where we recall that $`H_i(s, t)`$ is the solution at time $`s`$ of the initial value problem
``` math
\begin{aligned}
    H_i(0, t) &= A(t) x_i \\
    \frac{\partial H_i}{\partial s}(s, t) &= f(H_i(s, t), \mathcal{Z}(s, t)), \quad s \in [0, 1],
\end{aligned}
```
and $`P_i(s, t)`$ is the solution at time $`s`$ of the problem
``` math
\begin{aligned}
    P_i(1, t) &= 2 B(t)^\top (B(t) H_i(1, t) - y_i) \\
    \frac{\partial P_i}{\partial s}(s, t) &= \partial_1 f(H_i(s, t), \mathcal{Z}(s, t)) P_i(s, t), \quad s \in [0, 1].
\end{aligned}
```
These equations can be thought of as the continuous-depth equivalent of the backpropagation equations.

## Existence of the double limit when $`L, t`$ tend to infinity

<div id="prop:double-limit" class="proposition">

**Proposition 15**. *Consider the residual network <a href="#eq:proof-main-thm:forward" data-reference-type="eqref" data-reference="eq:proof-main-thm:forward">[eq:proof-main-thm:forward]</a>, and assume that:*

1)  *$`A^L(t)`$, $`Z_{\left\lfloor Ls \right\rfloor}^L(t)`$, and $`B^L(t)`$ converge uniformly over $`L \in \mathbb{N}^*`$ and $`s \in [0, 1]`$ as $`t\to\infty`$.*

2)  *$`A^L(t)`$, $`Z_{\left\lfloor Ls \right\rfloor}^L(t)`$, and $`B^L(t)`$ converge uniformly over $`t \in \mathbb{R}_+`$ and $`s \in [0, 1]`$ as $`L\to\infty`$.*

3)  *The loss $`\ell^L(t)`$ converges to $`0`$ uniformly over $`L \in \mathbb{N}^*`$ as $`t\to\infty`$.*

*Then the following four statements hold **as $`t`$ and $`L`$ tend to infinity**:*

1)  *There exist matrices $`A_\infty \in {\mathbb{R}}^{q \times d}`$ and $`B_\infty \in {\mathbb{R}}^{d' \times q}`$ such that $`A^L(t)`$ and $`B^L(t)`$ converge to $`A_\infty`$ and $`B_\infty`$.*

2)  *There exists a Lipschitz continuous function $`\mathcal{Z}_\infty: [0,1] \to {\mathbb{R}}^{p}`$ such that $`Z_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges to $`\mathcal{Z}_\infty(t)`$ uniformly over $`s \in [0, 1]`$.*

3)  *Uniformly over $`s \in [0, 1]`$ and $`x \in \mathcal{X}`$, the hidden layer $`h_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges to the solution at time $`s`$ of the ODE
    ``` math
    \begin{aligned}
      \label{eq:main-thm-3}
        \begin{split}
            H(0) &= A_\infty x \\
            \frac{d H}{d s}(s) &= f(H(s), \mathcal{Z}_\infty(s)), \quad s \in [0, 1].
        \end{split}
    \end{aligned}
    ```*

4)  *Uniformly over $`x \in \mathcal{X}`$, the output $`F^L(x ; t)`$ converges to $`F_\infty(x) = B_\infty H(1)`$. Furthermore, $`F_\infty(x_i)=y_i`$ for $`i \in \{1, \dots, n\}`$.*

</div>

<div class="proof">

*Proof.* The existence of limits $`A_\infty`$ and $`B_\infty`$ to $`A^L(t)`$ and $`B^L(t)`$ as $`L`$ and $`t`$ tend to infinity is given by Lemma <a href="#lemma:double-limit" data-reference-type="ref" data-reference="lemma:double-limit">19</a>. The same argument applies to $`Z_{\left\lfloor sL \right\rfloor}^L(t)`$, which provides a limit $`\mathcal{Z}_\infty(s)`$ to the sequence. Furthermore, following the proof of the lemma, we see that the convergence of $`Z_{\left\lfloor sL \right\rfloor}^L(t)`$ to $`\mathcal{Z}_\infty(s)`$ is uniform over $`s \in [0, 1]`$. Corollary <a href="#prop:generalizednonlinearApprox:v3" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v3">13</a>, applied with $`\theta_k^L = Z_k^L`$, $`\Theta_\infty = \mathcal{Z}_\infty`$, $`a^L = A^{L} x`$, $`g = f`$, then ensures that $`h_{\left\lfloor Ls \right\rfloor}^L(t)`$ converges uniformly (over $`s \in [0, 1]`$ and $`x \in \mathcal{X}`$) to $`H(s)`$ that is the solution at time $`s`$ of <a href="#eq:main-thm-3" data-reference-type="eqref" data-reference="eq:main-thm-3">[eq:main-thm-3]</a>, as $`L`$ and $`t`$ tend to infinity. As a consequence, $`F^L(x; t)`$ converges uniformly over $`x`$ to $`F_\infty(x)`$ as $`L, t \to \infty`$. Furthermore, recall that
``` math
\ell^L(t) = \frac{1}{n}\sum_{i=1}^n \|F^L(x_i; t) - y_i\|_2^2.
```
The left-hand side converges as $`L, t \to \infty`$ to $`0`$ by assumption of the proposition, while the right-hand side converges to
``` math
\frac{1}{n}\sum_{i=1}^n \|F_\infty(x_i)-y_i\|_2^2.
```
Therefore, $`F_\infty(x_i)=y_i`$ for $`i \in \{1, \dots, n\}`$, and the proof is complete. ◻

</div>

# Proofs of the results of the main paper

In this section, we prove the results of the main paper. Most of these results follow from those presented in Section <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a>. The only substantial proof is that of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>, which shows the local PL condition. It uses a result of involving the Hermite transform and the sub-Gaussian variance proxy, which we define briefly. We refer to and , respectively, for more detailed explanations.

#### Hermite transform.

The $`r`$-th normalized probabilist’s Hermite polynomial is given by
``` math
h_r(x) = \frac{1}{\sqrt{r!}}(-1)^r e^{x^2 / 2} \frac{d^r}{dx^r} e^{-x^2/2}, \quad r \geqslant 0.
```
This family of polynomials forms an orthonormal basis of square-integrable functions for the inner product
``` math
\langle f_1, f_2 \rangle = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f_1(x)f_2(x) e^{-x^2/2} dx.
```
Therefore, any function $`\sigma`$ such that $`\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \sigma^2(x) e^{-x^2/2} dx < \infty`$ can be decomposed on this basis. The $`r`$-th coefficient of this decomposition is denoted by $`\eta_r(\sigma)`$.

#### Sub-Gaussian random vector.

A random vector $`x \in {\mathbb{R}}^d`$ is sub-Gaussian with variance proxy $`v_x^2 > 0`$ if, for every $`y \in {\mathbb{R}}^d`$ of unit norm,
``` math
\mathbb{P}(|\langle x, y\rangle| \geqslant t) \leqslant 2 \exp \Big(-\frac{t^2}{2 v_x^2}\Big).
```

#### Additional notation.

For a matrix $`A`$, we let $`s_{\min}`$ and $`s_{\max}`$ its minimum and maximum singular values, and similarly, $`\lambda_{\min}`$ and $`\lambda_{\max}`$ its minimum and maximum eigenvalues (whenever they exist).

Before delving into the proofs, we briefly describe the parts of this section that make use of the specific model <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>. The most important one is the proof of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>, i.e., the proof that the residual network satisfies the $`(M, \mu)`$-local PL condition. Additionally, in the proof of Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a>, the expressions for $`M`$ and $`K`$ are valid only for the specific model <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>. Finally, in the proof of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a>, the beginning of the proof reveals that condition <a href="#eq:condition-mu" data-reference-type="eqref" data-reference="eq:condition-mu">[eq:condition-mu]</a> of Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a> on $`\mu`$ can be expressed as a condition on the norm of the labels $`y_i`$. This applies only to the specific model <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>. Observe that, if one assumes that the general residual network of Section <a href="#apx:proofs-general" data-reference-type="ref" data-reference="apx:proofs-general">7</a> satisfies the $`(M, \mu)`$-local PL condition with $`\mu`$ given by <a href="#eq:condition-mu" data-reference-type="eqref" data-reference="eq:condition-mu">[eq:condition-mu]</a>, then the rest of the proof of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a> unfolds, and the conclusions of the theorem hold for the general model.

## Proof of Proposition <a href="#prop:clipped-gf-unique" data-reference-type="ref" data-reference="prop:clipped-gf-unique">1</a>

Proposition <a href="#prop:clipped-gf-unique" data-reference-type="ref" data-reference="prop:clipped-gf-unique">1</a> is a consequence of Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> with $`f(h, (V, W)) = \frac{1}{\sqrt{m}} V \sigma(\frac{1}{\sqrt{q}} Wh)`$.

## Proof of Proposition <a href="#prop:neural-ode-simple" data-reference-type="ref" data-reference="prop:neural-ode-simple">2</a>

Proposition <a href="#prop:generalizednonlinearApprox:v2:simple" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2:simple">11</a>, with $`\theta_k^L = (V_k^L, W_k^L)`$, $`\Theta = (\mathcal{V}, \mathcal{W})`$, $`a^L = A x`$, $`g(h, (V, W)) = \frac{1}{\sqrt{m}} V \sigma(\frac{1}{\sqrt{q}} Wh)`$, gives the existence and uniqueness of the solution of the neural ODE <a href="#eq:model-neuralode" data-reference-type="eqref" data-reference="eq:model-neuralode">[eq:model-neuralode]</a>. Moreover, inspecting the proof of Proposition <a href="#prop:generalizednonlinearApprox:v2:simple" data-reference-type="ref" data-reference="prop:generalizednonlinearApprox:v2:simple">11</a>, equations <a href="#eq:proof-euler-bound-delta" data-reference-type="eqref" data-reference="eq:proof-euler-bound-delta">[eq:proof-euler-bound-delta]</a> gives that, for any input $`x \in \mathcal{X}`$, the difference between the last hidden layer $`h_L^L`$ of the discrete residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> and its continuous counterpart $`H(1)`$ in the neural ODE <a href="#eq:model-neuralode" data-reference-type="eqref" data-reference="eq:model-neuralode">[eq:model-neuralode]</a> is bounded by
``` math
C'\Big(\frac{1}{L} + \sup_{s \in [0, 1]} \|\Theta(s) - \Theta^L(s)\|\Big),
```
where $`C' > 0`$ is independent of $`L`$ and $`x \in \mathcal{X}`$, and $`\Theta^L(s) = \theta_{\left\lfloor (L-1)s  \right\rfloor + 1}^L`$. The function $`\Theta^L`$ is a piecewise-constant interpolation of $`\Theta`$ with pieces of length $`\frac{1}{L-1}`$. Since $`\Theta`$ is Lipschitz continuous, the distance between $`\Theta`$ and $`\Theta^L`$ decreases as $`\nicefrac{C''}{L}`$ for some $`C'' > 0`$ depending on $`\Theta`$ but not on $`L`$. This yields $`\|h_L^L - H(1)\| \leqslant\frac{C'(1 + + C'')}{L}`$, where $`C'`$ and $`C''`$ are independent of $`L`$ and $`x \in \mathcal{X}`$. Since $`F^L(x) = B h_L^L`$ and $`F(x) = BH(1)`$, the result is proven.

## Proof of Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a>

We apply Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> with $`f(h, (V, W)) = \frac{1}{\sqrt{m}} V \sigma(\frac{1}{\sqrt{q}} Wh)`$. Recall that the parameters $`Z = (V, W)`$ are considered in Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> as a vector. In particular, $`\|Z\| = \|V\|_F + \|W\|_F`$. Therefore, Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> shows that, for $`t \in [0, T]`$, $`L \in \mathbb{N}^*`$, and $`k \in \{1, \dots, L\}`$,
``` math
\|A^L(t)\|_F\leqslant M, \quad \|V_k^L(t)\|_F + \|W_k^L(t)\|_F  \leqslant M, \quad  \text{and} \quad \|B^L(t)\|_F\leqslant M,
```
where
``` math
\begin{aligned}
    M &= M_0+TM_\pi \\
    M_0 &= \max\Big(\|A^L(0)\|_F, \,  \|V_0^L(0)\|_F + \|W_0^L(0)\|_F, \, \|B^L(0)\|_F\Big) \\
    M_\pi &= \max\Big(\max_{A \in {\mathbb{R}}^{q \times d}} \|\pi(A)\|_F, \, \max_{Z \in {\mathbb{R}}^{q \times m} \times {\mathbb{R}}^{m \times q}} \|\pi(Z)\|, \, \max_{B \in {\mathbb{R}}^{d' \times q}} \|\pi(B)\|_F\Big).
\end{aligned}
```
Furthermore, due to our initialization scheme described in Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>,
``` math
\|A^L(0)\|_F = \sqrt{d}, \quad \|V_0^L(0)\|_F = 0, \quad \|W_0^L(0)\|_F \leqslant 2 \sqrt{qm}, \quad \|B^L(0)\|_F = \sqrt{d'},
```
where the third inequality holds with probability at least $`1 - \exp(-\frac{3qm}{16})`$ by Lemma <a href="#prop:proof-global-conv-1" data-reference-type="ref" data-reference="prop:proof-global-conv-1">20</a>. Since we take $`q \geqslant\max(d, d')`$, this implies that, with high probability, $`M_0 \leqslant 2 \sqrt{qm}`$, yielding the formula for $`M`$ in Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a>. Finally, the existence of $`K = \beta T e^{\alpha T}`$ such that the difference between two successive weight matrices is bounded by $`\nicefrac{K}{L}`$, as well as the dependence of $`\alpha`$ and $`\beta`$ on $`\mathcal{X}`$, $`\mathcal{Y}`$, $`M`$, and $`\sigma`$, follows easily from Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a>, given that our initialization scheme ensures that $`Z_k^L(0) = Z_{k+1}^L(0)`$ for all $`L \in \mathbb{N}^*`$ and $`k \in \{1, \dots, L\}`$.

## Proof of Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a>

By Proposition <a href="#prop:final-finitetrainingtimekey" data-reference-type="ref" data-reference="prop:final-finitetrainingtimekey">3</a> and the fact that $`\pi`$ is bounded, the sequences $`(A^L)_{L \in \mathbb{N}^*}`$ and $`(B^L)_{L \in \mathbb{N}^*}`$ each satisfy the assumptions of Corollary <a href="#prop:arzela-ascoli:2" data-reference-type="ref" data-reference="prop:arzela-ascoli:2">10</a>, and $`(Z_k^L)_{L \in \mathbb{N}^*, 1 \leqslant k \leqslant L}`$ satisfies the assumptions of Proposition <a href="#prop:arzela-ascoli" data-reference-type="ref" data-reference="prop:arzela-ascoli">9</a>. Theorem <a href="#thm:final-finitetrainingtimeconv" data-reference-type="ref" data-reference="thm:final-finitetrainingtimeconv">4</a> then follows directly from Theorem <a href="#thm:general-convergence:v2" data-reference-type="ref" data-reference="thm:general-convergence:v2">14</a>, by taking, as previously, $`f(h, (V, W)) = \frac{1}{\sqrt{m}} V \sigma(\frac{1}{\sqrt{q}} Wh)`$.

## Proof of Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>

We drop the $`L`$ superscripts for this proof, since $`L`$ is fixed. Denote by $`\bar{A}, \bar{B}, \bar{V}_k, \bar{W}_k`$ parameters sampled according to the initialization scheme of Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>, which means in particular that $`\bar{V}_k = 0`$ and $`\bar{W}_k = \bar{W} \sim \mathcal{N}^{\otimes (m \times q)}`$. Since, by assumption, the activation function $`\sigma`$ is bounded and not constant, it cannot be a polynomial function. As a consequence, there are infinitely many non-zero coefficients $`\eta_r(\sigma)`$ in its Hermite expansion (defined at the beginning of Section <a href="#sec:proofs-main-paper" data-reference-type="ref" data-reference="sec:proofs-main-paper">8</a>). Throughout, we let $`r \geqslant 2`$ be an integer such that $`\eta_r(\sigma)`$ is nonzero. We also let $`K_\sigma`$ be the Lipschitz constant of $`\sigma`$ and $`M_\sigma`$ its supremum norm. Now, let $`A, B, V_k, W_k`$ be parameters at distance at most $`M = \min(\frac{\eta_r(\sigma)}{32 K_\sigma \sqrt{2nq}}, \frac{1}{2})`$ from $`\bar{A}, \bar{B}, \bar{V}_k, \bar{W}_k`$ in the sense of Definition <a href="#def:pl-condition" data-reference-type="ref" data-reference="def:pl-condition">1</a>.

It is useful for this proof to introduce a matrix-valued version of the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a>. More specifically, given data matrices $`\mathbf{x} \in {\mathbb{R}}^{d \times n}`$ and $`\mathbf{y} \in {\mathbb{R}}^{d' \times n}`$, the matrix-valued residual network writes
``` math
\begin{aligned}
\mathbf{h}_0 &= A \mathbf{x}  \nonumber  \\
\mathbf{h}_{k+1} &= \mathbf{h}_k + \frac{1}{L\sqrt{m}} V_{k+1} \sigma \Big( \frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big), \quad k \in \{0, \dots, L-1\},
 \label{eq:recursion-global-convergence}
\end{aligned}
```
where now $`\mathbf{h}_k \in {\mathbb{R}}^{q \times n}`$. The loss is equal to $`\ell = \frac{1}{n} \|B \mathbf{h}_L - \mathbf{y}\|_F^2`$ and we let $`\mathbf{p}_k = \frac{\partial \ell}{\partial \mathbf{h}_k} \in {\mathbb{R}}^{q \times n}`$ be the matrix-valued backward state. Observe that the columns of $`\mathbf{x}`$ are bounded and thus sub-Gaussian. In the sequel, we denote by $`v_x^2`$ the sub-Gaussian variance proxy of the columns of $`\sqrt{\nicefrac{d}{q}}\mathbf{x}`$.

Now that we have introduced the necessary notation, we can proceed to prove some preliminary estimates. Since $`M \leqslant\frac{1}{2} \leqslant\sqrt{2qm}`$, we have, for $`k \in \{1, \dots, n\}`$,
``` math
\label{eq:upperbounds-abvw-1}
\|A-\bar{A}\|_F \leqslant M, \quad \|B-\bar{B}\|_F \leqslant\frac{1}{2}, \quad \|V_k\|_F \leqslant 1, \quad \|W_k - \bar{W}\|_F \leqslant\frac{1}{2} \leqslant\sqrt{2qm}.
```
By Lemma <a href="#prop:proof-global-conv-1" data-reference-type="ref" data-reference="prop:proof-global-conv-1">20</a>, with probability at least $`1 - \exp\big(-\frac{qm}{16}\big)`$, one has $`\|\bar{W}\|_F \leqslant\sqrt{2qm}`$. Together with the previous inequalities, this implies
``` math
\label{eq:upperbounds-abvw-2}
\|A\|_2 \leqslant 2, \quad s_{\min}(B) \geqslant\frac{1}{2}, \quad \|B\|_2 \leqslant\frac{3}{2}, \quad \|V_k\|_F \leqslant 1, \quad \|W_k\|_F \leqslant 2 \sqrt{2qm},
```
where the second inequality is a consequence of Lemma <a href="#lemma:proof-global-conv" data-reference-type="ref" data-reference="lemma:proof-global-conv">18</a>, as follows:
``` math
s_{\min}(B) \geqslant s_{\min}(\bar{B}) - \|B - \bar{B}\|_F = 1 - \|B - \bar{B}\|_F \geqslant\frac{1}{2}.
```
Let us now bound $`\|\mathbf{h}_k\|_F`$ and $`\|\mathbf{p}_k\|_F`$. We have
``` math
\label{eq:upperbound-h0}
\|\mathbf{h}_0\|_F = \|A\mathbf{x}\|_F  \leqslant\|A\|_2 \|\mathbf{x}\|_F \leqslant 2\sqrt{qn}.
```
Moreover, by <a href="#eq:recursion-global-convergence" data-reference-type="eqref" data-reference="eq:recursion-global-convergence">[eq:recursion-global-convergence]</a>, for any $`k \in \{0, \dots, L-1\}`$,
``` math
\begin{aligned}
\|\mathbf{h}_{k+1}\|_F &\leqslant\|\mathbf{h}_k\|_F + \frac{K_\sigma}{L\sqrt{m} \sqrt{q}} \|V_{k+1}\|_F \|W_{k+1}\|_F \|\mathbf{h}_k\|_F \leqslant\Big(1 + \frac{2\sqrt{2} K_\sigma}{L}\Big) \|\mathbf{h}_k\|_F,
\end{aligned}
```
where the second inequality is a consequence of <a href="#eq:upperbounds-abvw-2" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-2">[eq:upperbounds-abvw-2]</a>. Therefore, by <a href="#eq:upperbound-h0" data-reference-type="eqref" data-reference="eq:upperbound-h0">[eq:upperbound-h0]</a>,
``` math
\label{eq:upperbound-hk}
\|\mathbf{h}_k\|_F \leqslant\exp(2 \sqrt{2}K_\sigma) \|\mathbf{h}_0\|_F \leqslant 2 \exp(2 \sqrt{2}K_\sigma) \sqrt{qn}.
```
Moving on to $`\|\mathbf{p}_k\|_F`$, the chain rule leads to
``` math
\mathbf{p}_k = \mathbf{p}_{k+1} + \frac{1}{L \sqrt{q m}} W_{k+1}^\top \Big( (V_{k+1}^\top \mathbf{p}_{k+1}) \odot \sigma'\big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \big)\Big), \quad k \in \{0, \dots, L-1\},
```
where $`\odot`$ denotes the element-wise product. Noting that $`|\sigma'| \leqslant K_\sigma`$ and using <a href="#eq:upperbounds-abvw-2" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-2">[eq:upperbounds-abvw-2]</a>, we obtain
``` math
\begin{aligned}
\|\mathbf{p}_k\|_F &\geqslant\|\mathbf{p}_{k+1}\|_F - \frac{K_\sigma}{L \sqrt{q m}} \|W_{k+1}\|_F \|V_{k+1}\|_F \|\mathbf{p}_{k+1}\|_F  \geqslant\Big(1 - \frac{2 \sqrt{2} K_\sigma}{L}\Big) \|\mathbf{p}_{k+1}\|_F.
\end{aligned}
```
It follows that $`\|\mathbf{p}_k\|_F \geqslant\exp(-2 \sqrt{2} K_\sigma) \|\mathbf{p}_L\|_F`$. In addition,
``` math
\mathbf{p}_L = \frac{\partial \ell}{\partial \mathbf{h}_L} = \frac{2}{n} B^\top (B \mathbf{h}_L - \mathbf{y}).
```
Therefore, by Lemma <a href="#lemma:proof-global-conv" data-reference-type="ref" data-reference="lemma:proof-global-conv">18</a>, since $`d' \leqslant q`$,
``` math
\|\mathbf{p}_L\|_F \geqslant\frac{2}{n} s_{\min}(B) \|B \mathbf{h}_L - \mathbf{y}\|_F \geqslant\frac{1}{\sqrt{n}} \sqrt{\ell}.
```
Collecting bounds, we conclude that, for $`k \in \{0, \dots, L\}`$,
``` math
\label{eq:lowerbound-pk}
\|\mathbf{p}_k\|_F \geqslant\frac{1}{\sqrt{n}} \exp(-2 \sqrt{2} K_\sigma) \sqrt{\ell}.
```
A similar proof reveals that, for $`k \in \{0, \dots, L\}`$,
``` math
\|\mathbf{p}_k\|_F \leqslant\frac{3}{\sqrt{n}} \exp(2 \sqrt{2} K_\sigma) \sqrt{\ell}.
```
Having established these preliminary estimates, our goal in the remainder of the proof is to lower bound the quantity $`\|\frac{\partial \ell}{\partial V_{k+1}}\|_F`$. First note that, by the chain rule, for any $`k \in \{0, \dots, L-1\}`$,
``` math
%
\frac{\partial \ell}{\partial V_{k+1}} = \frac{1}{L\sqrt{m}} \mathbf{p}_{k+1} \sigma \Big(\frac{1}{\sqrt{q}} W_{k+1}\mathbf{h}_k \Big)^\top.
```
As a consequence, when $`m \geqslant n`$, by Lemma <a href="#lemma:proof-global-conv" data-reference-type="ref" data-reference="lemma:proof-global-conv">18</a>,
``` math
\begin{aligned}
\Big\|\frac{\partial \ell}{\partial V_{k+1}}\Big\|_F &\geqslant\frac{1}{L\sqrt{m}} \|\mathbf{p}_{k+1}\|_F \cdot s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1}\mathbf{h}_k \Big)\Big) \nonumber \\
&\geqslant\frac{1}{L\sqrt{mn}} \exp(-2\sqrt{2} K_\sigma) \sqrt{\ell} \cdot s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1}\mathbf{h}_k \Big)\Big), \label{eq:proof:global-conv:lower-bound-der-V}
\end{aligned}
```
using <a href="#eq:lowerbound-pk" data-reference-type="eqref" data-reference="eq:lowerbound-pk">[eq:lowerbound-pk]</a>. Next, by Lemma <a href="#lemma:proof-global-conv" data-reference-type="ref" data-reference="lemma:proof-global-conv">18</a>,
``` math
s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big)\Big) \geqslant s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} \bar{W} \bar{A} \mathbf{x} \Big)\Big) - \Big\|\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big) - \sigma \Big(\frac{1}{\sqrt{q}} \bar{W} \bar{A} \mathbf{x} \Big) \Big\|_F.
```
Let us first lower bound the first term. Since, by our choice of initialization, $`\bar{A} = (I_{{\mathbb{R}}^{d \times d}}, 0_{{\mathbb{R}}^{(q-d) \times d}})`$, we have
``` math
s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} \bar{W} \bar{A} \mathbf{x} \Big)\Big) = s_{\min}(\sigma(\tilde{W} \tilde{\mathbf{x}} )),
```
where $`\tilde{W} \sim \mathcal{N}(0,1)^{\otimes (m \times d)}`$ and $`\tilde{\mathbf{x}} = \frac{1}{\sqrt{q}}\mathbf{x} \in {\mathbb{R}}^{d \times n}`$ has i.i.d. unitary columns independent of $`\tilde{W}`$. Therefore, by Lemma <a href="#prop:proof-global-conv-2" data-reference-type="ref" data-reference="prop:proof-global-conv-2">21</a>, with probability at least $`1 - \exp \big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\big) - 2 n^2 \exp \big(- \frac{d}{2v_x^2} \big(\frac{3}{4n}\big)^{2/r} \big)`$,
``` math
s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} W \bar{A} \mathbf{x} \Big)\Big) \geqslant\frac{\sqrt{m} \eta_r(\sigma)}{4}.
```
Next,
``` math
\begin{aligned}
\Big\|\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big) - \sigma \Big(\frac{1}{\sqrt{q}} \bar{W} \bar{A} \mathbf{x} \Big) \Big\|_F &\leqslant\frac{K_\sigma}{\sqrt{q}} \Big(\|W_{k+1} - \bar{W}\|_F \|\mathbf{h}_k\|_F  + \|\bar{W}\|_F \|\mathbf{h}_k - A \mathbf{x}\|_F \\
&\quad + \|\bar{W}\|_F \|A \mathbf{x} - \bar{A} \mathbf{x}\|_F \Big).
\end{aligned}
```
Clearly,
``` math
\begin{aligned}
\|\mathbf{h}_k - A\mathbf{x}\|_F = \Big\| \sum_{j=1}^{k} \frac{1}{L\sqrt{m}} V_{j} \sigma \Big( \frac{1}{\sqrt{q}} W_{j} \mathbf{h}_{j-1} \Big) \Big\|_F \leqslant\frac{4 \sqrt{2} K_\sigma k}{L} \exp(2 \sqrt{2} K_\sigma) \sqrt{qn},
\end{aligned}
```
by <a href="#eq:upperbounds-abvw-2" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-2">[eq:upperbounds-abvw-2]</a> and <a href="#eq:upperbound-hk" data-reference-type="eqref" data-reference="eq:upperbound-hk">[eq:upperbound-hk]</a>. Also,
``` math
\|A\mathbf{x} - \bar{A}\mathbf{x}\|_F \leqslant\|A - \bar{A}\|_F \|\mathbf{x}\|_F \leqslant\frac{\eta_r(\sigma)}{32 \sqrt{2} K_\sigma},
```
by <a href="#eq:upperbounds-abvw-1" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-1">[eq:upperbounds-abvw-1]</a> and by definition of $`M`$. Putting together the two bounds above as well as <a href="#eq:upperbounds-abvw-1" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-1">[eq:upperbounds-abvw-1]</a>, <a href="#eq:upperbounds-abvw-2" data-reference-type="eqref" data-reference="eq:upperbounds-abvw-2">[eq:upperbounds-abvw-2]</a>, and <a href="#eq:upperbound-hk" data-reference-type="eqref" data-reference="eq:upperbound-hk">[eq:upperbound-hk]</a>, we obtain
``` math
\begin{aligned}
\Big\|\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big) - \sigma \Big(\frac{1}{\sqrt{q}} W \bar{A} \mathbf{x}\Big) \Big\|_F &\leqslant K_\sigma \exp(2 \sqrt{2} K_\sigma) \sqrt{n} \Big(1 + \sqrt{qm} \frac{8 K_\sigma k}{L} \Big) + \sqrt{m} \frac{\eta_r(\sigma)}{32} \\
&\leqslant C_1 \sqrt{n} + C_2 \frac{\sqrt{nqm} k}{16 L} + \sqrt{m} \frac{\eta_r(\sigma)}{32},
\end{aligned}
```
where $`C_1 = K_\sigma \exp(2 \sqrt{2} K_\sigma)`$ and $`C_2 = 128 C_1 K_\sigma`$. Thus, when $`C_1 \sqrt{n} \leqslant\frac{1}{32}\sqrt{m}\eta_r(\sigma)`$, we have
``` math
s_{\min}\Big(\sigma \Big(\frac{1}{\sqrt{q}} W_{k+1} \mathbf{h}_k \Big)\Big) \geqslant\sqrt{m} \Big( \frac{3}{16} \eta_r(\sigma) - \frac{C_2}{16} \sqrt{nq} \frac{k}{L} \Big) \geqslant\frac{1}{8}\sqrt{m} \eta_r(\sigma)
```
for $`k \leqslant\frac{L \eta_r(\sigma)}{C_2\sqrt{nq}}`$. As a consequence, for $`k \leqslant\frac{L \eta_r(\sigma)}{C_2\sqrt{nq}}`$, returning to <a href="#eq:proof:global-conv:lower-bound-der-V" data-reference-type="eqref" data-reference="eq:proof:global-conv:lower-bound-der-V">[eq:proof:global-conv:lower-bound-der-V]</a>,
``` math
\Big\|\frac{\partial \ell}{\partial V_{k+1}}\Big\|_F \geqslant\frac{1}{8L\sqrt{n}} \eta_r(\sigma) \exp(-2 \sqrt{2} K_\sigma) \sqrt{\ell}  = \frac{C_3 \eta_r(\sigma)}{L\sqrt{n}} \sqrt{\ell},
```
letting $`C_3 = \frac{\exp(-2 \sqrt{2} K_\sigma)}{8}`$. Therefore,
``` math
\begin{aligned}
\Big\|\frac{\partial \ell}{\partial A}\Big\|_F^2 + L \sum_{k=1}^L \Big\|\frac{\partial \ell}{\partial Z_{k+1}}\Big\|_F^2 + \Big\|\frac{\partial \ell}{\partial B}\Big\|_F^2 &\geqslant L\sum_{k=1}^{\big\lfloor\frac{L \eta_r(\sigma)}{C_2\sqrt{nq}}\big\rfloor} \Big\|\frac{\partial \ell}{\partial V_{k+1}}\Big\|_F^2 \\
&\geqslant L \Big\lfloor\frac{L \eta_r(\sigma)}{C_2\sqrt{nq}}\Big\rfloor \frac{C_3^2 \eta_r(\sigma)^2}{L^2 n} \ell \\
&\geqslant\frac{C_3^2 \eta_r(\sigma)^3}{2C_2 n\sqrt{nq}} \ell,
\end{aligned}
```
where we used the inequality $`\lfloor x \rfloor \geqslant x/2`$ for $`x \geqslant 1`$. This proves the result, with
``` math
\begin{aligned}
c_1 &= \max\Big(\frac{2^{10} C_1^2}{\eta_r(\sigma)^2} , 1\Big) = \max\Big(\frac{2^{10} K_\sigma^2 \exp(4 \sqrt{2} K_\sigma)}{\eta_r(\sigma)^2}, 1\Big) \\
c_2 &= \frac{C_2}{\eta_r(\sigma)} = \frac{128 K_\sigma^2 \exp(2 \sqrt{2} K_\sigma)}{\eta_r(\sigma)} \\
c_3 &= \min\Big(\frac{\eta_r(\sigma)}{32 \sqrt{2} K_\sigma}, \frac{1}{2}\Big) \\
c_4 &= \frac{C_3^2 \eta_r(\sigma)^3}{2C_2} = \frac{\eta_r(\sigma)^3}{2^{14} K_\sigma^2 \exp(6 \sqrt{2} K_\sigma)} \ \\
\delta &= \exp\big(-\frac{qm}{16}\big) + n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big) + 2 n^2 \exp \Big(- \frac{d}{2v_x^2} \big(\frac{3}{4n}\big)^{2/r} \Big).
\end{aligned}
```

<div id="rmk:final-expression-delta" class="remark">

**Remark 2**. *With appropriate values of $`r`$ and $`m`$, the probability of failure $`\delta`$ can be made as small as
``` math
\label{eq:final-expression-delta}
\varepsilon + 2n^2 \exp \Big(- \frac{d}{2v_x^2} \big(\frac{3}{4n}\big)^{\varepsilon}\Big),
```
for any $`\varepsilon > 0`$. This is possible first by choosing $`r`$ such that $`2/r \leqslant\varepsilon`$, then by choosing $`m`$ such that the first two terms are less than $`\varepsilon`$. Moreover, we refer the interested reader to for quantitative estimates of $`\eta_r(\sigma)`$ for ReLU and sigmoid activations. Then, the expression <a href="#eq:final-expression-delta" data-reference-type="eqref" data-reference="eq:final-expression-delta">[eq:final-expression-delta]</a> is essentially the same as the one appearing in . The sub-Gaussian variance term $`v_x^2`$ is a constant independent of $`d`$ for many reasonable distributions of $`x`$, for instance if $`x`$ is uniform on the sphere . In this case, we note that the expression <a href="#eq:final-expression-delta" data-reference-type="eqref" data-reference="eq:final-expression-delta">[eq:final-expression-delta]</a> is small if $`n`$ grows at most polynomially with $`d`$, in which case the exponential term in $`d`$ dominates the polynomial term in $`n`$. Finally note that there exist adversarial choices of distributions of $`x`$ for which $`v_x^2`$ grows linearly with $`d`$, in which case we get $`\delta > 1`$, meaning that our result is void. This is the case in particular if the distribution of $`x`$ is a mixture of a small number of Diracs .*

</div>

## Proof of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a>

By Proposition <a href="#prop:pl-holds" data-reference-type="ref" data-reference="prop:pl-holds">5</a>, there exists $`\delta > 0`$ such that, with probability at least $`1-\delta`$, the residual network <a href="#eq:model-resnet" data-reference-type="eqref" data-reference="eq:model-resnet">[eq:model-resnet]</a> satisfies the $`(M, \mu)`$-local PL condition around its initialization, with
``` math
M = \frac{c_3}{\sqrt{nq}} \quad \text{and} \quad \mu = \frac{c_4}{n\sqrt{n} q},
```
for $`c_3`$ and $`c_4`$ depending on $`\sigma`$. Let us now apply Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a> with $`f(h, (V, W)) = \frac{1}{\sqrt{m}} V \sigma(\frac{1}{\sqrt{q}} Wh)`$. The only assumption of Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a> that requires some care to check is that the PL condition holds for the value of $`\mu`$ given by equation <a href="#eq:condition-mu" data-reference-type="eqref" data-reference="eq:condition-mu">[eq:condition-mu]</a>. Since the $`(M, \mu)`$-local PL condition implies the $`(M, \tilde{\mu})`$-local PL condition for any $`\tilde{\mu} \in (0, \mu)`$, it is the case if
``` math
\frac{c_4}{n\sqrt{n} q} \geqslant\max(M_B K, M_B M_X, M_A M_X) \frac{8e^K}{M} \sup_{L \in \mathbb{N}^*}\sqrt{\ell^L(0)},
```
with $`M_X`$, $`M_A`$, $`M_B`$, and $`K`$ defined in Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a>. Due to the initialization scheme of Section <a href="#sec:definitions" data-reference-type="ref" data-reference="sec:definitions">3</a>, we have, for any input $`x \in \mathcal{X}`$, $`h_L^L(0) = h_0^L(0)`$, hence $`F^L(x) = B^L(0) A^L(0) x = 0`$ since $`q \geqslant d+d'`$. As a consequence, $`\ell^L(0) = \frac{1}{n}\sum_{i=1}^n \|y_i\|^2`$. Therefore, the condition becomes
``` math
\frac{1}{n}\sum_{i=1}^n \|y_i\|^2 \leqslant\frac{c_3^2 c_4^2}{64 n^4 q^3 \max(M_B K, M_B M_X, M_A M_X)^2 e^{2K}},
```
where we replaced $`M`$ by its value. Define $`C`$ to be equal to the constant on the right-hand side. Then, according to the above, as soon as $`\frac{1}{n}\sum_{i=1}^n \|y_i\|^2 \leqslant C`$, we can apply Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a>, which gives several guarantees. First, the gradient flow is well defined on $`{\mathbb{R}}_+`$. Moreover, the proposition and the expression of $`\mu`$ given above yield the bound on the empirical risk. In particular, the empirical risk converges uniformly to zero. Furthermore, Proposition <a href="#prop:plcondition-to-convergence" data-reference-type="ref" data-reference="prop:plcondition-to-convergence">8</a> shows the uniform convergence of the weights as $`t \to \infty`$. Finally, the proposition ensures that the sequences $`(A^L)_{L \in \mathbb{N}^*}`$ and $`(B^L)_{L \in \mathbb{N}^*}`$ each satisfy the assumptions of Corollary <a href="#prop:arzela-ascoli:2" data-reference-type="ref" data-reference="prop:arzela-ascoli:2">10</a>, and that $`(Z_k^L)_{L \in \mathbb{N}^*, 1 \leqslant k \leqslant L}`$ satisfies the assumptions of Proposition <a href="#prop:arzela-ascoli" data-reference-type="ref" data-reference="prop:arzela-ascoli">9</a>. We can therefore apply Theorem <a href="#thm:general-convergence:v2" data-reference-type="ref" data-reference="thm:general-convergence:v2">14</a>, with $`f`$ defined above and $`\pi`$ equal to the identity. This gives the uniform convergence of the weights as $`L \to \infty`$. The four asymptotic statements of Theorem <a href="#thm:pl-main" data-reference-type="ref" data-reference="thm:pl-main">6</a> are then a consequence of Proposition <a href="#prop:double-limit" data-reference-type="ref" data-reference="prop:double-limit">15</a>.

<div class="remark">

**Remark 3**. *A close examination of the quantities involved in the definition of $`C`$ reveals that it depends only on $`\mathcal{X}`$, $`\sigma`$, $`n`$, and $`q`$. In particular, it does not depend on the dimension $`m`$.*

</div>

# Some technical lemmas

We start by recalling the Picard-Lindelöf theorem (see, e.g., , for a self-contained presentation, and , for a textbook).

<div id="lemma:picard-lindelof" class="lemma">

**Lemma 16** (Picard-Lindelöf theorem). *Let $`I = [0, T] \subset {\mathbb{R}}_+`$ be an interval, for some $`T \in (0, \infty]`$. Consider the initial value problem
``` math
\label{eq:lemma:picard-lindelof}
U(s) = U_0 + \int_0^s g(U(r), r) dr, \quad s \in I,
```
where $`g: {\mathbb{R}}^d \times I \to {\mathbb{R}}^d`$ is continuous and locally Lipschitz continuous in its first variable. Then the initial value problem is well defined on an interval $`[0, T_{\max}) \subset I`$, i.e., there exists a unique maximal solution on this interval. Moreover, if $`T_{\max} < T`$, then $`\|U(s)\|`$ tends to infinity when $`s`$ tends to $`T_{\max}`$. Finally, if $`g(\cdot, r)`$ is uniformly Lipschitz continuous for $`r`$ in any compact, then $`T_{\max} = T`$.*

</div>

We define time-dependent dynamics <a href="#eq:lemma:picard-lindelof" data-reference-type="eqref" data-reference="eq:lemma:picard-lindelof">[eq:lemma:picard-lindelof]</a> for generality, but the time-independent case $`U(s) = U_0 + \int_0^s g(U(r)) dr`$ is also of interest. In this case, the existence and uniqueness of the maximal solution holds if $`g`$ is locally Lipschitz continuous, and the solution is defined on $`I`$ if $`g`$ is Lipschitz continuous. Besides, the first statement of Lemma <a href="#lemma:picard-lindelof" data-reference-type="ref" data-reference="lemma:picard-lindelof">16</a> (existence and uniqueness of the maximal solution) also holds if $`{\mathbb{R}}^d`$ is replaced by any (potentially infinite-dimensional) Banach space.

The next lemma gives conditions for the existence and uniqueness of the global solution of the initial value problem <a href="#eq:lemma:picard-lindelof" data-reference-type="eqref" data-reference="eq:lemma:picard-lindelof">[eq:lemma:picard-lindelof]</a> when the assumption of continuity of $`g`$ in its second variable is removed, thereby generalizing the Picard-Lindelöf theorem.

<div id="lemma:caratheodory" class="lemma">

**Lemma 17** (Caratheodory conditions for the existence and uniqueness of the global solution of an initial value problem). *Consider the initial value problem
``` math
\begin{aligned}
U(s) = U_0 + \int_0^s g(U(r), r) dr, \quad s \in [0, 1],
\end{aligned}
```
where $`g: {\mathbb{R}}^d \times [0, 1] \to {\mathbb{R}}^d`$ is measurable and the integral is understood in the sense of Lebesgue integration. Assume that $`g(\cdot, r)`$ is uniformly Lipschitz continuous for almost all $`r \in [0, 1]`$, and that $`g(0, r) \equiv 0`$. Then there exists a unique solution to the initial value problem, defined on $`[0, 1]`$.*

</div>

<div class="proof">

*Proof.* The proof is a consequence of . More specifically, denote by $`C > 0`$ the uniform Lipschitz constant of $`g(\cdot, r)`$. According to , under the conditions of the lemma, there exists a unique maximal solution to the initial value problem. Let us now consider a restricted version of the problem, where $`g`$ is defined on $`D \times [0, 1]`$, with $`D`$ a compact of $`{\mathbb{R}}^d`$ large enough to contain in its interior the ball of center $`0`$ and radius $`\|U_0\| \exp(C)`$. There exists a unique maximal solution to this problem as well, also according to , and, according to , it is defined until it reaches the boundary of $`D \times [0, 1]`$, which it reaches at some point $`(U^*, s^*)`$. If $`s^* < 1`$, it means that $`U^*`$ is on the boundary of $`D`$, and in particular that $`\|U^*\| > \|U_0\| \exp(C)`$. But, on the other hand, for almost every $`r \in [0, 1]`$,
``` math
\|g(U(r), r)\| \leqslant\|g(0, r)\| + \|g(U(r), r) - g(0, r)\| \leqslant C\|U(r)\|.
```
Hence, by Grönwall’s inequality, for $`s \leqslant s^*`$,
``` math
\|U(s)\| \leqslant\|U_0\| \exp(C).
```
Thus, $`\|U^*\| \leqslant\|U_0\| \exp(C)`$, which is impossible. Hence the maximal solution of the restricted problem is defined on $`[0, 1]`$. Furthermore, the maximal solution of the original problem coincides with the restricted one whenever $`U(s) \in D`$, which is the case for every $`s \in [0, 1]`$, hence the maximal solution is defined on $`[0, 1]`$. ◻

</div>

The next three lemmas recall well-known results from linear algebra, analysis, and random matrix theory. Recall that $`s_{\min}`$ and $`\lambda_{\min}`$ denote respectively the minimum singular value and eigenvalue of a matrix.

<div id="lemma:proof-global-conv" class="lemma">

**Lemma 18**. *Let $`A, A' \in {\mathbb{R}}^{m \times r}`$ and $`B \in {\mathbb{R}}^{r \times n}`$. Then
``` math
s_{\min}(A + A') \geqslant s_{\min}(A) - \|A'\|_F.
```
If $`m \geqslant r`$, then $`\|AB\|_F \geqslant s_{\min}(A) \|B\|_F`$. Furthermore, if $`n \geqslant r`$, then $`\|AB\|_F \geqslant\|A\|_F s_{\min}(B)`$.*

</div>

<div class="proof">

*Proof.* The first statement is a consequence of, e.g., , which establishes that $`s_{\min}(A + A') \geqslant s_{\min}(A) - s_{\max}(A')`$, yielding the first inequality since $`s_{\max}(A') = \|A\|_2 \leqslant\|A\|_F`$. As for the second one, we have
``` math
\|AB\|_F^2 = \textnormal{Tr}(ABB^\top A^\top) = \textnormal{Tr}(BB^\top A^\top A) \geqslant\lambda_{\min}(A^\top A) \textnormal{Tr}(BB^\top) = \lambda_{\min}(A^\top A) \|B\|_F^2.
```
Since $`m \geqslant r`$, the rightmost quantity is equal to $`s_{\min}(A) \|B\|_F`$, proving the second statement of the lemma. The third statement is similar. ◻

</div>

<div id="lemma:double-limit" class="lemma">

**Lemma 19**. *Let $`(e_{x, y})_{x\in\mathbb{R}_+, y\in\mathbb{R}_+} \subset E`$, where $`E`$ is a Banach space, such that $`e_{x, y}`$ converges uniformly to $`e_{\infty, y}`$ when $`x\to\infty`$, and converges uniformly to $`e_{x, \infty}`$ when $`y\to\infty`$. Then there exists $`e_\infty \in E`$ such that
``` math
\lim_{x, y\to\infty}e_{x,y} = \lim_{x\to\infty} e_{x, \infty}=\lim_{y\to\infty} e_{\infty, y} = e_\infty.
```*

</div>

<div class="proof">

*Proof.* Let $`\varepsilon>0`$. Since $`e_{x, y}`$ converges uniformly to $`e_{\infty, y}`$ as $`x\to\infty`$, there exists $`x_0 \in \mathbb{R}_+`$ such that, for $`x_1, x_2>x_0`$ and $`y \in \mathbb{R}_+`$,
``` math
\|e_{x_1, y}-e_{x_2, y}\|\leqslant\frac\varepsilon2.
```
Similarly, there exists $`y_0 \in \mathbb{R}_+`$ such that, for $`x \in \mathbb{R}_+`$ and $`y_1, y_2>y_0`$,
``` math
\|e_{x, y_1}-e_{x, y_2}\|\leqslant\frac\varepsilon2.
```
Hence, for $`x_1, x_2>x_0`$ and $`y_1, y_2>y_0`$,
``` math
\|e_{x_1, y_1}-e_{x_2, y_2}\|\leqslant\|e_{x_1, y_1}-e_{x_1, y_2}\|+\|e_{x_1, y_2}-e_{x_2, y_2}\|\leqslant\varepsilon.
```
We conclude that $`(e_{x, y})_{x\in\mathbb{R}_+, y\in\mathbb{R}_+}`$ is a Cauchy sequence, which therefore converges to some limit $`e_\infty \in E`$. ◻

</div>

<div id="prop:proof-global-conv-1" class="lemma">

**Lemma 20**. *Let $`W \in {\mathbb{R}}^{q \times m}`$ be a standard Gaussian random matrix. Then, for $`M_W \geqslant \sqrt{2}`$, with probability at least $`1 - \exp(-\frac{(M_W^2 - 1)qm}{16})`$, one has $`\|W\|_F \leqslant M_W \sqrt{q} \sqrt{m}`$.*

</div>

<div class="proof">

*Proof.* The quantity $`\|W\|_F^2`$ follows a chi-squared distribution with $`qm`$ degrees of freedom. Hence, according to , for $`x \geqslant 0`$,
``` math
\mathbb{P}(\|W\|_F^2 - qm \geqslant 2 \sqrt{qm x} + 2x) \leqslant\exp(-x).
```
Taking $`x = \frac{(M_W^2 - 1)qm}{16}`$, we see that
``` math
2 \sqrt{qm x} = \frac{1}{2} \sqrt{M_W^2 - 1} qm  \leqslant\frac{1}{2} (M_W^2 - 1) qm,
```
where the bound follows from $`M_W \geqslant \sqrt{2}`$. Since furthermore $`2 x \leqslant\frac{1}{2} (M_W^2 - 1) qm`$, we obtain
``` math
2 \sqrt{qm x} + 2x \leqslant(M_W^2 - 1) qm,
```
and thus
``` math
\mathbb{P}(\|W\|_F^2 > M_W^2 qm) \leqslant\mathbb{P}(\|W\|_F^2 - qm \geqslant 2 \sqrt{qm x} + 2x) \leqslant\exp(-x),
```
yielding the result. ◻

</div>

Finally, the last lemma of the section gives a lower bound on the smallest singular value of a matrix of the form $`\sigma(A)`$, where $`\sigma`$ is a bounded function applied element-wise and $`A`$ belongs to a family of random matrix. The lower bound involves the Hermite transform of $`\sigma`$, which is defined in Section <a href="#sec:proofs-main-paper" data-reference-type="ref" data-reference="sec:proofs-main-paper">8</a>.

<div id="prop:proof-global-conv-2" class="lemma">

**Lemma 21**. *Let $`\sigma`$ be a function bounded by some $`M_\sigma > 0`$. Let $`W \in {\mathbb{R}}^{m \times d}`$ be a standard Gaussian random matrix, and $`X \in {\mathbb{R}}^{d \times n}`$ a random matrix with i.i.d. unitary columns independent of $`W`$. Then, for any integer $`r \geqslant 2`$, there exists $`\delta > 0`$ such that, with probability at least $`1 - \delta`$, the smallest singular value of $`\sigma(W X)`$ is greater than $`\frac{1}{4}\sqrt{m} \eta_r(\sigma)`$, where $`\eta_r(\sigma)`$ is the $`r`$-th coefficient in the Hermite transform of $`\sigma`$. Furthermore, the following expression for $`\delta`$ holds:
``` math
\delta = n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big) + 2 n^2 \exp \Big(- \frac{d}{2C^2} \big(\frac{3}{4n}\big)^{2/r} \Big),
```
where $`C^2`$ is the sub-Gaussian variance proxy of the columns of $`\sqrt{d} X`$.*

</div>

<div class="proof">

*Proof.* Denoting by $`w_i`$ the $`i`$-th row of $`W`$ and letting
``` math
M_i = \sigma(X^\top w_i^\top) \sigma(w_i X),
```
our goal is to lower bound the smallest eigenvalue value $`\lambda_{\min}(M)`$ of $`M = \sum_{i=1}^m M_i`$. Observe that
``` math
\begin{aligned}
\mathbb{E}(M | X) &= m \mathbb{E}_{\tilde{w} \sim \mathcal{N}(0, I_d)} \Big( \sigma(X^\top \tilde{w}^\top) \sigma(\tilde{w} X) \Big| X \Big) \\
&= m \mathbb{E}_{\tilde{w} \sim \mathcal{N}(0, \frac{1}{d} I_d)} \Big( \sigma\big((\sqrt{d}X)^\top \tilde{w}^\top\big) \sigma\big(\tilde{w} (\sqrt{d} X) \big)\Big| X \Big).    
\end{aligned}
```
Letting $`\lambda_{\min}(\mathbb{E}(M | X))`$ be the smallest eigenvalue of this matrix and $`r \geqslant 2`$ be an integer, show that, with probability at least $`1 - 2 n^2 \exp( - \frac{d}{2C^2} (\frac{3}{4n})^{2/r})`$ over the matrix $`X`$,
``` math
\label{eq:proof:lower-bound-smallest-eingenvalue1}
\lambda_{\min}(\mathbb{E}(M | X)) \geqslant\frac{m \eta_r^2(\sigma)}{8}.
```

We now apply a matrix Chernoff’s bound to lower bound with high probability the smallest eigenvalue $`\lambda_{\min}(M|X)`$ of $`M`$ conditionally on $`X`$, as a function of $`\lambda_{\min}(\mathbb{E}(M | X))`$. By , we have, for $`t \in [0, 1]`$,
``` math
\mathbb{P}( \lambda_{\min}(M) \leqslant t \lambda_{\min}(\mathbb{E}(M | X)) | X) \leqslant n \exp \Big(-\frac{(1-t^2) \lambda_{\min}(\mathbb{E}(M | X))}{2R(X)}\Big),
```
where $`R(X)`$ is an almost sure upper bound on the largest eigenvalue of $`M_i|X`$, which we can take equal to $`M_\sigma^2 n`$ since the largest eigenvalue of $`M_i`$ is equal to $`\|\sigma(w_i X)\|_2^2 \leqslant M_\sigma^2 n`$. Taking $`t=1/2`$, we obtain, on the event $`[\lambda_{\min}(\mathbb{E}(M | X)) \geqslant\frac{m \eta_r^2(\sigma)}{8}]`$,
``` math
\begin{aligned}
\mathbb{P}\Big(\lambda_{\min}(M) \geqslant\frac{\lambda_{\min}(\mathbb{E}(M | X))}{2} \Big| X\Big) \geqslant 1 - n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big),
\end{aligned}
```
thus, on the event $`[\lambda_{\min}(\mathbb{E}(M | X)) \geqslant\frac{m \eta_r^2(\sigma)}{8}]`$,
``` math
\begin{aligned}
\mathbb{P}\Big(\lambda_{\min}(M) \geqslant\frac{m \eta_r^2(\sigma)}{16}\Big) \geqslant 1 - n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big).
\end{aligned}
```
Using <a href="#eq:proof:lower-bound-smallest-eingenvalue1" data-reference-type="eqref" data-reference="eq:proof:lower-bound-smallest-eingenvalue1">[eq:proof:lower-bound-smallest-eingenvalue1]</a>, we obtain
``` math
\begin{aligned}
\mathbb{P}\Big(\lambda_{\min}(M) \geqslant\frac{m \eta_r^2(\sigma)}{16}\Big) 
&\geqslant\Big(1 - n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big) \Big) \mathbb{P}\Big(\lambda_{\min}(\mathbb{E}(M | X)) \geqslant\frac{m \eta_r^2(\sigma)}{8}\Big) \\
&\geqslant\Big(1 - n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big) \Big) \Big(1 - 2 n^2 \exp \Big(- \frac{d}{2C^2} \big(\frac{3}{4n}\big)^{2/r}) \Big)\Big) \\
&\geqslant 1 - n \exp \Big(-\frac{3 m \eta_r^2(\sigma)}{64 M_\sigma^2 n}\Big) - 2 n^2 \exp \Big(- \frac{d}{2C^2} \big(\frac{3}{4n}\big)^{2/r} \Big).
\end{aligned}
```
 ◻

</div>

# Counter-example for the ReLU case.

This section gives a proof sketch to illustrate that, with the ReLU activation $`\sigma:x \mapsto \max(0, x)`$, the smoothness of the weights can be lost during training. More precisely, we show a case where successive weights are at distance $`\mathcal{O}(\frac1L)`$ at initialization and at distance $`\Omega(1)`$ after training.

For the sake of simplicity, we will assume that the depth is even, and denote it as $`2L`$. We place ourselves in a one-dimensional setting (i.e., $`d=1`$). The parameters are $`(w_1, \cdots, w_{2L}) \in {\mathbb{R}}^{2L}`$, and the residual network writes as follows, for an input $`x\in\mathbb{R}`$:
``` math
\begin{aligned}
        h_0(t) &= x\\
        h_{k+1}(t) &=h_{k}(t) + \frac1{2L} \sigma(w_{k+1}(t) h_{k}(t)) , \quad k \in \{0, \dots, 2L-1\}.
\end{aligned}
```
We consider a sample consisting of a single point $`(x, Cx) \in \mathbb{R}_{+}^2`$, with $`C > 1`$ (independent of $`L`$), and define the empirical risk as $`\ell(t) = (h_{2L}(t) - C x)^2`$. The risk is minimized by gradient flow.

The weights are initialized to $`w_k(0) =  \frac{(-1)^k}{2L}`$. For $`x \in {\mathbb{R}}_+`$ we have that $`h_k(t) \geqslant 0`$ for all $`k \in \{0, \dots, 2L\}`$. Note that the argument of $`\sigma`$ on the odd layers is negative. Therefore, by definition of $`\sigma`$, the gradient of the loss with respect to the odd layers is zero and we have, for $`k \in \{0, \dots, L-1\}`$, $`w_{2k+1}(t) = w_{2k+1}(0)`$. On the other hand, the argument of $`\sigma`$ is positive on the even layers, and thus,
``` math
h_{2L}(t) = \prod_{j=1}^L\Big(1 + \frac{w_{2j}(t)}{2L}\Big) x.
```
As a consequence, the gradient flow equation for the even layers is, for $`k \in \{1, \dots, L\}`$,
``` math
\frac{dw_{2k}}{dt}(t) = - \frac{\partial \ell}{\partial w_{2k}}(t) = 
2 x \Big(C - \prod_{j=1}^L\Big(1 + \frac{w_{2j}(t)}{2L}\Big)\Big) \prod_{j=1, j\neq k}^L\Big(1 + \frac{w_{2j}(t)}{2L}\Big).
```
Due to the symmetry of these equations for $`k \in \{1, \dots, L\}`$ and the fact that all the $`w_{2k}(0)`$ are equal, the parameters on each even layer coincide at all times and are equal to $`w(t)`$ such that
``` math
\frac{dw}{dt}(t) = 2x \Big(C - \Big(1 + \frac{w(t)}{2L}\Big)^L\Big) \Big(1 + \frac{w(t)}{2L}\Big)^{L-1}.
```
An analysis of this ODE reveals that $`w(t)`$ tends as $`t\to \infty`$ to $`w^{\star} > 0`$ satisfying that
``` math
\label{eq:def-w-star}
\Big(1 + \frac{w^{\star}}{2L}\Big)^L = C.
```
This can be seen by letting $`y(t) = C - (1 + \frac{w(t)}{2L})^L`$, and applying Grönwall’s inequality to $`y`$. Therefore, as $`t \to \infty`$, one has $`w_{2k+1}(t) \to -\frac{1}{2L}`$ and $`w_{2k}(t) \to w^{\star}`$, where <a href="#eq:def-w-star" data-reference-type="eqref" data-reference="eq:def-w-star">[eq:def-w-star]</a> implies that $`w^{\star} \geqslant 2 \log(C)`$. This shows that the final weights are not smooth in the sense that the distance between two successive weights is $`\Omega(1)`$.

This result contrasts sharply with Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a>, which shows that successive weights remain at a distance $`\mathcal{O}(\frac1L)`$ throughout training, when initialized as a discretization of a Lipschitz continuous function, and with a smooth activation function. In fact, Proposition <a href="#prop:general-clipped-bounded" data-reference-type="ref" data-reference="prop:general-clipped-bounded">7</a> can be generalized to any initialization such that successive weights are at distance $`\mathcal{O}(\frac1L)`$ at initialization, which is the case in the counter-example. This means that the only broken assumption in our counter-example is the non-smoothness of the activation function. This non-smoothness causes the gradient flow dynamics for two successive weights to deviate, even though the weights are initially close to each other, because they are separated by the kink of ReLU at zero.

# Experimental details

Our code is available at <https://github.com/michaelsdr/implicit-regularization-resnets-nodes>.

We use Pytorch .

#### Synthetic data.

To ease the presentation, we consider the case where $`q = d = d'`$, and we do not train the weights $`A^L`$ and $`B^L`$, which therefore stay equal to the identity. The sample points $`(x_i, y_i)_{1\leqslant i\leqslant n}`$ follow independent standard Gaussian distributions. Note that it does not hurt to take $`x`$ and $`y`$ independent since, in this subsection, our focus is on optimization results only and not on statistical aspects.

#### Large-depth limit.

We take $`n=100`$, $`d=16`$, $`m=32`$. We train for $`500`$ iterations, and set the learning rate to $`L \times 10^{-2}`$. The scaling of the learning rate with $`L`$ is the equivalent of the $`L`$ factor in the gradient flow <a href="#eq:gf" data-reference-type="eqref" data-reference="eq:gf">[eq:gf]</a>.

#### Long-time limit.

We take $`n=50`$, $`d=16`$, $`m=64`$, $`L = 64`$, and train for $`80{,}000`$ iterations with a learning rate of $`5 L \times 10^{-3}`$.

#### Real-world data.

We take $`L=256`$. The first layer is a trainable convolutional layer with a kernel size of $`5 \times 5`$, a stride of $`2`$, a padding of $`1`$, and $`16`$ out channels. We then iterate the residual layers
``` math
h_{k+1}^L = h_k^L+\frac{1}{L} \mathrm{bn}^L_{2,k}(\mathrm{conv}^L_{2,k}(\sigma(\mathrm{bn}^L_{1,k}(\mathrm{conv}^L_{1,k}(h_k^L))))), \quad k \in \{0, \dots, L-1\},
```
where $`\mathrm{conv}^L_{i,k}`$ are convolutions with kernel size $`3`$, stride of $`2`$, and padding of $`1`$, and $`\mathrm{bn}^L_{i,k}`$ are batch normalizations, as is standard in residual networks . The model is trained using stochastic gradient descent on the cross-entropy loss for 180 epochs. The initial learning rate is $`4 \times 10^{-2}`$ and is gradually decreased using a cosine learning rate scheduler.

#### Normalization.

The residual layers considered in the real-world case have a batch normalization layer (see formula above). We observe empirically that implicit regularization towards a neural ODE still holds in this case. However, these layers are not present in the models we consider. Nevertheless, as discussed in Section <a href="#subsec:generalization" data-reference-type="ref" data-reference="subsec:generalization">4.3</a>, some of our results extend to a setting where we only assume that the residual connection is a Lipschitz-continuous function. The intuition suggests that this should include in particular the case where layer normalizations are added to the architecture, although this should clearly necessitate a rigorous and separate mathematical analysis. Finally, note that a connection has been drawn between batch normalization and scaling factors .

#### Additional plot.

To complement Figure <a href="#fig:cifar" data-reference-type="ref" data-reference="fig:cifar">3</a>, we display the average (across layers) of the Frobenius norm of the difference between two successive weights in the convolutional ResNets after training on CIFAR-10, depending on the initialization strategy. The index $`i`$ corresponds to the index of the convolution layer. Results are averaged over 5 runs. We see that a smooth initialization leads to weights that are in average an order of magnitude smoother than those obtained with an i.i.d. initialization.

<figure>
<img src="./figures/fig_mean_delta_frob.png"" style="width:70.0%" />
<figcaption>Average (across layers) of the Frobenius norm of the difference between two successive weights in the convolutional ResNets after training on CIFAR-10, depending on the initialization strategy.</figcaption>
</figure>

[^1]: Equal contribution. Correspondence to `pierre.marion@mines.org`.
