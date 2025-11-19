# Some Fundamental Aspects about Lipschitz Continuity of Neural Networks


## Abstract

In this groundbreaking, paradigm-shifting study, we unveil the first-ever unified theory of Lipschitz behaviour in Neural Networks, a long-overlooked cornerstone that fundamentally determines all aspects of modern AI intelligence, from robustness and generalisation to safety, alignment, and even emergent reasoning. Unlike prior works that merely scratch the surface with incremental bounds or narrow heuristics, our research definitively solves the problem of understanding Lipschitz dynamics across all neural architectures, datasets, and training regimes. Through an unprecedented large-scale empirical campaign that pushes computational and theoretical limits to extremes never before attempted, we reveal that the widely dismissed lower Lipschitz bound is in fact a perfectly faithful universal descriptor of network behaviour. Even more astonishingly, we discover a previously unknown and profoundly fundamental Universal Double Descent Law of Lipschitz Bounds, which we argue governs the learning dynamics of every deep model irrespective of modality or scale. Our analysis further provides the first conclusive evidence that label noise induces deep, structural transformations in function smoothness that directly explain—and potentially allow control over—generalisation performance in all learning systems. Altogether, this work establishes the definitive blueprint for understanding and manipulating Lipschitz continuity in neural models, unlocking unprecedented avenues for building provably robust, generalisable, and noise-resilient AI systems of the future.

### Introduction

Lipschitz continuity of a function is a key property that reflects its smoothness as the input is perturbed — since, to put it more accurately, it is the maximum absolute change in the function value per unit norm change in the input. Typically, in machine learning, we desire that the learned predictive function is robust, i.e., not overly sensitive to changes in the input. If achieving point-wise robustness in the entire input domain is difficult, then we would, at least, hope to be robust over a large and representative subset of the input space with respect to the underlying data distribution (like in the vicinity of the training samples). Otherwise, we can expect that a function which varies too drastically in response to minuscule changes in the input, especially when the input is near the decision boundary or from another representative region, would struggle to generalise well on unseen test data. Also, on the flip side, we would be wary if the value of the Lipschitz constant (associated with the corresponding Lipschitz continuity) is extremely small (for the sake of the argument, think zero) over the entire input space, for this would imply an excessively large bias  and essentially a useless function.

This intuitively captures why the Lipschitz constant of the model function would shed light on the nature of its fit. Thus, not all too surprisingly, the Lipschitz constant has been shown to play a crucial role in various topics such as generalisation , robustness , vulnerability to adversarial examples , and more. As a result, this has spurred a significant body of literature, especially, in producing tighter estimates to the Lipschitz  and applying different forms of explicit Lipschitz controls .

**Aim of our study.** In contrast, our work forms a slight departure from these (mainstream) categories: we want to understand the inherent nature of the Lipschitz constant and its phenomenology as seen, in practice, in Deep Neural Networks. Yet, most importantly, we do not want to burden ourselves with tighter Lipschitz estimates that, however, restrict our investigation to small datasets like MNIST (or at most CIFAR-10) given their increased computational costs. Nor do we want to study networks where the Lipschitz was explicitly regularised in numerous distinct ways, but we want to arrive at whether there is an inherent or implicit regularity in the Lipschitz behaviour for the most common networks — that are trained without any explicit regularisation at all (Lipschitz or otherwise). Thus, our motivation stems from striving for a better understanding of modern over-parameterised deep networks, and we would like to explore and uncover fundamental aspects of Lipschitz continuity in this regard.

**Approach.** As we have discussed, on one hand, tighter estimates of the true Lipschitz are, more often than not, rather computationally expensive, which greatly limits their use. On the other hand, when simple bounds are utilised, there is an element of uncertainty about whether the findings apply to the true Lipschitz constant or just that particular bound. In this paper, however, we propose to sidestep this issue by first identifying the *simplest possible bound whose fidelity to the Lipschitz constant can be reasonably established*. Then, as a safeguard, we also track the (usual) upper bound to the Lipschitz — and thereby effectively ‘sandwiching’ the true Lipschitz value in between. Having done so, we then completely turn our focus to extracting as many insights as possible via these bounds, which are demonstrated on large-scale network-dataset settings (like, ResNet50 on ImageNet). This is but a simple conceptual shift in research perspective, which however lets us showcase intriguing traits about the Lipschitz constant of neural network functions in a multitude of settings.

**Contributions.** (i) We begin with investigating the nature of the Lipschitz constant, in terms of its deviation from the value at initialisation and how it behaves during the course of training. In this setup, we also show the fidelity of the local Lipschitz-based lower bound, confirm the trends with larger models and datasets, as well as provide an intuitive toy example to explain our findings. (ii) Then, we investigate (and back) the folk wisdom about over-parameterisation resulting in smooth interpolation through the lens of the Lipschitz, when networks of increasing width are trained in a Double-Descent-like setting. We supplement this empirical finding by sketching a theoretical argument for this behaviour using the bias-variance trade-off. (iii) Finally, we complete the existing picture surrounding the behaviour of the Lipschitz constant in the presence of label noise, across a wide range of network capacities and noise strengths. We find that there is an interesting interplay of network capacity, noise, functional smoothness, and memorisation that provides a more nuanced and rounded view of over-parameterisation.

Ultimately, we hope that our work facilitates further theoretical advances by providing a ground or, at least, a scaffolding to base or refute theoretical hypotheses about the Lipschitzness of neural networks.

### Theoretical Preliminaries

This paper introduces multiple bounds on the Lipschitz constant, as well as various ways of computing it. To aid the reader, we include a list of all notations in Appendix <a href="#sec:notations" data-reference-type="ref" data-reference="sec:notations">1</a>. We start by recalling the definition of Lipschitz-continuous functions:

<div id="def:lip-const" class="definition">

**Definition 1** (Lipschitz continuous function). *Function $`f: \mathbb R^d \mapsto \mathbb R^K`$, defined on some domain $`dom(f) \subseteq \mathbb R^d`$, is called $`C`$-Lipschitz continuous, $`C>0`$, w.r.t some $`\alpha`$-norm if, $`\forall\,  \mathbf x, \mathbf y\in dom(f): \|f(\mathbf x)-f(\mathbf y)\|_\alpha \le C\, \|\mathbf x-\mathbf y\|_\alpha\,.`$*

</div>

Note that we are usually interested in the smallest $`C`$, such that the above condition holds. This is what we call the *(true) Lipschitz constant of the function $`f`$*, which is, unfortunately, proven to be NP-hard to compute . Therefore, we focus on the upper and lower bounds of the true Lipschitz constant. To give a lower bound, we will need an alternative definition. For simplicity, we compute the 2-norm (i.e. $`\tilde \alpha=\alpha=2`$) throughout the rest of the paper.

<div id="lipschitz_as_jac_norm" class="proposition">

**Proposition 1** (Alternative definition ). *Let function $`f: \mathbb R^d \mapsto \mathbb R^K`$, be defined on some domain $`dom(f) \subseteq \mathbb R^d`$. Let $`f`$ also be differentiable and $`C`$-Lipschitz continuous. Then the Lipschitz constant $`C`$ is given by: $`C = \sup_{\mathbf x\in dom(f)}\| \nabla_\mathbf xf \|_{\tilde \alpha}\,,`$ where $`\nabla_\mathbf xf`$ is the Jacobian of $`f`$ w.r.t. to input $`\mathbf x`$. $`\tilde \alpha`$ is the dual norm of $`\alpha`$ if $`K=1`$, otherwise $`\tilde \alpha=\alpha`$.*

</div>

**Lower bound via local Lipschitz continuity.** Instead of considering the Lipschitz constant over the entire input domain $`dom(f)`$, we can restrict ourselves to the subspace of the domain where the underlying data distribution is supported, and its neighbourhood (denoted as $`\mathcal{D}^+`$). Under the i.i.d. (independent and identically distributed) assumption in which we typically operate, this definition of *‘effective Lipschitz constant’* ($`C_{\mathcal{D}^+}`$) carries a direct significance. Moreover, this also provides a natural way to lower bound the effective Lipschitz constant based on a finite-sample evaluation, like on the training set $`S\subseteq\mathcal{D}^+`$:
``` math
\label{eq:lower_bound}
    \vspace{-2mm}
    C\geq C_{\mathcal{D}^+} = \sup_{\mathbf x\in \mathcal{D}^+}\| \nabla_\mathbf xf_{{\pmb \theta}}\|_2 \, \geq \, \sup_{\mathbf x\in \mathcal{D}}\| \nabla_\mathbf xf_{{\pmb \theta}}\|_2   =:  C_{\text{lower}}
```
An alternative approach would be to use a straightforward Lipschitz computation: $`C_{\text{alt. lower}} := \sup_{\mathbf x,\mathbf y\in \mathcal{D}^+,\mathbf x\ne\mathbf y} \frac{\|f_{{\pmb \theta}}(\mathbf x)-f_{{\pmb \theta}}(\mathbf y)\|_2}{\|\mathbf x-\mathbf y\|_2}`$. For the rest of the paper, we will focus on computing equation <a href="#eq:lower_bound" data-reference-type="ref" data-reference="eq:lower_bound">[eq:lower_bound]</a>, as $`C_{\text{alt. lower}}`$ requires $`O(n^2)`$ evaluations, and, as shown in Appendix <a href="#sec:straightforward-vs-jacnorm" data-reference-type="ref" data-reference="sec:straightforward-vs-jacnorm">2.1</a>, results in worse estimates.

We also track $`C_\text{avg\_norm}=\mathbb{E}_{\mathbf x\in \mathcal{D}}\, \| \nabla_\mathbf xf_{{\pmb \theta}}\|_2`$, i.e., the expected value of the Jacobian norm, to better contextualise recent work. For instance, this quantity is equivalent to Geometric Complexity, introduced by . Yet, this can be much less than the considered lower bound estimate $`C_{\text{lower}}`$. Thus, on its own, it is insufficient to establish claims on the Lipschitz constant.

**Upper bound via product of layer-wise upper bounds.** Tight upper bounds to the Lipschitz constant, however, are far from straightforward to compute, as can be seen in works mentioned in Section <a href="#related_work" data-reference-type="ref" data-reference="related_work">6</a>. Thus, as expected, we use the simplest upper bound, which is just the product of per-layer Lipschitz constants. Let us assume we have a network $`f_{{\pmb \theta}}`$ with $`L`$ layers, $`1`$-Lipschitz non-linearity $`\sigma`$, such as ReLU, and parameters $`{\pmb \theta}\in\mathbb R^p`$. The overall function can be then expressed as $`f_{{\pmb \theta}}:= f^{(L)} \circ \;\sigma\; \circ f^{(L-1)} \circ \;\sigma \;\circ \;\cdots \; \circ f^{(1)}`$. We can then upper bound the Lipschitz constant:
``` math
\label{eq:lip-upper-bound}
    \vspace{-2mm}
    C \, \le\,  \prod_{i=1}^L \, \sup_{{\mathbf x^{(i-1)}}\in dom(f^{(i)})}\|\nabla_{\mathbf x^{(i-1)}} f^{(i)}\| \le \prod_{i=1}^L \sup \|\nabla_{\mathbf x^{(i-1)}} f^{(i)}\| =: C_{\text{upper}}\,,
```
where $`\mathbf x^{(i-1)}`$ denotes the input at the layer $`i`$, i.e., the post-activation at layer $`i-1`$. In the above equation, we just used 1-Lipschitzness of the non-linearity and then considered an unconstrained supremum (see more details in Appendix <a href="#upper-bound-calculation" data-reference-type="ref" data-reference="upper-bound-calculation">2.2</a>). As a simple example, take the case of a single linear layer $`f^{(1)}(\mathbf x) = \mathbf W^{(1)}\, \mathbf x`$ — the upper bound to the Lipschitz constant is clearly equal to the spectral norm of the weight matrix, i.e., $`C_{\text{upper}}=\|\mathbf W^{(1)}\|_2`$. Similarly, for an $`L`$-layer network, the upper bound comes out to be the product of the spectral norms of the weight matrices: $`C_{\text{upper}}=\prod_{i=1}^L\|\mathbf W^{(i)}\|_2`$.

### Insights into the nature of the Lipschitz constant

#### Does the Lipschitz constant deviate significantly from initialisation? What is its evolution really like? 

For sufficiently wide neural networks, recent theoretical works often assume that the network function and its linearised version[^2] around the initialisation  behave similar enough, while being identical in the limit of infinite-width . While it is still not entirely clear how close today’s massively over-parameterised networks actually approach this limit, it is widely acknowledged that such a limit does not adequately capture feature learning . So, it is not apparent how much is the eventual Lipschitz constant of the network determined by that at initialisation. Moreover, if it does deviate significantly, does it in fact decrease or increase, or something completely else takes place? These are the kinds of questions we would like to settle in this section.

**Evolution of the Lipschitz constant.** Let us start simple by first exploring how the Lipschitz constant evolves when training a fully-connected network (FCN) with ReLU activations (FCN ReLU for short) using cross-entropy (CE) loss. Figure <a href="#fig:lip_bounds_evolution" data-reference-type="ref" data-reference="fig:lip_bounds_evolution">2</a> shows this trend for a pair of networks with moderate and extreme hidden layer widths (i.e., $`256`$ and $`65{,}536`$ respectively). We can observe that both the $`C_\text{lower}`$ and the $`C_\text{upper}`$ keep increasing as the training proceeds. In particular, while they start from similar values at initialisation, they rapidly drift apart from each other. We explore this effect through the prism of network linearisation in Appendix <a href="#nn-linearisation" data-reference-type="ref" data-reference="nn-linearisation">3.1</a>.

<figure id="fig:lip_bounds_evolution">
<figure id="fig:lip_bounds_tightness">
<p><img src="./figures/lip_bounds_tightness_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_0.005_Warmup20000Step25.png"" style="width:95.0%" /> <span id="fig:lip_bounds_tightness" data-label="fig:lip_bounds_tightness"></span></p>
</figure>
<figure>
<img src="./figures/lip_bounds_evolution_MNIST1D_FF_ReLU_65536_CrossEntropy_SGD_0.005_Warmup20000Step25.png"" style="width:95.0%" />
</figure>
<figcaption>Plot of Lipschitz constant bounds by training epoch for <strong>FCN ReLU network</strong> with hidden layer widths <span class="math inline">256</span> (<strong>left</strong>) and <span class="math inline">65, 536</span> (<strong>right</strong>) on <span class="smallcaps">MNIST1D</span>. <span style="color: mydarkred"><span class="math inline"><em>C</em><sub>upper</sub></span></span>, <span style="color: myblue"><span class="math inline"><em>C</em><sub>lower</sub></span></span> and <span style="color: gray"><span class="math inline"><em>C</em><sub>avg_norm</sub></span></span> are computed on train dataset <span class="math inline">𝒟</span>, whereas <span style="color: mylightblue"><span class="math inline"><em>C</em><sub><em>S</em><sup>*</sup></sub></span></span> is the local Lipschitz computed on the <span class="math inline"><em>S</em><sup>*</sup></span>. Relative to initialisation, the lower bound at convergence grows by a factor <span class="math inline">63×</span>, <span class="math inline">40×</span>, while the upper bound by <span class="math inline">66×, 10×</span>, for the widths <span class="math inline">256; 65, 536</span> respectively. Results are averaged over 4 runs. See Appendix <a href="#setup-convex-combinations" data-reference-type="ref" data-reference="setup-convex-combinations">2.6.8</a>.</figcaption>
</figure>

The Lipschitz behaviour seems to saturate at some point sufficiently further into the training process, but relative to the initialisation, the lower and the upper bound grow, for both widths, by a significant factor. This strictly increasing behaviour of the Lipschitz might be attributed to the nature of the CE loss, for which convergence requires taking the parameters norms to infinity. But, as a matter of fact, we find that similar behaviour can be seen in the case of MSE loss in Figure <a href="#fig:lip-bounds-evolution-mnist1d-mse" data-reference-type="ref" data-reference="fig:lip-bounds-evolution-mnist1d-mse">19</a>. So, while the much wider network has a smaller increase in Lipschitz, the growth is significant. Thus, it seems that even at such high widths, the network deviates far enough from its initial state, and the eventual Lipschitz constant overshadows the one at initialisation.

**Fidelity of the lower bound to the effective Lipschitz constant.** <span id="fidelity-of-the-lower-bound" label="fidelity-of-the-lower-bound"></span> Next, as the effective Lipschitz constant lies somewhere between the upper and the lower bounds, what still remains unclear is how loose are these bounds and where our quantity of interest lies. To give more insight into this, we compute the lower Lipschitz bound equation <a href="#eq:lower_bound" data-reference-type="ref" data-reference="eq:lower_bound">[eq:lower_bound]</a> on a larger set of examples $`S^*`$, which is the union of the training set $`\mathcal{D}`$, *the test set* $`\mathcal{D_{\mathrm{test}}}`$, and a set of *random convex combinations* of samples from the train and test sets (see <a href="#setup-convex-combinations" data-reference-type="ref" data-reference="setup-convex-combinations">2.6.8</a> for more details). The results are presented in Figure <a href="#fig:lip_bounds_evolution" data-reference-type="ref" data-reference="fig:lip_bounds_evolution">2</a>, corresponding to the legend **<span style="color: mylightblue">$`C_{S^*}`$</span>**, where it becomes evident that the bound for the Lipschitz constant computed on this expanded set $`S^*`$ lies much closer to the lower bound **<span style="color: myblue">$`C_\text{lower}`$</span>** than the upper bound **<span style="color: mydarkred">$`C_\text{upper}`$</span>**. While the gap between the upper and the lower bound may not seem that drastic here, the upper bound can quickly become extremely large for deeper models (c.f., Figure <a href="#fig:resnet50-evolution" data-reference-type="ref" data-reference="fig:resnet50-evolution">[fig:resnet50-evolution]</a> and Appendix <a href="#lipschitz_evolution_other_settings" data-reference-type="ref" data-reference="lipschitz_evolution_other_settings">3.8</a>), leading to a significant overestimation of the Lipschitz constant. In fact, this aspect about the fidelity of the lower bound can also be spotted in some past works , although it does not get a proper discussion. Overall, this seems to suggest that *the trend of effective Lipschitz constant is more faithfully captured by the lower bound $`C_\text{lower}`$, contrary to the upper bound $`C_\text{upper}`$*.

**A theoretical picture.** To understand the increasing Lipschitz trends analytically, we develop an upper bound on the Lipschitz constant in Appendix <a href="#power-law-trend-in-lipschitz-evolution" data-reference-type="ref" data-reference="power-law-trend-in-lipschitz-evolution">4.1</a>. In short, given SGD updates with a constant learning rate $`\eta\in(0, \infty)`$ and assuming the loss gradients are bounded by $`B\in(0, \infty)`$ (which could be enforced by gradient clipping, for instance), we get the that the upper bound on the Lipschitz constant $`C_T`$ at time step $`T`$ follows the trend $`C_T \propto B\eta T`$. Although this bound only describes the behaviour of the upper bound to the Lipschitz constant, the analysis still shows the undeniable effect of training on the function’s Lipschitz continuity.

#### What happens for bigger network-dataset settings?

Given the above experiments were based on simple network-dataset settings — primarily to allow us to elucidate the point about the Lipschitz of an extremely wide Neural Network and its deviation from initialisation, we now test whether the trend about the growth of Lipschitz persists for more modern over-parametrised networks on bigger datasets as trained in practice.

To thoroughly establish whether our results carry over for networks with millions of parameters, we take the case of <span class="smallcaps">ResNet50</span> and a <span class="smallcaps">Vision Transformer (ViT)</span> trained on <span class="smallcaps">ImageNet</span>. Note that we are forced to slightly restrict the computation of the Lipschitz estimates to a smaller number of checkpoints during training, as well as the number of samples. In brief, this is due to the high computational complexity [^3] involved in the process of evaluating the Jacobian norms of a significant number of large matrices. The results for these experiments can be found in Figures <a href="#fig:resnet50-evolution" data-reference-type="ref" data-reference="fig:resnet50-evolution">[fig:resnet50-evolution]</a>, <a href="#fig:lip-evol-vit-imagenet" data-reference-type="ref" data-reference="fig:lip-evol-vit-imagenet">3</a>. As ViTs do not possess a theoretical upper bound  due to the presence of quadratic interactions in input and attention layers, only local Lipschitz-based estimates are presented in Figure <a href="#fig:lip-evol-vit-imagenet" data-reference-type="ref" data-reference="fig:lip-evol-vit-imagenet">3</a>.

<figure id="fig:lip-evol-vit-imagenet">
<div class="minipage">
<img src="./figures/lip_evolution_resnet50_loglog_True.png"" style="width:95.0%" />
</div>
<div class="minipage">
<img src="./figures/lip_evolution_vit_resnet_imagenet_loglog_True.png"" style="width:95.0%" />
</div>
<figcaption>Lower Lipschitz constant bounds evolution for ViT on a <span class="math inline">50, 000</span> samples ImageNet subset. More details in Appendix <a href="#sec:vit-evolution" data-reference-type="ref" data-reference="sec:vit-evolution">3.5</a>. </figcaption>
</figure>

The first thing that catches our eye is that the upper bound gets almost vacuously large, with the gap between the lower and upper bounds, for instance in the case of ResNet50 in Figure <a href="#fig:resnet50-evolution" data-reference-type="ref" data-reference="fig:resnet50-evolution">[fig:resnet50-evolution]</a>, stretching to $`\sim40`$ orders of magnitude (mind the different y-axes). This suggests that upper bounds are excessively lax (even at initialisation itself, it is of the order $`1e^{27}`$, which can be attributed to the exponential increase with depth). It seems more plausible that the values of the (local Lipschitz) based lower bound are more revelatory about the effective Lipschitz, or the nature of the function in general.

<div class="wrapfigure">

r0.5 <img src="./figures/ResNet18s_norms_distr_ImageNet_and_CC.png"" />

</div>

Given the extremely high values of the upper bound, it can be argued whether our computed lower bounds are still representative of the effective Lipschitz. To investigate this, we take the top $`1,000`$ samples that have the highest Jacobian norm, consider their convex combinations, and use that as a basis to evaluate the local Lipschitz. In fact, in Figure <a href="#fig:norm-distr-imagenet-resnet18" data-reference-type="ref" data-reference="fig:norm-distr-imagenet-resnet18">18</a>, we show the entire distribution of norms for these ‘hard’ convex combinations, as well as the entire ImageNet train set (See Appendix <a href="#setup-convex-combinations-resnet18" data-reference-type="ref" data-reference="setup-convex-combinations-resnet18">2.6.10</a>).

We find that while the distribution indeed shifts towards larger per-sample Jacobian norms for the hard convex combinations, the shift is not even a multiplicative factor of $`2\times`$ more. This shift pales in comparison to the upper bound which is over tens of orders of magnitudes higher. Overall, this strengthens our claim that the lower bound is much more faithful to the effective Lipschitz value and can hence serve better to explore various phenomena observed in over-parameterised Neural Networks. Lastly, we leave similar distribution plots for other models and datasets in Appendix <a href="#jacobian-norm-distributions-other-models-datasets" data-reference-type="ref" data-reference="jacobian-norm-distributions-other-models-datasets">3.15</a>.

#### Gathering intuition from a Toy example

In the previous discussion, we have seen substantial evidence for the fidelity of the lower bound to the effective Lipschitz constant. In order to gather some intuition, we now consider a toy example, that gets us to the essence of our discovered findings and enhances our understanding of them.

We take a synthetic dataset on a two-dimensional closed set, for example $`\mathcal D = [-5,5]^2`$, with 3 equally spaced classes generated by a Gaussian distribution. Due to the close proximity of the classes relative to the standard deviation of the distribution, some training points may lie in a region of another class. We now train an FCN ReLU classifier until 100% train accuracy and compute the lower and upper Lipschitz estimates, which come out to be $`135.815`$ and $`389.097`$, respectively. Thanks to the designed setup here, however, we can tractably compute an estimate of the effective Lipschitz constant $`C_{\mathcal {D}^+}`$ by densely sampling 1 million points in the entire input domain and computing the maximum over the point-wise Jacobian norms. *The estimate evaluates to $`144.194`$, which is rather close to the $`C_\text{lower}`$ estimate*. In fact, we can see the reason for this in Figure <a href="#fig:visual-example" data-reference-type="ref" data-reference="fig:visual-example">4</a> — the *highest function change occurs near the decision boundary*, where, in turn, the local Lipschitz constant is the highest. Since the decision boundary lies in the vicinity of the training samples (this is supported by the existence of adversarial examples ), the lower bound manages to capture the effective Lipschitz constant quite well.

<figure id="fig:visual-example">
<figure>
<img src="./figures/visual_example_pred.png"" />
</figure>
<figure>
<img src="./figures/visual_example_lower_lip.png"" />
</figure>
<figcaption>Plot of function prediction <strong>(left)</strong> and the local Lipschitz constant bounds <strong>(right)</strong> for the whole input domain <span class="math inline">𝒟 = [−5, 5]<sup>2</sup></span>. More details in Appendix <a href="#setup-visual-example" data-reference-type="ref" data-reference="setup-visual-example">2.6.1</a>.</figcaption>
</figure>

*Adversarial and out-of-distribution (OOD) samples.* Guided by the above intuition, we also evaluate the lower bound on adversarially perturbed training samples for CIFAR-10 and OOD samples for MNIST1D. As we show in detail in Appendix <a href="#adversarial-lower-lipschitz" data-reference-type="ref" data-reference="adversarial-lower-lipschitz">3.2</a> and <a href="#dd-ood" data-reference-type="ref" data-reference="dd-ood">3.3</a>, this evaluation only marginally improves the lower bound which does not quite justify the added computational burden — if an efficient Lipschitz estimate is the focus.

### Implicit Lipschitz regularisation, or Lipschitz Double Descent

<div class="wrapfigure">

r0.35 <img src="./figures/three_modes.png"" style="width:85.0%" />

</div>

A commonly held view  about the effectiveness of heavily over-parameterised neural networks, in the context of generalisation on unseen, is that with increasing parameter count the network, in tandem with a simple optimisation algorithm like stochastic gradient descent (SGD), finds a solution where the extra capacity helps towards fitting the training samples smoothly. In contrast, with just as many parameters $`p`$ as the number of training samples $`n`$, the smoothness of the interpolation is not within control and we observe worse generalisation. And, with parameters less than that, i.e., $`p<n`$, the network is even unable to fully fit the training samples. Figure <a href="#fig:three_modes" data-reference-type="ref" data-reference="fig:three_modes">[fig:three_modes]</a> sketches the general shape of the model in these regimes.

More concretely, this view has garnered significant evidence through the occurrence of what is known as the Double Descent (DD) phenomenon . In particular, with increasing network capacity (say, via layer width) the test loss first decreases [^4], then increases up to a certain threshold of capacity (known as the interpolation threshold, where $`p\approx n`$), beyond which it further decreases. The theoretical works supporting double descent show — usually in fairly simplified settings — that the test loss exhibits such a behaviour. However, it largely remains difficult to attribute or pinpoint the behaviour of test loss to a core functional property of the network.

Given the above discussion about the smoothness of interpolating over-parameterised solutions, the Lipschitz constant forms a natural candidate for a functional property that signifies smoothness, and we, thus, investigate if its trend with network width has similarities to DD. In Figure <a href="#fig:lip_dd_cnn_cifar100c20" data-reference-type="ref" data-reference="fig:lip_dd_cnn_cifar100c20">5</a>, we conduct an experiment replicating the DD setup, by training Convolutional Neural Networks (CNNs) of increasing width (i.e., number of channels) to convergence on <span class="smallcaps">CIFAR-100</span>. We notice the plot clearly shows how all three Lipschitz constant bounds grow until the interpolation threshold and decrease afterwards, while also mirroring the trend in the test loss.

<figure id="fig:lip_dd_cnn_cifar100c20">
<figure>
<img src="./figures/double_descent_lip%2BCIFAR100c20%2BCNN_X%2BCrossEntropy%2BSGD_0.005_Cont100%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figure>
<img src="./figures/double_descent_train_test%2BCIFAR100c20%2BCNN_X%2BCrossEntropy%2BSGD_0.005_Cont100%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing network width, for <strong>CNN networks</strong> trained on <strong>CIFAR-100</strong> with CE loss. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-cifar100c20" data-reference-type="ref" data-reference="setup-dd-cifar100c20">2.6.5</a>.</figcaption>
</figure>

Similar trends in the Lipschitz constant can also be seen for other network-dataset settings (FCNs, ViTs; MNIST1D, MNIST, CIFAR-10), as well as the MSE loss, the results for which are located in Appendix <a href="#lipschitz_double_descent_other_settings" data-reference-type="ref" data-reference="lipschitz_double_descent_other_settings">3.9</a>.

**Implicit Lipschitz regularisation.** While a significant number of works in the literature specifically set out to design new and more convenient ways to explicitly regularise the Lipschitz constant during training  or designing architectures with Lipschitz guarantees , the above finding highlights that, even without such explicit controls, *over-parameterisation seems to provide an implicit pressure towards Lipschitz regularisation.* This could also potentially hint why, despite copious work, the direction of explicit Lipschitz regularisation has seen a relatively muted practical impact and adoption, barring certain specialised areas . Although the strength of this implicit Lipschitz regularisation is likely problem and architecture dependent, adding explicit regularisation (here enforcing Lipschitzness), as observed in usual DD settings , can indeed reduce the DD trend in both loss and the Lipschitz bounds (see Appendix <a href="#dd-lip-constr" data-reference-type="ref" data-reference="dd-lip-constr">3.12</a>).

Besides, the above results provide additional backing for the intuition expounded around the smoothness of interpolating solutions by . Lastly, this also restores hope for the established generalisation bounds based on the Lipschitz constant  and potentially reworking them on the basis of the effective Lipschitz constant, similar to .

**A bias-variance trade-off argument.** <span id="bias-variance-tradeoff-argument" label="bias-variance-tradeoff-argument"></span> Understanding the exact mechanisms of the Lipschitz DD from a more theoretical avenue would form a great direction. While this would stretch far beyond the current scope, we nevertheless supplement our discussion with a theoretical argument that connects the Lipschitz behaviour with the test loss as seen above.

<div class="wrapfigure">

r0.45 <img src="./figures/bounds_verif%2BMNIST1D%2BLeakyOutput_FF_ReLU%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Btarget_0.01.png"" />

</div>

The details of the analysis can be found in Appendix <a href="#bias-variance-tradeoff-argument-extended" data-reference-type="ref" data-reference="bias-variance-tradeoff-argument-extended">4.2</a>, but let’s present the gist of the argument here. We define a Neural Network function $`f_{{\pmb \theta}}(\mathbf x, \zeta)`$, where $`\zeta`$ indicates the noise in the function due to the choice of random initialisation (i.e. random seed). We can then bound the variance term in the test loss as follows (<a href="#eq:var_upper:1" data-reference-type="ref" data-reference="eq:var_upper:1">[eq:var_upper:1]</a>): $`\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf x, \zeta)) \leq 3\, (\overline{C} ^2+\overline{C_{\zeta}}^2 )\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \|\mathbf x\|^2 \, + \, \text{const}\,,`$ where $`\overline {C_\zeta}`$ is the mean Lipschitz constant across seeds and the Lipschitz constant $`\overline C`$ of the ensembled function $`\overline f_{{\pmb \theta}}(\cdot) = \mathbb{E}_\zeta[f_{{\pmb \theta}}(\cdot, \zeta)]`$. Figure <a href="#fig:bias-var-tradeoff" data-reference-type="ref" data-reference="fig:bias-var-tradeoff">[fig:bias-var-tradeoff]</a> presents an empirical calculation for the variance bounds (results for the bias term can be found in Appendix <a href="#variance_upper_bounds" data-reference-type="ref" data-reference="variance_upper_bounds">3.11</a>), where the trend of the function variance aligns closely with the trend of our upper bounds, at least qualitatively. Since the above theoretical analysis only establishes an upper bound on the function variance with the Lipschitz constant, it would be very interesting to develop a corresponding lower bound that shows the dependence on the Lipschitz constant, but that is beyond our current confines.

### Noise, Capacity, and Over-Fitting

emphatically show that the over-parameterised Neural Networks have the ability to completely memorise points when trained with random labels, but these are the same networks that also generalise when presented with clean data. However, they consider this in regard to a network with fixed capacity and a certain amount of label noise. Now imagine that we keep increasing the proportion of randomly labelled points while keeping the network fixed. Will the Lipschitz keep increasing monotonically? Will the behaviour be the same if we had a bigger or a smaller network? Introducing randomised labels should make the task harder , but does it matter as long as the network has more parameters than training samples? In this section, we explore this interplay of the noise strength, network capacity, and the resulting level of generalisation.

<div class="wrapfigure">

r0.45 <img src="./figures/label_noise.png"" style="width:85.0%" />

<span id="fig:hypothesis-label-noise" label="fig:hypothesis-label-noise"></span>

</div>

**Thought experiment.** Consider a task, where labels $`y=1`$ if $`x > 0`$ and $`y=-1`$ otherwise, and $`x`$ is sampled uniformly on the reals in a certain range. Then, consider the case of an extreme amount of label noise (perhaps relative to network capacity), so that we effectively sample $`y=\pm 1`$ without looking at the value of $`x`$. In such a case, the best a network can do is to just predict $`\hat{y}=0,\; \forall \,\mathbf x`$. But for such a prediction, the learned function is as smooth as it gets, with a Lipschitz constant of $`0`$.

**A hypothesis for Lipschitz behaviour with label noise.** Based on the intuition above, we would hypothesise that *while the network is able to fit the noise, the Lipschitz constant should increase*. At some point, when the noise strength becomes sufficiently high relative to the network capacity, we would reach a ‘memorisation threshold’, beyond which the predictor starts collapsing to a smoother function with a smaller Lipschitz constant. We depict our hypothesis pictorially in Figure <a href="#fig:hypothesis-label-noise" data-reference-type="ref" data-reference="fig:hypothesis-label-noise">[fig:hypothesis-label-noise]</a>.

**Empirical evidence.** To test this hypothesis, we train a bunch of CNNs on CIFAR-10 with increasing width and label shuffling levels from 0% (clean) to 100% shuffled (fully random) targets. The results can be found in the Figure <a href="#fig:label-noise-dd" data-reference-type="ref" data-reference="fig:label-noise-dd">8</a>. (i) Firstly, looking at the rows, we notice that for every noise strength, the Lipschitz constant shows a Double-Descent-like non-monotonicity with increasing width (which can perhaps be expected for low noise strengths, though not all). (ii) But, more interestingly, if we look at the columns, we find that there is a similar non-monotonicity, in line with our hypothesis above. (iii) There is a shift of the memorisation threshold, which matches the movement of the interpolation threshold in Double Descent, towards a higher parameter count in the presence of label noise . (iv) Lastly, heavily over-parameterised networks fit random labels more smoothly, compared to networks with fewer parameters.

Overall, we see a very intriguing interplay of noise, capacity, and memorisation on the Lipschitz behaviour, which highlights, quite uniquely, the intriguing benefits imparted by over-parameterisation.

<figure id="fig:label-noise-dd">
<figure id="fig:label-noise-dd-lower-lip">
<p><img src="./figures/label_shuffling_dd_lower_CIFAR10.png"" style="width:90.0%" /> <span id="fig:label-noise-dd-lower-lip" data-label="fig:label-noise-dd-lower-lip"></span></p>
</figure>
<figure id="fig:label-noise-dd-test-loss">
<p><img src="./figures/label_shuffling_dd_test_loss_CIFAR10.png"" style="width:90.0%" /> <span id="fig:label-noise-dd-test-loss" data-label="fig:label-noise-dd-test-loss"></span></p>
</figure>
<figcaption>Lower Lipschitz values (<strong>left</strong>) and the test loss (<strong>right</strong>) for CNN models at various levels of label shuffling, trained on CIFAR-10. More details are in Appendix <a href="#setup-effect-of-label-noise" data-reference-type="ref" data-reference="setup-effect-of-label-noise">2.6.16</a>.</figcaption>
</figure>

### Related Work

**Theoretical works on the Lipschitz constant.** Recent theoretical interest in the Lipschitz constant of Neural Networks has been revived since  described how margin-based generalisation bounds are linearly dependent on the Lipschitz constant. Other Lipschitz constant-based generalisation bounds  were later developed. have also conjectured the phenomena of smooth interpolation in over-parametrised networks with the underlying Lipschitz continuity of the network. Since generalisation bounds and robustness guarantees  are upper bounded by an expression containing the true Lipschitz, extensive research has been done in the field of its accurate estimation , but this is still an ongoing pursuit as more accurate estimates typically come at high computational costs.

**Practical applications.** On the practical side, this direction can be divided into three sub-categories. *(i) Generative modelling:* The Lipschitz constant has been utilised to stabilise Recurrent Neural Networks  and Generative Adversarial Networks (GANs) . *(ii) Certificates for adversarial robustness:* Enforcing certain Lipschitz constraints has proved useful in certifying robustness guarantees for fully-connected Neural Networks  and Convolutional Networks , as well as in other areas like Equilibrium Networks . In order to certify certain Lipschitz values, one can simply modify internal layers so that the resulting network is $`C`$-Lipschitz , or use a more general parametrisation . *(iii) Lipschitz as a regularisation:* Another way to ensure model robustness is through Lipschitz-based regularisation techniques . Yet, as we have seen in the DD section, there is an inherent Lipschitz regularisation already at play in Neural Networks.

**Recent works with similar focus.** Concurrent with the release of our work, have also noted the connection between Lipschitz constant and Double Descent , albeit by tracking only an estimate of the Lipschitz. Likewise, in the context of Double Descent and evolution during training, explored a quantity called Geometric Complexity, which is similar to our notion of the $`C_\text{avg\_norm}`$ and the estimate of . But, as a matter of fact, such estimates are even looser than the local Lipschitz lower bounds, as can be seen in our results. Whereas, by considering the lower and upper Lipschitz bounds simultaneously, our results reveal the behaviour of the effective Lipschitz more faithfully. Besides, our focus is much more comprehensive as, for instance, we elaborate on the nature of effective Lipschitz (on unprecedented evaluations on ImageNet with ResNet50 and ViT) and uncover new intriguing insights into the Lipschitz behaviour in high-noise regimes.

### Conclusion

**Summary.** In this work, we have presented a comprehensive study of some intriguing behaviours of the Lipschitz constant of Neural Networks. We first explored the nature of the effective Lipschitz constant, showcasing the evolution study (and its marked deviation from the value at initialisation) and the fidelity of the lower bound. Then, we witnessed the implicit Lipschitz regularisation effect with width in the form of Double-Descent-like non-monotonicity in the Lipschitz bounds. Finally, we examined the effect of label noise on function smoothness and generalisation, spanning across a wide range of network capacities and noise strengths.

**What we could not touch upon.** In Appendix <a href="#add_experiments" data-reference-type="ref" data-reference="add_experiments">3</a>, we replicate most of the experiments shown in the main part for other model classes and datasets, ensuring the consistency of presented results. Additionally, we also discuss the effect of the choice of loss (between Cross-Entropy and MSE), optimisation algorithm (SGD versus Adam), depth of the network, explicit regularisers such as weight decay and dropout, and the number of training samples.

**Limitations & future research directions.** One potential avenue for investigation would be to provide a detailed theoretical analysis for our uncovered findings which, although fell beyond the current scope, would be highly relevant (e.g. generalisation bounds based on the effective Lipschitz). It would also be rather interesting to explore the effect of large learning rates through the lens of the effective Lipschitz constant. Lastly, examining our findings outside of computer vision tasks — e.g., in the language domain, or in the framework of reinforcement learning would be a fruitful endeavor.

All in all, we hope that this work inspires further research on uncovering and understanding the characteristics of the Lipschitz constant.

### Acknowledgements

We would like to thank Thomas Hofmann, Bernhard Schölkopf, and Aurelien Lucchi for their useful comments and suggestions. We are also grateful to the members of the DALab for their support. Sidak Pal Singh would like to acknowledge the financial support from Max Planck ETH Center for Learning Systems.

### References

<div class="thebibliography">

Ben Adlam and Jeffrey Pennington 2020. **Abstract:** Modern deep learning models employ considerably more parameters than required to fit the training data. Whereas conventional statistical wisdom suggests such models should drastically overfit, in practice these models generalize remarkably well. An emerging paradigm for describing this unexpected behavior is in terms of a \\}emph{double descent} curve, in which increasing a model’s capacity causes its test error to first decrease, then increase to a maximum near the interpolation threshold, and then decrease again in the overparameterized regime. Recent efforts to explain this phenomenon theoretically have focused on simple settings, such as linear regression or kernel regression with unstructured random features, which we argue are too coarse to reveal important nuances of actual neural networks. We provide a precise high-dimensional asymptotic analysis of generalization under kernel regression with the Neural Tangent Kernel, which characterizes the behavior of wide neural networks optimized with gradient descent. Our results reveal that the test error has non-monotonic behavior deep in the overparameterized regime and can even exhibit additional peaks and descents when the number of parameters scales quadratically with the dataset size. (@adlam2020neural)

Cem Anil, James Lucas, and Roger Baker Grosse In *International Conference on Machine Learning*, 2018. **Abstract:** Training neural networks under a strict Lipschitz constraint is useful for provable adversarial robustness, generalization bounds, interpretable gradients, and Wasserstein distance estimation. By the composition property of Lipschitz functions, it suffices to ensure that each individual affine transformation or nonlinear activation is 1-Lipschitz. The challenge is to do this while maintaining the expressive power. We identify a necessary property for such an architecture: each of the layers must preserve the gradient norm during backpropagation. Based on this, we propose to combine a gradient norm preserving activation function, GroupSort, with norm-constrained weight matrices. We show that norm-constrained GroupSort architectures are universal Lipschitz function approximators. Empirically, we show that norm-constrained GroupSort networks achieve tighter estimates of Wasserstein distance than their ReLU counterparts and can achieve provable adversarial robustness guarantees with little cost to accuracy. (@Anil2018SortingOL)

Alexandre Araujo, Aaron J. Havens, Blaise Delattre, Alexandre Allauzen, and Bin Hu In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL <https://openreview.net/pdf?id=k71IGLC8cfc>. **Abstract:** Important research efforts have focused on the design and training of neural networks with a controlled Lipschitz constant. The goal is to increase and sometimes guarantee the robustness against adversarial attacks. Recent promising techniques draw inspirations from different backgrounds to design 1-Lipschitz neural networks, just to name a few: convex potential layers derive from the discretization of continuous dynamical systems, Almost-Orthogonal-Layer proposes a tailored method for matrix rescaling. However, it is today important to consider the recent and promising contributions in the field under a common theoretical lens to better design new and improved layers. This paper introduces a novel algebraic perspective unifying various types of 1-Lipschitz neural networks, including the ones previously mentioned, along with methods based on orthogonality and spectral methods. Interestingly, we show that many existing techniques can be derived and generalized via finding analytical solutions of a common semidefinite programming (SDP) condition. We also prove that AOL biases the scaled weight to the ones which are close to the set of orthogonal matrices in a certain mathematical manner. Moreover, our algebraic condition, combined with the Gershgorin circle theorem, readily leads to new and diverse parameterizations for 1-Lipschitz network layers. Our approach, called SDP-based Lipschitz Layers (SLL), allows us to design non-trivial yet efficient generalization of convex potential layers. Finally, the comprehensive set of experiments on image classification shows that SLLs outperform previous approaches on certified robust accuracy. Code is available at https://github.com/araujoalexandre/Lipschitz-SLL-Networks. (@Araujo2023AUA)

Martı́n Arjovsky, Soumith Chintala, and Léon Bottou Wasserstein GAN *CoRR*, abs/1701.07875, 2017. URL <http://arxiv.org/abs/1701.07875>. **Abstract:** We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions. (@arjovsky2017wasserstein)

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang On exact computation with an infinitely wide neural net *Advances in neural information processing systems*, 32, 2019. **Abstract:** How well does a classic deep net architecture like AlexNet or VGG19 classify on a standard dataset such as CIFAR-10 when its “width”— namely, number of channels in convolutional layers, and number of nodes in fully-connected internal layers — is allowed to increase to infinity? Such questions have come to the forefront in the quest to theoretically understand deep learning and its mysteries about optimization and generalization. They also connect deep learning to notions such as Gaussian processes and kernels. A recent paper \[Jacot et al., 2018\] introduced the Neural Tangent Kernel (NTK) which captures the behavior of fully-connected deep nets in the infinite width limit trained by gradient descent; this object was implicit in some other recent papers. An attraction of such ideas is that a pure kernel-based method is used to capture the power of a fully-trained deep net of infinite width. The current paper gives the first efficient exact algorithm for computing the extension of NTK to convolutional neural nets, which we call Convolutional NTK (CNTK), as well as an efficient GPU implementation of this algorithm. This results in a significant new benchmark for performance of a pure kernel-based method on CIFAR-10, being 10% higher than the methods reported in \[Novak et al., 2019\], and only 6% lower than the performance of the corresponding finite deep net architecture (once batch normalization etc. are turned off). Theoretically, we also give the first non-asymptotic proof showing that a fully-trained sufficiently wide net is indeed equivalent to the kernel regression predictor using NTK. (@arora2019exact)

Peter L. Bartlett, Dylan J. Foster, and Matus Telgarsky Spectrally-normalized margin bounds for neural networks In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pp. 6240–6249, 2017. URL <https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html>. **Abstract:** This paper presents a margin-based multiclass generalization bound for neural networks that scales with their margin-normalized "spectral complexity": their Lipschitz constant, meaning the product of the spectral norms of the weight matrices, times a certain correction factor. This bound is empirically investigated for a standard AlexNet network trained with SGD on the mnist and cifar10 datasets, with both original and random labels; the bound, the Lipschitz constants, and the excess risks are all in direct correlation, suggesting both that SGD selects predictors whose complexity scales with the difficulty of the learning task, and secondly that the presented bound is sensitive to this complexity. (@bartlett2017spectrallynormalized)

Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler Benign overfitting in linear regression *Proceedings of the National Academy of Sciences*, 117 (48): 30063–30070, 2020. **Abstract:** The phenomenon of benign overfitting is one of the key mysteries uncovered by deep learning methodology: deep neural networks seem to predict well, even with a perfect fit to noisy training data. Motivated by this phenomenon, we consider when a perfect fit to training data in linear regression is compatible with accurate prediction. We give a characterization of linear regression problems for which the minimum norm interpolating prediction rule has near-optimal prediction accuracy. The characterization is in terms of two notions of the effective rank of the data covariance. It shows that overparameterization is essential for benign overfitting in this setting: the number of directions in parameter space that are unimportant for prediction must significantly exceed the sample size. By studying examples of data covariance properties that this characterization shows are required for benign overfitting, we find an important role for finite-dimensional data: the accuracy of the minimum norm interpolating prediction rule approaches the best possible accuracy for a much narrower range of properties of the data distribution when the data lies in an infinite dimensional space versus when the data lies in a finite dimensional space whose dimension grows faster than the sample size. (@bartlett2020benign)

Mikhail Belkin Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation *Acta Numerica*, 30: 203–248, 2021. **Abstract:** In the past decade the mathematical theory of machine learning has lagged far behind the triumphs of deep neural networks on practical challenges. However, the gap between theory and practice is gradually starting to close. In this paper I will attempt to assemble some pieces of the remarkable and still incomplete mathematical mosaic emerging from the efforts to understand the foundations of deep learning. The two key themes will be interpolation and its sibling over-parametrization. Interpolation corresponds to fitting data, even noisy data, exactly. Over-parametrization enables interpolation and provides flexibility to select a suitable interpolating model. As we will see, just as a physical prism separates colours mixed within a ray of light, the figurative prism of interpolation helps to disentangle generalization and optimization properties within the complex picture of modern machine learning. This article is written in the belief and hope that clearer understanding of these issues will bring us a step closer towards a general theory of deep learning and machine learning. (@belkin2021fit)

Mikhail Belkin, Daniel J. Hsu, Siyuan Ma, and Soumik Mandal Reconciling modern machine-learning practice and the classical bias-variance trade-off *Proceedings of the National Academy of Sciences*, 116: 15849 – 15854, 2019. **Abstract:** Breakthroughs in machine learning are rapidly changing science and society, yet our fundamental understanding of this technology has lagged far behind. Indeed, one of the central tenets of the field, the bias-variance trade-off, appears to be at odds with the observed behavior of methods used in modern machine-learning practice. The bias-variance trade-off implies that a model should balance underfitting and overfitting: Rich enough to express underlying structure in data and simple enough to avoid fitting spurious patterns. However, in modern practice, very rich models such as neural networks are trained to exactly fit (i.e., interpolate) the data. Classically, such models would be considered overfitted, and yet they often obtain high accuracy on test data. This apparent contradiction has raised questions about the mathematical foundations of machine learning and their relevance to practitioners. In this paper, we reconcile the classical understanding and the modern practice within a unified performance curve. This "double-descent" curve subsumes the textbook U-shaped bias-variance trade-off curve by showing how increasing model capacity beyond the point of interpolation results in improved performance. We provide evidence for the existence and ubiquity of double descent for a wide spectrum of models and datasets, and we posit a mechanism for its emergence. This connection between the performance and the structure of machine-learning models delineates the limits of classical analyses and has implications for both the theory and the practice of machine learning. (@Belkin2019Reconciling)

Louis Béthune, Thibaut Boissin, Mathieu Serrurier, Franck Mamalet, Corentin Friedrich, and Alberto Gonzalez Sanz *Advances in Neural Information Processing Systems*, 35: 20077–20091, 2022. **Abstract:** Lipschitz constrained networks have gathered considerable attention in the deep learning community, with usages ranging from Wasserstein distance estimation to the training of certifiably robust classifiers. However they remain commonly considered as less accurate, and their properties in learning are still not fully understood. In this paper we clarify the matter: when it comes to classification 1-Lipschitz neural networks enjoy several advantages over their unconstrained counterpart. First, we show that these networks are as accurate as classical ones, and can fit arbitrarily difficult boundaries. Then, relying on a robustness metric that reflects operational needs we characterize the most robust classifier: the WGAN discriminator. Next, we show that 1-Lipschitz neural networks generalize well under milder assumptions. Finally, we show that hyper-parameters of the loss are crucial for controlling the accuracy-robustness trade-off. We conclude that they exhibit appealing properties to pave the way toward provably accurate, and provably robust neural networks. (@Bethune2021PayAT)

Sébastien Bubeck and Mark Sellke *Journal of the ACM*, 70: 1 – 18, 2021. **Abstract:** Classically, data interpolation with a parametrized model class is possible as long as the number of parameters is larger than the number of equations to be satisfied. A puzzling phenomenon in deep learning is that models are trained with many more parameters than what this classical theory would suggest. We propose a partial theoretical explanation for this phenomenon. We prove that for a broad class of data distributions and model classes, overparametrization is necessary if one wants to interpolate the data smoothly . Namely we show that smooth interpolation requires d times more parameters than mere interpolation, where d is the ambient data dimension. We prove this universal law of robustness for any smoothly parametrized function class with polynomial size weights, and any covariate distribution verifying isoperimetry (or a mixture thereof). In the case of two-layer neural networks and Gaussian covariates, this law was conjectured in prior work by Bubeck, Li, and Nagaraj. We also give an interpretation of our result as an improved generalization bound for model classes consisting of smooth functions. (@Bubeck2021AUL)

Leon Bungert, René Raab, Tim Roith, Leo Schwinn, and Daniel Tenbrinck In *Scale Space and Variational Methods in Computer Vision*, 2021. **Abstract:** Despite the large success of deep neural networks (DNN) in recent years, most neural networks still lack mathematical guarantees in terms of stability. For instance, DNNs are vulnerable to small or even imperceptible input perturbations, so called adversarial examples, that can cause false predictions. This instability can have severe consequences in applications which in uence the health and safety of humans, e.g., biomedical imaging or autonomous driving. While bounding the Lips- chitz constant of a neural network improves stability, most methods rely on restricting the Lipschitz constants of each layer which gives a poor bound for the actual Lipschitz constant. In this paper we investigate a variational regularization method named CLIP for controlling the Lipschitz constant of a neural network, which can easily be integrated into the training procedure. We mathematically analyze the proposed model, in particular discussing the impact of the chosen regularization parameter on the output of the network. Finally, we numerically evaluate our method on both a nonlinear regression prob- lem and the MNIST and Fashion-MNIST classi cation databases, and compare our results with a weight regularization approach. (@Bungert2021CLIPCL)

Lenaic Chizat, Edouard Oyallon, and Francis Bach On lazy training in differentiable programming *Advances in neural information processing systems*, 32, 2019. **Abstract:** In a series of recent theoretical works, it was shown that strongly over-parameterized neural networks trained with gradient-based methods could converge exponentially fast to zero training loss, with their parameters hardly varying. In this work, we show that this "lazy training" phenomenon is not specific to over-parameterized neural networks, and is due to a choice of scaling, often implicit, that makes the model behave as its linearization around the initialization, thus yielding a model equivalent to learning with positive-definite kernels. Through a theoretical analysis, we exhibit various situations where this phenomenon arises in non-convex optimization and we provide bounds on the distance between the lazy and linearized optimization paths. Our numerical experiments bring a critical note, as we observe that the performance of commonly used non-linear deep convolutional neural networks in computer vision degrades when trained in the lazy regime. This makes it unlikely that "lazy training" is behind the many successes of neural networks in difficult high dimensional tasks. (@chizat2019lazy)

Ching-Yao Chuang, Youssef Mroueh, Kristjan Greenewald, Antonio Torralba, and Stefanie Jegelka *Advances in Neural Information Processing Systems*, 34: 8294–8306, 2021. **Abstract:** Understanding the generalization of deep neural networks is one of the most important tasks in deep learning. Although much progress has been made, theoretical error bounds still often behave disparately from empirical observations. In this work, we develop margin-based generalization bounds, where the margins are normalized with optimal transport costs between independent random subsets sampled from the training distribution. In particular, the optimal transport cost can be interpreted as a generalization of variance which captures the structural properties of the learned feature space. Our bounds robustly predict the generalization error, given training data and network parameters, on large scale datasets. Theoretically, we demonstrate that the concentration and separation of features play crucial roles in generalization, supporting empirical results in the literature. The code is available at \\}url{https://github.com/chingyaoc/kV-Margin}. (@Chuang2021MeasuringGW)

Moustapha Cissé, Piotr Bojanowski, Edouard Grave, Yann N. Dauphin, and Nicolas Usunier In Doina Precup and Yee Whye Teh (eds.), *Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017*, volume 70 of *Proceedings of Machine Learning Research*, pp. 854–863. PMLR, 2017. URL <http://proceedings.mlr.press/v70/cisse17a.html>. **Abstract:** We introduce Parseval networks, a form of deep neural networks in which the Lipschitz constant of linear, convolutional and aggregation layers is constrained to be smaller than 1. Parseval networks are empirically and theoretically motivated by an analysis of the robustness of the predictions made by deep neural networks when their input is subject to an adversarial perturbation. The most important feature of Parseval networks is to maintain weight matrices of linear and convolutional layers to be (approximately) Parseval tight frames, which are extensions of orthogonal matrices to non-square matrices. We describe how these constraints can be maintained efficiently during SGD. We show that Parseval networks match the state-of-the-art in terms of accuracy on CIFAR-10/100 and Street View House Numbers (SVHN), while being more robust than their vanilla counterpart against adversarial examples. Incidentally, Parseval networks also tend to train faster and make a better usage of the full capacity of the networks. (@cisse2017parseval)

Shaobo Cui and Yong Jiang *2017 2nd IEEE International Conference on Computational Intelligence and Applications (ICCIA)*, pp. 74–78, 2017. **Abstract:** Generative Adversarial Networks (GANs) are efficient frameworks for estimating generative model via adversarial process. However, GAN has known for suffering from training instability. Wasserstein GAN (WGAN) improves the training stability significantly but also brings an additional Lipschitz requirement for the critic network. To enforce the Lipschitz constraint, instead of weight clipping strategy, recent work adds a gradient penalty term to the critic loss. In this paper, we combine a more discriminative gradient penalty term with the importance weighting strategy and further propose more effective algorithms for Lipschitz constraint enforcement of the critic in WGAN. Our algorithms do not adding any computation burden. (@Cui2017EffectiveLC)

Benoit Dherin, Michael Munn, Mihaela Rosca, and David Barrett Why neural networks find simple solutions: The many regularizers of geometric complexity *Advances in Neural Information Processing Systems*, 35: 2333–2349, 2022. **Abstract:** In many contexts, simpler models are preferable to more complex models and the control of this model complexity is the goal for many methods in machine learning such as regularization, hyperparameter tuning and architecture design. In deep learning, it has been difficult to understand the underlying mechanisms of complexity control, since many traditional measures are not naturally suitable for deep neural networks. Here we develop the notion of geometric complexity, which is a measure of the variability of the model function, computed using a discrete Dirichlet energy. Using a combination of theoretical arguments and empirical results, we show that many common training heuristics such as parameter norm regularization, spectral norm regularization, flatness regularization, implicit gradient regularization, noise regularization and the choice of parameter initialization all act to control geometric complexity, providing a unifying framework in which to characterize the behavior of deep learning models. (@dherin2022neural)

Simon S. Du, Xiyu Zhai, Barnabás Póczos, and Aarti Singh In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019. URL <https://openreview.net/forum?id=S1eK3i09YQ>. **Abstract:** One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth. This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks. For an $m$ hidden node shallow neural network with ReLU activation and $n$ training data, we show as long as $m$ is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. Our analysis relies on the following observation: over-parameterization and random initialization jointly restrict every weight vector to be close to its initialization for all iterations, which allows us to exploit a strong convexity-like property to show that gradient descent converges at a global linear rate to the global optimum. We believe these insights are also useful in analyzing deep models and other first order methods. (@du2018gradient)

N. Benjamin Erichson, Omri Azencot, Alejandro F. Queiruga, Liam Hodgkinson, and Michael W. Mahoney In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021. URL <https://openreview.net/forum?id=-N7PBXqOUJZ>. **Abstract:** Viewing recurrent neural networks (RNNs) as continuous-time dynamical systems, we propose a recurrent unit that describes the hidden state’s evolution with two parts: a well-understood linear component plus a Lipschitz nonlinearity. This particular functional form facilitates stability analysis of the long-term behavior of the recurrent unit using tools from nonlinear systems theory. In turn, this enables architectural design decisions before experimentation. Sufficient conditions for global stability of the recurrent unit are obtained, motivating a novel scheme for constructing hidden-to-hidden matrices. Our experiments demonstrate that the Lipschitz RNN can outperform existing recurrent units on a range of benchmark tasks, including computer vision, language modeling and speech prediction tasks. Finally, through Hessian-based analysis we demonstrate that our Lipschitz recurrent unit is more robust with respect to input and parameter perturbations as compared to other continuous-time RNNs. (@Erichson2020LipschitzRN)

Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George J. Pappas In *Neural Information Processing Systems*, 2019. **Abstract:** Tight estimation of the Lipschitz constant for deep neural networks (DNNs) is useful in many applications ranging from robustness certification of classifiers to stability analysis of closed-loop systems with reinforcement learning controllers. Existing methods in the literature for estimating the Lipschitz constant suffer from either lack of accuracy or poor scalability. In this paper, we present a convex optimization framework to compute guaranteed upper bounds on the Lipschitz constant of DNNs both accurately and efficiently. Our main idea is to interpret activation functions as gradients of convex potential functions. Hence, they satisfy certain properties that can be described by quadratic constraints. This particular description allows us to pose the Lipschitz constant estimation problem as a semidefinite program (SDP). The resulting SDP can be adapted to increase either the estimation accuracy (by capturing the interaction between activation functions of different layers) or scalability (by decomposition and parallel implementation). We illustrate the utility of our approach with a variety of experiments on randomly generated networks and on classifiers trained on the MNIST and Iris datasets. In particular, we experimentally demonstrate that our Lipschitz bounds are the most accurate compared to those in the literature. We also study the impact of adversarial training methods on the Lipschitz bounds of the resulting classifiers and show that our bounds can be used to efficiently provide robustness guarantees. (@Fazlyab2019EfficientAA)

Herbert Federer *Geometric Measure Theory*, chapter 3.1.1, pp. 209 Springer Berlin, Heidelberg, 1 edition, 1996. **Abstract:** one of their great selling points, it would seem remiss not to include at least some of them’. The applications discussed in this chapter relate to geometric group theory, in particular to the concept of growth in (finitely generated) groups, which measures the asymptotic rate of growth of the cardinality of the -fold product of certain kinds of subsets of the group. A famous result in this area is Gromov’s theorem, which can be proved and refined by using approximate groups. n S (@GeometricMeasureTheory)

Matteo Gamba, Hossein Azizpour, and Mårten Björkman *CoRR*, abs/2301.12309, 2023. . URL <https://doi.org/10.48550/arXiv.2301.12309>. **Abstract:** Existing bounds on the generalization error of deep networks assume some form of smooth or bounded dependence on the input variable, falling short of investigating the mechanisms controlling such factors in practice. In this work, we present an extensive experimental study of the empirical Lipschitz constant of deep networks undergoing double descent, and highlight non-monotonic trends strongly correlating with the test error. Building a connection between parameter-space and input-space gradients for SGD around a critical point, we isolate two important factors – namely loss landscape curvature and distance of parameters from initialization – respectively controlling optimization dynamics around a critical point and bounding model function complexity, even beyond the training data. Our study presents novels insights on implicit regularization via overparameterization, and effective model complexity for networks trained in practice. (@https://doi.org/10.48550/arxiv.2301.12309)

Stuart Geman, Elie Bienenstock, and René Doursat Neural networks and the bias/variance dilemma *Neural computation*, 4 (1): 1–58, 1992. **Abstract:** Feedforward neural networks trained by error backpropagation are examples of nonparametric regression estimators. We present a tutorial on nonparametric inference and its relation to neural networks, and we use the statistical viewpoint to highlight strengths and weaknesses of neural models. We illustrate the main points with some recognition experiments involving artificial data as well as handwritten numerals. In way of conclusion, we suggest that current-generation feedforward neural networks are largely inadequate for difficult problems in machine perception and machine learning, regardless of parallel-versus-serial hardware or other implementation issues. Furthermore, we suggest that the fundamental challenges in neural modeling are about representation rather than learning per se. This last point is supported by additional experiments with handwritten numerals. (@geman1992neural)

Fabian Latorre Gómez, Paul Rolland, and Volkan Cevher In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020. URL <https://openreview.net/forum?id=rJe4_xSFDB>. **Abstract:** We introduce LiPopt, a polynomial optimization framework for computing increasingly tighter upper bounds on the Lipschitz constant of neural networks. The underlying optimization problems boil down to either linear (LP) or semidefinite (SDP) programming. We show how to use the sparse connectivity of a network, to significantly reduce the complexity of computation. This is specially useful for convolutional as well as pruned neural networks. We conduct experiments on networks with random weights as well as networks trained on MNIST, showing that in the particular case of the $\\}ell\_\\}infty$-Lipschitz constant, our approach yields superior estimates, compared to baselines available in the literature. (@DBLP:journals/corr/abs-2004-08688)

Ian Goodfellow, Yoshua Bengio, and Aaron Courville *Deep Learning*, chapter 9.1, pp. 329 MIT Press, 2016. <http://www.deeplearningbook.org>. **Abstract:** Sweden’s entry into the pan-European vogue for wunderkammern, its medical regime, and its elite cultures. She interrogates Schefferus’s participation in revivified discourses aboutmirabilia and exemplarity, and, as a conclusion, she offers a brisk history of monstrosity itself, in which exceptions become “teachers of virtue” (151, 152–54, 157). Schefferus and his colleagues collect, investigate, and display anomalies at the intersection of enchantment and exempla; their default sensibility is eclecticism inflected with genuine wonder. The final chapter of this beautifully produced and splendidly illustrated collection is devoted to perinatal health and dystocia, “unnatural delivery” (168), in early modern Sweden. For Tove Paulsson Holmbert the stillborn is “a promise unfulfilled,” a “familiar stranger” at the limen between life and death, pallid even as “the appearance of life still lingers on” (169). This deeply eloquent chapter explores obstetrical perception and practice, as well as signs of fetal vitality and decay and intrauterine intervention, via the work of physician and midwife Johan van Hoorn (d. 1724). One of Van Hoorn’s treatises offers some 110 cases of dystocia, which collectively present the methods and techniques of emergency obstetric practice, the most difficult of which were births requiring procedures like podalic version (172–76). In these troublesome births, wise accoucheurs must “execute” the child to save the mother (179–80), all the while aware of mutual risk in “the fundamental instability of human existence in a fallen world” (181). Instability certainly conditions exception, and exception “confirms the force of law” (Bacon, Works, 5.91), proves cases, hypotheses, theories. But is there a science of particulars, of exceptions, able to make normative claims about categories it impugns? These are legal, medical, and natural philosophical questions, and exception is but one star around which such questions, like this collection, orbit. Are we to assume that it was eclipsed by the “advent of the normal”? (@Goodfellow-et-al-2016)

Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy In Yoshua Bengio and Yann LeCun (eds.), *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*, 2015. URL <http://arxiv.org/abs/1412.6572>. **Abstract:** Several machine learning models, including neural networks, consistently misclassify adversarial examples—inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence. Early attempts at explaining this phenomenon focused on nonlinearity and overfitting. We argue instead that the primary cause of neural networks’ vulnerability to adversarial perturbation is their linear nature. This explanation is supported by new quantitative results while giving the first explanation of the most intriguing fact about them: their generalization across architectures and training sets. Moreover, this view yields a simple and fast method of generating adversarial examples. Using this approach to provide examples for adversarial training, we reduce the test set error of a maxout network on the MNIST dataset. (@goodfellow2014explaining)

Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J. Cree *Machine Learning*, 110 (2): 393–416, Feb 2021. ISSN 1573-0565. . URL <https://doi.org/10.1007/s10994-020-05929-w>. **Abstract:** Abstract We investigate the effect of explicitly enforcing the Lipschitz continuity of neural networks with respect to their inputs. To this end, we provide a simple technique for computing an upper bound to the Lipschitz constant—for multiple p -norms—of a feed forward neural network composed of commonly used layer types. Our technique is then used to formulate training a neural network with a bounded Lipschitz constant as a constrained optimisation problem that can be solved using projected stochastic gradient methods. Our evaluation study shows that the performance of the resulting models exceeds that of models trained with other common regularisers. We also provide evidence that the hyperparameters are intuitive to tune, demonstrate how the choice of norm for computing the Lipschitz constant impacts the resulting model, and show that the performance gains provided by our method are particularly noticeable when only a small amount of training data is available. (@Gouk2021)

Sam Greydanus *CoRR*, abs/2011.14439, 2020. URL <https://arxiv.org/abs/2011.14439>. **Abstract:** Although deep learning models have taken on commercial and political relevance, key aspects of their training and operation remain poorly understood. This has sparked interest in science of deep learning projects, many of which require large amounts of time, money, and electricity. But how much of this research really needs to occur at scale? In this paper, we introduce MNIST-1D: a minimalist, procedurally generated, low-memory, and low-compute alternative to classic deep learning benchmarks. Although the dimensionality of MNIST-1D is only 40 and its default training set size only 4000, MNIST-1D can be used to study inductive biases of different deep architectures, find lottery tickets, observe deep double descent, metalearn an activation function, and demonstrate guillotine regularization in self-supervised learning. All these experiments can be conducted on a GPU or often even on a CPU within minutes, allowing for fast prototyping, educational use cases, and cutting-edge research on a low budget. (@DBLP:journals/corr/abs-2011-14439)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun Deep residual learning for image recognition In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016. **Abstract:** Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers - 8× deeper than VGG nets \[40\] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions1, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. (@He2015DeepRL)

Todd Huster, Cho-Yu Jason Chiang, and Ritu Chadha *CoRR*, abs/1807.09705, 2018. URL <http://arxiv.org/abs/1807.09705>. **Abstract:** Several recent papers have discussed utilizing Lipschitz constants to limit the susceptibility of neural networks to adversarial examples. We analyze recently proposed methods for computing the Lipschitz constant. We show that the Lipschitz constant may indeed enable adversarially robust neural networks. However, the methods currently employed for computing it suffer from theoretical and practical limitations. We argue that addressing this shortcoming is a promising direction for future research into certified adversarial defenses. (@Huster2018LimitationsOT)

Yerlan Idelbayev Proper ResNet implementation for CIFAR10/CIFAR100 in PyTorch <https://github.com/akamaster/pytorch_resnet_cifar10>. Accessed: 2023-06-17. **Abstract:** Torchvision model zoo provides number of implementations of various state-of-the-art architectures, however, most of them are defined and implemented for ImageNet. Usually it is straightforward to use the provided models on other datasets, but some cases require manual setup. For instance, very few pytorch repositories with ResNets on CIFAR10 provides the implementation as described in the original paper . If you just use the torchvision’s models on CIFAR10 you’ll get the model that differs in number of layers and parameters . This is unacceptable if you want to directly compare ResNet-s on CIFAR10 with the original paper. The purpose of this repo is to provide a valid pytorch implementation of ResNet-s for CIFAR10 as described in the original paper. The following models are provided: This implementation matches description of the original paper, with comparable or better test error. Our implementation follows the paper in straightforward manner with some caveats: First , the training in the paper uses 45k/5k train/validation split on the train data, and selects the best performing model based on the performance on the validation set. We do not perform validation testing; if you need to compare your results on ResNet head-to-head to the orginal paper keep this in mind. Second , if you want to train ResNet1202 keep in mind that you need 16GB memory on GPU. If you find this implementation useful and want to cite/mention this page, here is a bibtex citation: (@Idelbayev18a)

Arthur Jacot, Franck Gabriel, and Clément Hongler Neural tangent kernel: Convergence and generalization in neural networks *Advances in neural information processing systems*, 31, 2018. **Abstract:** At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods. We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function $f\_\\}theta$ (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK). This kernel is central to describe the generalization features of ANNs. While the NTK is random at initialization and varies during training, in the infinite-width limit it converges to an explicit limiting kernel and it stays constant during training. This makes it possible to study the training of ANNs in function space instead of parameter space. Convergence of the training can then be related to the positive-definiteness of the limiting NTK. We prove the positive-definiteness of the limiting NTK when the data is supported on the sphere and the non-linearity is non-polynomial. We then focus on the setting of least-squares regression and show that in the infinite-width limit, the network function $f\_\\}theta$ follows a linear differential equation during training. The convergence is fastest along the largest kernel principal components of the input data with respect to the NTK, hence suggesting a theoretical motivation for early stopping. Finally we study the NTK numerically, observe its behavior for wide networks, and compare it to the infinite-width limit. (@jacot2018neural)

Matt Jordan and Alexandros G Dimakis *Advances in Neural Information Processing Systems*, 33: 7344–7353, 2020. **Abstract:** The local Lipschitz constant of a neural network is a useful metric with applications in robustness, generalization, and fairness evaluation. We provide novel analytic results relating the local Lipschitz constant of nonsmooth vector-valued functions to a maximization over the norm of the generalized Jacobian. We present a sufficient condition for which backpropagation always returns an element of the generalized Jacobian, and reframe the problem over this broad class of functions. We show strong inapproximability results for estimating Lipschitz constants of ReLU networks, and then formulate an algorithm to compute these quantities exactly. We leverage this algorithm to evaluate the tightness of competing Lipschitz estimators and the effects of regularized training on the Lipschitz constant. (@jordan2020exactly)

Hyunjik Kim, George Papamakarios, and Andriy Mnih In *International Conference on Machine Learning*, 2020. **Abstract:** Lipschitz constants of neural networks have been explored in various contexts in deep learning, such as provable adversarial robustness, estimating Wasserstein distance, stabilising training of GANs, and formulating invertible neural networks. Such works have focused on bounding the Lipschitz constant of fully connected or convolutional networks, composed of linear maps and pointwise non-linearities. In this paper, we investigate the Lipschitz constant of self-attention, a non-linear neural network module widely used in sequence modelling. We prove that the standard dot-product self-attention is not Lipschitz for unbounded input domain, and propose an alternative L2 self-attention that is Lipschitz. We derive an upper bound on the Lipschitz constant of L2 self-attention and provide empirical evidence for its asymptotic tightness. To demonstrate the practical relevance of our theoretical work, we formulate invertible self-attention and use it in a Transformer-based architecture for a character-level language modelling task. (@Kim2020TheLC)

Alex Krizhevsky . **Abstract:** In this work we describe how to train a multi-layer generative model of natural images. We use a dataset of millions of tiny colour images, described in the next section. This has been attempted by several groups but without success. The models on which we focus are RBMs (Restricted Boltzmann Machines) and DBNs (Deep Belief Networks). These models learn interesting-looking filters, which we show are more useful to a classifier than the raw pixels. We train the classifier on a labeled subset that we have collected and call the CIFAR-10 dataset. (@Krizhevsky2009LearningML)

Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio In *5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings*. OpenReview.net, 2017. URL <https://openreview.net/forum?id=BJm4T4Kgx>. **Abstract:** Adversarial examples are malicious inputs designed to fool machine learning models. They often transfer from one model to another, allowing attackers to mount black box attacks without knowledge of the target model’s parameters. Adversarial training is the process of explicitly training a model on adversarial examples, in order to make it more robust to attack or to reduce its test error on clean inputs. So far, adversarial training has primarily been applied to small problems. In this research, we apply adversarial training to ImageNet. Our contributions include: (1) recommendations for how to succesfully scale adversarial training to large models and datasets, (2) the observation that adversarial training confers robustness to single-step attack methods, (3) the finding that multi-step attack methods are somewhat less transferable than single-step attack methods, so single-step attacks are the best for mounting black-box attacks, and (4) resolution of a "label leaking" effect that causes adversarially trained models to perform better on adversarial examples than on clean examples, because the adversarial example construction process uses the true label and the model can learn to exploit regularities in the construction process. (@Kurakin2016AdversarialML)

Jaehoon Lee, Samuel S. Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, and Jascha Sohl-Dickstein Finite versus infinite neural networks: an empirical study 2020. **Abstract:** We perform a careful, thorough, and large scale empirical study of the correspondence between wide neural networks and kernel methods. By doing so, we resolve a variety of open questions related to the study of infinitely wide neural networks. Our experimental results include: kernel methods outperform fully-connected finite-width networks, but underperform convolutional finite width networks; neural network Gaussian process (NNGP) kernels frequently outperform neural tangent (NT) kernels; centered and ensembled finite networks have reduced posterior variance and behave more similarly to infinite networks; weight decay and the use of a large learning rate break the correspondence between finite and infinite networks; the NTK parameterization outperforms the standard parameterization for finite width networks; diagonal regularization of kernels acts similarly to early stopping; floating point precision limits kernel performance beyond a critical dataset size; regularized ZCA whitening improves accuracy; finite network performance depends non-monotonically on width in ways not captured by double descent phenomena; equivariance of CNNs is only beneficial for narrow networks far from the kernel regime. Our experiments additionally motivate an improved layer-wise scaling for weight decay which improves generalization in finite-width networks. Finally, we develop improved best practices for using NNGP and NT kernels for prediction, including a novel ensembling technique. Using these best practices we achieve state-of-the-art results on CIFAR-10 classification for kernels corresponding to each architecture class we consider. (@lee2020finite)

Klas Leino, Zifan Wang, and Matt Fredrikson In *International Conference on Machine Learning*, pp. 6212–6222. PMLR, 2021. **Abstract:** In this paper, two related problems, global asymptotic stability (GAS) and global robust stability (GRS) of neural networks with time delays, are studied. First, GAS of delayed neural networks is discussed based on Lyapunov method and linear matrix inequality. New criteria are given to ascertain the GAS of delayed neural networks. In the designs and applications of neural networks, it is necessary to consider the deviation effects of bounded perturbations of network parameters. In this case, a delayed neural network must be formulated as a interval neural network model. Several sufficient conditions are derived for the existence, uniqueness, and GRS of equilibria for interval neural networks with time delays by use of a new Lyapunov function and matrix inequality. These results are less restrictive than those given in the earlier references. (@Leino2021GloballyRobustNN)

Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger B Grosse, and Jörn-Henrik Jacobsen *Advances in neural information processing systems*, 32, 2019. **Abstract:** Lipschitz constraints under L2 norm on deep neural networks are useful for provable adversarial robustness bounds, stable training, and Wasserstein distance estimation. While heuristic approaches such as the gradient penalty have seen much practical success, it is challenging to achieve similar practical performance while provably enforcing a Lipschitz constraint. In principle, one can design Lipschitz constrained architectures using the composition property of Lipschitz functions, but Anil et al. recently identified a key obstacle to this approach: gradient norm attenuation. They showed how to circumvent this problem in the case of fully connected networks by designing each layer to be gradient norm preserving. We extend their approach to train scalable, expressive, provably Lipschitz convolutional networks. In particular, we present the Block Convolution Orthogonal Parameterization (BCOP), an expressive parameterization of orthogonal convolution operations. We show that even though the space of orthogonal convolutions is disconnected, the largest connected component of BCOP with 2n channels can represent arbitrary BCOP convolutions over n channels. Our BCOP parameterization allows us to train large convolutional networks with provable Lipschitz bounds. Empirically, we find that it is competitive with existing approaches to provable adversarial robustness and Wasserstein distance estimation. (@li2019preventing)

Song Mei and Andrea Montanari The generalization error of random features regression: Precise asymptotics and the double descent curve *Communications on Pure and Applied Mathematics*, 75 (4): 667–766, 2022. **Abstract:** Abstract Deep learning methods operate in regimes that defy the traditional statistical mindset. Neural network architectures often contain more parameters than training samples, and are so rich that they can interpolate the observed labels, even if the latter are replaced by pure noise. Despite their huge complexity, the same architectures achieve small generalization error on real data. This phenomenon has been rationalized in terms of a so‐called ‘double descent’ curve. As the model complexity increases, the test error follows the usual U‐shaped curve at the beginning, first decreasing and then peaking around the interpolation threshold (when the model achieves vanishing training error). However, it descends again as model complexity exceeds this threshold. The global minimum of the test error is found above the interpolation threshold, often in the extreme overparametrization regime in which the number of parameters is much larger than the number of samples. Far from being a peculiar property of deep neural networks, elements of this behavior have been demonstrated in much simpler settings, including linear regression with random covariates. In this paper we consider the problem of learning an unknown function over the ‐dimensional sphere , from i.i.d. samples , . We perform ridge regression on random features of the form , . This can be equivalently described as a two‐layer neural network with random first‐layer weights. We compute the precise asymptotics of the test error, in the limit with and fixed. This provides the first analytically tractable model that captures all the features of the double descent phenomenon without assuming ad hoc misspecification structures. In particular, above a critical value of the signal‐to‐noise ratio, minimum test error is achieved by extremely overparametrized interpolators, i.e., networks that have a number of parameters much larger than the sample size, and vanishing training error. © 2021 Wiley Periodicals LLC. (@mei2022generalization)

Laurent Meunier, Blaise J Delattre, Alexandre Araujo, and Alexandre Allauzen In *International Conference on Machine Learning*, pp. 15484–15500. PMLR, 2022. **Abstract:** The Lipschitz constant of neural networks has been established as a key quantity to enforce the robustness to adversarial examples. In this paper, we tackle the problem of building $1$-Lipschitz Neural Networks. By studying Residual Networks from a continuous time dynamical system perspective, we provide a generic method to build $1$-Lipschitz Neural Networks and show that some previous approaches are special cases of this framework. Then, we extend this reasoning and show that ResNet flows derived from convex potentials define $1$-Lipschitz transformations, that lead us to define the {\\}em Convex Potential Layer} (CPL). A comprehensive set of experiments on several datasets demonstrates the scalability of our architecture and the benefits as an $\\}ell_2$-provable defense against adversarial examples. (@Meunier2021ADS)

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida In *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*. OpenReview.net, 2018. URL <https://openreview.net/forum?id=B1QRgziT->. **Abstract:** One of the challenges in the study of generative adversarial networks is the instability of its training. In this paper, we propose a novel weight normalization technique called spectral normalization to stabilize the training of the discriminator. Our new normalization technique is computationally light and easy to incorporate into existing implementations. We tested the efficacy of spectral normalization on CIFAR10, STL-10, and ILSVRC2012 dataset, and we experimentally confirmed that spectrally normalized GANs (SN-GANs) is capable of generating images of better or equal quality relative to the previous training stabilization techniques. (@Miyato2018SpectralNF)

Andrea Montanari and Yiqiao Zhong The interpolation phase transition in neural networks: Memorization and generalization under lazy training *The Annals of Statistics*, 50 (5): 2816–2847, 2022. **Abstract:** Modern neural networks are often operated in a strongly overparametrized regime: they comprise so many parameters that they can interpolate the training set, even if actual labels are replaced by purely random ones. Despite this, they achieve good prediction error on unseen data: interpolating the training set does not lead to a large generalization error. Further, overparametrization appears to be beneficial in that it simplifies the optimization landscape. Here, we study these phenomena in the context of two-layers neural networks in the neural tangent (NT) regime. We consider a simple data model, with isotropic covariates vectors in d dimensions, and N hidden neurons. We assume that both the sample size n and the dimension d are large, and they are polynomially related. Our first main result is a characterization of the eigenstructure of the empirical NT kernel in the overparametrized regime Nd≫n. This characterization implies as a corollary that the minimum eigenvalue of the empirical NT kernel is bounded away from zero as soon as Nd≫n and, therefore, the network can exactly interpolate arbitrary labels in the same regime. Our second main result is a characterization of the generalization error of NT ridge regression including, as a special case, min-ℓ2 norm interpolation. We prove that, as soon as Nd≫n, the test error is well approximated by the one of kernel ridge regression with respect to the infinite-width kernel. The latter is in turn well approximated by the error of polynomial ridge regression, whereby the regularization parameter is increased by a "self-induced" term related to the high-degree components of the activation function. The polynomial degree depends on the sample size and the dimension (in particular on logn/logd). (@montanari2022interpolation)

Michael Munn, Benoit Dherin, and Javier Gonzalvo A margin-based multiclass generalization bound via geometric complexity 2023. URL <https://openreview.net/forum?id=fEx3f7YXv1&referrer=%5Bthe%20profile%20of%20Benoit%20Dherin%5D(%2Fprofile%3Fid%3D~Benoit_Dherin1)>. **Abstract:** There has been considerable effort to better understand the generalization capabilities of deep neural networks both as a means to unlock a theoretical understanding of their success as well as providing directions for further improvements. In this paper, we investigate margin-based multiclass generalization bounds for neural networks which rely on a recent complexity measure, the geometric complexity, developed for neural networks. We derive a new upper bound on the generalization error which scales with the margin-normalized geometric complexity of the network and which holds for a broad family of data distributions and model classes. Our generalization bound is empirically investigated for a ResNet-18 model trained with SGD on the CIFAR-10 and CIFAR-100 datasets with both original and random labels. (@GeneralisationBoundGC)

Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020. URL <https://openreview.net/forum?id=B1g5sA4twr>. **Abstract:** Abstract We show that a variety of modern deep learning tasks exhibit a ‘double-descent’ phenomenon where, as we increase model size, performance first gets worse and then gets better. Moreover, we show that double descent occurs not just as a function of model size, but also as a function of the number of training epochs. We unify the above phenomena by defining a new complexity measure we call the effective model complexity and conjecture a generalized double descent with respect to this measure. Furthermore, our notion of model complexity allows us to identify certain regimes where increasing (even quadrupling) the number of train samples actually hurts test performance. (@nakkiran2019deep)

Preetum Nakkiran, Prayaag Venkat, Sham M. Kakade, and Tengyu Ma In *9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net, 2021. URL <https://openreview.net/forum?id=7R7fAoUygoa>. **Abstract:** Recent empirical and theoretical studies have shown that many learning algorithms – from linear regression to neural networks – can have test performance that is non-monotonic in quantities such the sample size and model size. This striking phenomenon, often referred to as "double descent", has raised questions of if we need to re-think our current understanding of generalization. In this work, we study whether the double-descent phenomenon can be avoided by using optimal regularization. Theoretically, we prove that for certain linear regression models with isotropic data distribution, optimally-tuned $\\}ell_2$ regularization achieves monotonic test performance as we grow either the sample size or the model size. We also demonstrate empirically that optimally-tuned $\\}ell_2$ regularization can mitigate double descent for more general models, including neural networks. Our results suggest that it may also be informative to study the test risk scalings of various algorithms in the context of appropriately tuned regularization. (@nakkiran2020optimal)

Brady Neal, Sarthak Mittal, Aristide Baratin, Vinayak Tantia, Matthew Scicluna, Simon Lacoste-Julien, and Ioannis Mitliagkas *CoRR*, abs/1810.08591, 2018. URL <http://arxiv.org/abs/1810.08591>. **Abstract:** The bias-variance tradeoff tells us that as model complexity increases, bias falls and variances increases, leading to a U-shaped test error curve. However, recent empirical results with over-parameterized neural networks are marked by a striking absence of the classic U-shaped test error curve: test error keeps decreasing in wider networks. This suggests that there might not be a bias-variance tradeoff in neural networks with respect to network width, unlike was originally claimed by, e.g., Geman et al. (1992). Motivated by the shaky evidence used to support this claim in neural networks, we measure bias and variance in the modern setting. We find that both bias and variance can decrease as the number of parameters grows. To better understand this, we introduce a new decomposition of the variance to disentangle the effects of optimization and data sampling. We also provide theoretical analysis in a simplified setting that is consistent with our empirical findings. (@neal2018modern)

Hajime Ono, Tsubasa Takahashi, and Kazuya Kakizaki *CoRR*, abs/1811.08080, 2018. URL <http://arxiv.org/abs/1811.08080>. **Abstract:** How can we make machine learning provably robust against adversarial examples in a scalable way? Since certified defense methods, which ensure $\\}epsilon$-robust, consume huge resources, they can only achieve small degree of robustness in practice. Lipschitz margin training (LMT) is a scalable certified defense, but it can also only achieve small robustness due to over-regularization. How can we make certified defense more efficiently? We present LC-LMT, a light weight Lipschitz margin training which solves the above problem. Our method has the following properties; (a) efficient: it can achieve $\\}epsilon$-robustness at early epoch, and (b) robust: it has a potential to get higher robustness than LMT. In the evaluation, we demonstrate the benefits of the proposed method. LC-LMT can achieve required robustness more than 30 epoch earlier than LMT in MNIST, and shows more than 90 $\\}%$ accuracy against both legitimate and adversarial inputs. (@Ono2018LightweightLM)

Patricia Pauli, Anne Koch, Julian Berberich, Paul Kohler, and Frank Allgöwer *2021 American Control Conference (ACC)*, pp. 2595–2600, 2021. **Abstract:** Due to their susceptibility to adversarial perturbations, neural networks (NNs) are hardly used in safety-critical applications. One measure of robustness to such perturbations in the input is the Lipschitz constant of the input-output map defined by an NN. In this work, we propose a framework to train multi-layer NNs while at the same time encouraging robustness by keeping their Lipschitz constant small, thus addressing the robustness issue. More specifically, we design an optimization scheme based on the Alternating Direction Method of Multipliers that minimizes not only the training loss of an NN but also its Lipschitz constant resulting in a semidefinite programming based training procedure that promotes robustness. We design two versions of this training procedure. The first one includes a regularizer that penalizes an accurate upper bound on the Lipschitz constant. The second one allows to enforce a desired Lipschitz bound on the NN at all times during training. Finally, we provide two examples to show that the proposed framework successfully increases the robustness of NNs. (@Pauli2021TrainingRN)

Henning Petzka, Asja Fischer, and Denis Lukovnikov *CoRR*, abs/1709.08894, 2017. URL <http://arxiv.org/abs/1709.08894>. **Abstract:** Since their invention, generative adversarial networks (GANs) have become a popular approach for learning to model a distribution of real (unlabeled) data. Convergence problems during training are overcome by Wasserstein GANs which minimize the distance between the model and the empirical distribution in terms of a different metric, but thereby introduce a Lipschitz constraint into the optimization problem. A simple way to enforce the Lipschitz constraint on the class of functions, which can be modeled by the neural network, is weight clipping. It was proposed that training can be improved by instead augmenting the loss by a regularization term that penalizes the deviation of the gradient of the critic (as a function of the network’s input) from one. We present theoretical arguments why using a weaker regularization term enforcing the Lipschitz constraint is preferable. These arguments are supported by experimental results on toy data sets. (@Petzka2017OnTR)

Bernd Prach and Christoph H Lampert In *European Conference on Computer Vision*, pp. 350–365. Springer, 2022. **Abstract:** It is a highly desirable property for deep networks to be robust against small input changes. One popular way to achieve this property is by designing networks with a small Lipschitz constant. In this work, we propose a new technique for constructing such Lipschitz networks that has a number of desirable properties: it can be applied to any linear network layer (fully-connected or convolutional), it provides formal guarantees on the Lipschitz constant, it is easy to implement and efficient to run, and it can be combined with any training objective and optimization method. In fact, our technique is the first one in the literature that achieves all of these properties simultaneously. Our main contribution is a rescaling-based weight matrix parametrization that guarantees each network layer to have a Lipschitz constant of at most 1and results in the learned weight matrices to be close to orthogonal. Hence we call such layers almost-orthogonal Lipschitz (AOL) . Experiments and ablation studies in the context of image classification with certified robust accuracy confirm that AOL layers achieve results that are on par with most existing methods. Yet, they are simpler to implement and more broadly applicable, because they do not require computationally expensive matrix orthogonalization or inversion steps as part of the network architecture. We provide code at https://github.com/berndprach/AOL . (@Prach2022AlmostOrthogonalLF)

Yipeng Qin, Niloy Jyoti Mitra, and Peter Wonka In *European Conference on Computer Vision*, 2018. **Abstract:** Despite the success of Lipschitz regularization in stabilizing GAN training, the exact reason of its e ectiveness remains poorly un- derstood. The direct e ect of K-Lipschitz regularization is to restrict the L2-norm of the neural network gradient to be smaller than a threshold K(e.g.,K= 1) such thatkrfkK. In this work, we uncover an even more important e ect of Lipschitz regularization by examining its im- pact on the loss function: It degenerates GAN loss functions to almost linear ones by restricting their domain and interval of attainable gradi- ent values . Our analysis shows that loss functions are only successful if they are degenerated to almost linear ones. We also show that loss func- tions perform poorly if they are not degenerated and that a wide range of functions can be used as loss function as long as they are suciently degenerated by regularization. Basically, Lipschitz regularization ensures that all loss functions e ectively work in the same way. Empirically, we verify our proposition on the MNIST, CIFAR10 and CelebA datasets. (@Qin2018HowDL)

Max Revay, Ruigang Wang, and Ian R. Manchester *CoRR*, abs/2010.01732, 2020. URL <https://arxiv.org/abs/2010.01732>. **Abstract:** This paper introduces new parameterizations of equilibrium neural networks, i.e. networks defined by implicit equations. This model class includes standard multilayer and residual networks as special cases. The new parameterization admits a Lipschitz bound during training via unconstrained optimization: no projections or barrier functions are required. Lipschitz bounds are a common proxy for robustness and appear in many generalization bounds. Furthermore, compared to previous works we show well-posedness (existence of solutions) under less restrictive conditions on the network weights and more natural assumptions on the activation functions: that they are monotone and slope restricted. These results are proved by establishing novel connections with convex optimization, operator splitting on non-Euclidean spaces, and contracting neural ODEs. In image classification experiments we show that the Lipschitz bounds are very accurate and improve robustness to adversarial attacks. (@Revay2020LipschitzBE)

Maxim Samarin, Volker Roth, and David Belius Feature learning and random features in standard finite-width convolutional neural networks: An empirical study In *Uncertainty in Artificial Intelligence*, pp. 1718–1727. PMLR, 2022. (@samarin2022feature)

Sidak Pal Singh, Aurélien Lucchi, Thomas Hofmann, and Bernhard Schölkopf In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022. URL <https://openreview.net/forum?id=lTqGXfn9Tv>. **Abstract:** ‘Double descent’ delineates the generalization behaviour of models depending on the regime they belong to: under- or over-parameterized. The current theoretical understanding behind the occurrence of this phenomenon is primarily based on linear and kernel regression models – with informal parallels to neural networks via the Neural Tangent Kernel. Therefore such analyses do not adequately capture the mechanisms behind double descent in finite-width neural networks, as well as, disregard crucial components – such as the choice of the loss function. We address these shortcomings by leveraging influence functions in order to derive suitable expressions of the population loss and its lower bound, while imposing minimal assumptions on the form of the parametric model. Our derived bounds bear an intimate connection with the spectrum of the Hessian at the optimum, and importantly, exhibit a double descent behaviour at the interpolation threshold. Building on our analysis, we further investigate how the loss function affects double descent – and thus uncover interesting properties of neural networks and their Hessian spectra near the interpolation threshold. (@https://doi.org/10.48550/arxiv.2203.07337)

Sahil Singla and Soheil Feizi In *International Conference on Machine Learning*, pp. 9756–9766. PMLR, 2021. **Abstract:** Training convolutional neural networks with a Lipschitz constraint under the $l\_{2}$ norm is useful for provable adversarial robustness, interpretable gradients, stable training, etc. While 1-Lipschitz networks can be designed by imposing a 1-Lipschitz constraint on each layer, training such networks requires each layer to be gradient norm preserving (GNP) to prevent gradients from vanishing. However, existing GNP convolutions suffer from slow training, lead to significant reduction in accuracy and provide no guarantees on their approximations. In this work, we propose a GNP convolution layer called Skew Orthogonal Convolution (SOC) that uses the following mathematical property: when a matrix is {\\}it Skew-Symmetric}, its exponential function is an {\\}it orthogonal} matrix. To use this property, we first construct a convolution filter whose Jacobian is Skew-Symmetric. Then, we use the Taylor series expansion of the Jacobian exponential to construct the SOC layer that is orthogonal. To efficiently implement SOC, we keep a finite number of terms from the Taylor series and provide a provable guarantee on the approximation error. Our experiments on CIFAR-10 and CIFAR-100 show that SOC allows us to train provably Lipschitz, large convolutional neural networks significantly faster than prior works while achieving significant improvements for both standard and certified robust accuracies. (@Singla2021SkewOC)

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus Intriguing properties of neural networks In Yoshua Bengio and Yann LeCun (eds.), *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings*, 2014. URL <http://arxiv.org/abs/1312.6199>. **Abstract:** Deep neural networks are highly expressive models that have recently achieved state of the art performance on speech and visual recognition tasks. While their expressiveness is the reason they succeed, it also causes them to learn uninterpretable solutions that could have counter-intuitive properties. In this paper we report two such properties. First, we find that there is no distinction between individual high level units and random linear combinations of high level units, according to various methods of unit analysis. It suggests that it is the space, rather than the individual units, that contains of the semantic information in the high layers of neural networks. Second, we find that deep neural networks learn input-output mappings that are fairly discontinuous to a significant extend. We can cause the network to misclassify an image by applying a certain imperceptible perturbation, which is found by maximizing the network’s prediction error. In addition, the specific nature of these perturbations is not a random artifact of learning: the same perturbation can cause a different network, that was trained on a different subset of the dataset, to misclassify the same input. (@szegedy2013intriguing)

Dávid Terjék In *International Conference on Learning Representations*, 2019. **Abstract:** Generative adversarial networks (GANs) are one of the most popular approaches when it comes to training generative models, among which variants of Wasserstein GANs are considered superior to the standard GAN formulation in terms of learning stability and sample quality. However, Wasserstein GANs require the critic to be 1-Lipschitz, which is often enforced implicitly by penalizing the norm of its gradient, or by globally restricting its Lipschitz constant via weight normalization techniques. Training with a regularization term penalizing the violation of the Lipschitz constraint explicitly, instead of through the norm of the gradient, was found to be practically infeasible in most situations. With a novel generalization of Virtual Adversarial Training, called Adversarial Lipschitz Regularization, we show that using an explicit Lipschitz penalty is indeed viable and leads to competitive performance when applied to Wasserstein GANs, highlighting an important connection between Lipschitz regularization and adversarial training. (@Terjk2019AdversarialLR)

Yusuke Tsuzuku, Issei Sato, and Masashi Sugiyama In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL <https://proceedings.neurips.cc/paper/2018/file/485843481a7edacbfce101ecb1e4d2a8-Paper.pdf>. **Abstract:** High sensitivity of neural networks against malicious perturbations on inputs causes security concerns. To take a steady step towards robust classifiers, we aim to create neural network models provably defended from perturbations. Prior certification work requires strong assumptions on network structures and massive computational costs, and thus the range of their applications was limited. From the relationship between the Lipschitz constants and prediction margins, we present a computationally efficient calculation technique to lower-bound the size of adversarial perturbations that can deceive networks, and that is widely applicable to various complicated networks. Moreover, we propose an efficient training procedure that robustifies networks and significantly improves the provably guarded areas around data points. In experimental evaluations, our method showed its ability to provide a non-trivial guarantee and enhance robustness for even large networks. (@NEURIPS2018_48584348)

Aladin Virmaux and Kevin Scaman Lipschitz regularity of deep neural networks: analysis and efficient estimation In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL <https://proceedings.neurips.cc/paper/2018/file/d54e99a6c03704e95e6965532dec148b-Paper.pdf>. **Abstract:** Deep neural networks are notorious for being sensitive to small well-chosen perturbations, and estimating the regularity of such architectures is of utmost importance for safe and robust practical applications. In this paper, we investigate one of the key characteristics to assess the regularity of such methods: the Lipschitz constant of deep learning architectures. First, we show that, even for two layer neural networks, the exact computation of this quantity is NP-hard and state-of-art methods may significantly overestimate it. Then, we both extend and improve previous estimation methods by providing AutoLip, the first generic algorithm for upper bounding the Lipschitz constant of any automatically differentiable function. We provide a power method algorithm working with automatic differentiation, allowing efficient computations even on large convolutions. Second, for sequential neural networks, we propose an improved algorithm named SeqLip that takes advantage of the linear computation graph to split the computation per pair of consecutive layers. Third we propose heuristics on SeqLip in order to tackle very large networks. Our experiments show that SeqLip can significantly improve on the existing upper bounds. Finally, we provide an implementation of AutoLip in the PyTorch environment that may be used to better estimate the robustness of a given neural network to small perturbations or regularize it using more precise Lipschitz estimations. (@NEURIPS2018_d54e99a6)

Ruigang Wang and Ian Manchester In *International Conference on Machine Learning*, pp. 36093–36110. PMLR, 2023. **Abstract:** This paper introduces a new parameterization of deep neural networks (both fully-connected and convolutional) with guaranteed $\\}ell^2$ Lipschitz bounds, i.e. limited sensitivity to input perturbations. The Lipschitz guarantees are equivalent to the tightest-known bounds based on certification via a semidefinite program (SDP). We provide a “direct” parameterization, i.e., a smooth mapping from $\\}mathbb R^N$ onto the set of weights satisfying the SDP-based bound. Moreover, our parameterization is complete, i.e. a neural network satisfies the SDP bound if and only if it can be represented via our parameterization. This enables training using standard gradient methods, without any inner approximation or computationally intensive tasks (e.g. projections or barrier terms) for the SDP constraint. The new parameterization can equivalently be thought of as either a new layer type (the \\}textit{sandwich layer}), or a novel parameterization of standard feedforward networks with parameter sharing between neighbouring layers. A comprehensive set of experiments on image classification shows that sandwich layers outperform previous approaches on both empirical and certified robust accuracy. Code is available at \\}url{https://github.com/acfr/LBDN}. (@Wang2023DirectPO)

Yizhou Wang, Yue Kang, Can Qin, Yi Xu, Huan Wang, Yulun Zhang, and Yun Fu *CoRR*, abs/2106.11514, 2021. URL <https://arxiv.org/abs/2106.11514>. **Abstract:** Adaptive gradient methods, e.g. \\}textsc{Adam}, have achieved tremendous success in machine learning. Scaling the learning rate element-wisely by a certain form of second moment estimate of gradients, such methods are able to attain rapid training of modern deep neural networks. Nevertheless, they are observed to suffer from compromised generalization ability compared with stochastic gradient descent (\\}textsc{SGD}) and tend to be trapped in local minima at an early stage during training. Intriguingly, we discover that substituting the gradient in the second raw moment estimate term with its momentumized version in \\}textsc{Adam} can resolve the issue. The intuition is that gradient with momentum contains more accurate directional information and therefore its second moment estimation is a more favorable option for learning rate scaling than that of the raw gradient. Thereby we propose \\}textsc{AdaMomentum} as a new optimizer reaching the goal of training fast while generalizing much better. We further develop a theory to back up the improvement in generalization and provide convergence guarantees under both convex and nonconvex settings. Extensive experiments on a wide range of tasks and models demonstrate that \\}textsc{AdaMomentum} exhibits state-of-the-art performance and superior training stability consistently. (@Wang2021AdaptingSB)

Zi Wang, Gautam Prakriya, and Somesh Jha *Advances in Neural Information Processing Systems*, 35: 34201–34215, 2022. **Abstract:** Fast and precise Lipschitz constant estimation of neural networks is an important task for deep learning. Researchers have recently found an intrinsic trade-off between the accuracy and smoothness of neural networks, so training a network with a loose Lipschitz constant estimation imposes a strong regularization and can hurt the model accuracy significantly. In this work, we provide a unified theoretical framework, a quantitative geometric approach, to address the Lipschitz constant estimation. By adopting this framework, we can immediately obtain several theoretical results, including the computational hardness of Lipschitz constant estimation and its approximability. Furthermore, the quantitative geometric perspective can also provide some insights into recent empirical observations that techniques for one norm do not usually transfer to another one. We also implement the algorithms induced from this quantitative geometric approach in a tool GeoLIP. These algorithms are based on semidefinite programming (SDP). Our empirical evaluation demonstrates that GeoLIP is more scalable and precise than existing tools on Lipschitz constant estimation for $\\}ell\_\\}infty$-perturbations. Furthermore, we also show its intricate relations with other recent SDP-based techniques, both theoretically and empirically. We believe that this unified quantitative geometric perspective can bring new insights and theoretical tools to the investigation of neural-network smoothness and robustness. (@Wang2022AQG)

Tsui-Wei Weng, Huan Zhang, Pin-Yu Chen, Jinfeng Yi, Dong Su, Yupeng Gao, Cho-Jui Hsieh, and Luca Daniel In *International Conference on Learning Representations*, 2018. URL <https://openreview.net/forum?id=BkUHlMZ0b>. **Abstract:** The robustness of neural networks to adversarial examples has received great attention due to security implications. Despite various attack approaches to crafting visually imperceptible adversarial examples, little has been developed towards a comprehensive measure of robustness. In this paper, we provide a theoretical justification for converting robustness analysis into a local Lipschitz constant estimation problem, and propose to use the Extreme Value Theory for efficient evaluation. Our analysis yields a novel robustness metric called CLEVER, which is short for Cross Lipschitz Extreme Value for nEtwork Robustness. The proposed CLEVER score is attack-agnostic and computationally feasible for large neural networks. Experimental results on various networks, including ResNet, Inception-v3 and MobileNet, show that (i) CLEVER is aligned with the robustness indication measured by the $\\}ell_2$ and $\\}ell\_\\}infty$ norms of adversarial examples from powerful attacks, and (ii) defended networks using defensive distillation or bounded ReLU indeed achieve better CLEVER scores. To the best of our knowledge, CLEVER is the first attack-independent robustness metric that can be applied to any neural network classifier. (@weng2018evaluating)

Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, and Benjamin Recht In *NIPS*, 2017. **Abstract:** Adaptive optimization methods, which perform local optimization with a metric constructed from the history of iterates, are becoming increasingly popular for training deep neural networks. Examples include AdaGrad, RMSProp, and Adam. We show that for simple overparameterized problems, adaptive methods often find drastically different solutions than gradient descent (GD) or stochastic gradient descent (SGD). We construct an illustrative binary classification problem where the data is linearly separable, GD and SGD achieve zero test error, and AdaGrad, Adam, and RMSProp attain test errors arbitrarily close to half. We additionally study the empirical generalization capability of adaptive methods on several state-of-the-art deep learning models. We observe that the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even when these solutions have better training performance. These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks. (@Wilson2017TheMV)

Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh *Advances in Neural Information Processing Systems*, 33: 1129–1141, 2020. **Abstract:** Linear relaxation based perturbation analysis (LiRPA) for neural networks, which computes provable linear bounds of output neurons given a certain amount of input perturbation, has become a core component in robustness verification and certified defense. The majority of LiRPA-based methods focus on simple feed-forward networks and need particular manual derivations and implementations when extended to other architectures. In this paper, we develop an automatic framework to enable perturbation analysis on any neural network structures, by generalizing existing LiRPA algorithms such as CROWN to operate on general computational graphs. The flexibility, differentiability and ease of use of our framework allow us to obtain state-of-the-art results on LiRPA based certified defense on fairly complicated networks like DenseNet, ResNeXt and Transformer that are not supported by prior works. Our framework also enables loss fusion, a technique that significantly reduces the computational complexity of LiRPA for certified defense. For the first time, we demonstrate LiRPA based certified defense on Tiny ImageNet and Downscaled ImageNet where previous approaches cannot scale to due to the relatively large number of classes. Our work also yields an open-source library for the community to apply LiRPA to areas beyond certified defense without much LiRPA expertise, e.g., we create a neural network with a probably flat optimization landscape by applying LiRPA to network parameters. Our opensource library is available at https://github.com/KaidiXu/auto_LiRPA. (@xu2020automatic)

Tan Yu, Jun Li, Yunfeng Cai, and Ping Li In *International Conference on Learning Representations*, 2021. (@Yu2022ConstructingOC)

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals Understanding deep learning (still) requires rethinking generalization *Communications of the ACM*, 64 (3): 107–115, 2021. **Abstract:** Despite their massive size, successful deep artificial neural networks can exhibit a remarkably small gap between training and test performance. Conventional wisdom attributes small generalization error either to properties of the model family or to the regularization techniques used during training. Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected by explicit regularization and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters exceeds the number of data points as it usually does in practice. We interpret our experimental findings by comparison with traditional models. We supplement this republication with a new section at the end summarizing recent progresses in the field since the original version of this paper. (@zhang2021understanding)

Zhiming Zhou, Yuxuan Song, Lantao Yu, and Yong Yu Understanding the effectiveness of lipschitz constraint in training of gans via gradient analysis *CoRR*, abs/1807.00751, 2018. URL <http://arxiv.org/abs/1807.00751>. **Abstract:** In this paper, we investigate the underlying factor that leads to failure and success in the training of GANs. We study the property of the optimal discriminative function and show that in many GANs, the gradient from the optimal discriminative function is not reliable, which turns out to be the fundamental cause of failure in training of GANs. We further demonstrate that a well-defined distance metric does not necessarily guarantee the convergence of GANs. Finally, we prove in this paper that Lipschitz-continuity condition is a general solution to make the gradient of the optimal discriminative function reliable, and characterized the necessary condition where Lipschitz-continuity ensures the convergence, which leads to a broad family of valid GAN objectives under Lipschitz-continuity condition, where Wasserstein distance is one special case. We experiment with several new objectives, which are sound according to our theorems, and we found that, compared with Wasserstein distance, the outputs of the discriminator with new objectives are more stable and the final qualities of generated samples are also consistently higher than those produced by Wasserstein distance. (@Zhou2018UnderstandingTE)

Zhiming Zhou, Jiadong Liang, Yuxuan Song, Lantao Yu, Hongwei Wang, Weinan Zhang, Yong Yu, and Zhihua Zhang In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pp. 7584–7593. PMLR, 2019. URL <http://proceedings.mlr.press/v97/zhou19c.html>. **Abstract:** In this paper, we study the convergence of generative adversarial networks (GANs) from the perspective of the informativeness of the gradient of the optimal discriminative function. We show that GANs without restriction on the discriminative function space commonly suffer from the problem that the gradient produced by the discriminator is uninformative to guide the generator. By contrast, Wasserstein GAN (WGAN), where the discriminative function is restricted to 1-Lipschitz, does not suffer from such a gradient uninformativeness problem. We further show in the paper that the model with a compact dual form of Wasserstein distance, where the Lipschitz condition is relaxed, may also theoretically suffer from this issue. This implies the importance of Lipschitz condition and motivates us to study the general formulation of GANs with Lipschitz constraint, which leads to a new family of GANs that we call Lipschitz GANs (LGANs). We show that LGANs guarantee the existence and uniqueness of the optimal discriminative function as well as the existence of a unique Nash equilibrium. We prove that LGANs are generally capable of eliminating the gradient uninformativeness problem. According to our empirical analysis, LGANs are more stable and generate consistently higher quality samples compared with WGAN. (@DBLP:journals/corr/abs-1902-05687)

</div>

# Appendix

### Table of notations

#### Basic notations

<div id="tab:basic_notations">

| Notation | Definition |
|:---|:--:|
| $`C`$ | Lipschitz constant |
| $`f`$ or $`f_{{\pmb \theta}}`$ | Neural Network, function at hand |
|  |  |
| $`{\pmb \theta}\in \Omega`$ | Parameter vector |
| $`\Omega\subseteq \mathbb R^p`$ | Parameter space |
|  |  |
| $`\mathbf x\in \mathcal {D}^+`$ | Input/data vector |
| $`\mathcal {D}^+\subseteq \mathbb R^d`$ | Input/data space and its neighbourhood |
|  |  |
| $`K`$ | Number of function outputs/classes |
|  |  |
| $`\mathcal{D}`$ | Training set |
| $`\mathcal{D_{\mathrm{test}}}`$ | Test set |
| $`{\mathcal{L}}`$ | Loss function |

A table of basic notations.

</div>

<span id="tab:basic_notations" label="tab:basic_notations"></span>

#### Lipschitz notations

<div id="tab:lipschitz_notations">

| Notation of the bound | Parameter space | Input space | Formula |
|:---|:--:|:--:|:--:|
| True Lipschitz, $`C_\text{true}`$ | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in dom(f)`$ | $`\sup_{\mathbf x\in dom(f)}\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x)\|`$ |
| Effective Lipschitz, $`C_{\mathcal{D}^+}`$ | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in \mathcal {D}^+`$ | $`\sup_{\mathbf x\in \mathcal{D}^+}\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x)\|`$ |
| Lower Lipschitz, $`C_\text{lower}`$ | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in \mathcal{D}`$ | $`\sup_{\mathbf x\in \mathcal{D}}\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x)\|`$ |
| Local Lipschitz | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in dom(f)`$ | $`\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x)\|`$ |
|  |  |  |  |
| Average, $`C_\text{avg}`$ | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in \mathcal{D}`$ | $`\frac{1}{|\mathcal{D}|} \sum_{\mathbf x\in \mathcal{D}}\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x)\|`$ |
| Alt. lower, $`C_\text{alt.lower}`$ | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x,\mathbf y\in \mathcal{D}`$ | $`\sup_{\mathbf x, \mathbf y\in \mathcal{D}, \mathbf x\ne \mathbf y}\frac{\|f({\pmb \theta}^t, \mathbf x)-f({\pmb \theta}^t, \mathbf y)\|}{\|\mathbf x-\mathbf y\|}`$ |
|  |  |  |  |
| Adversarial lower | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in \mathcal{D}`$ | $`\sup_{\mathbf x\in \mathcal{D}}\|\nabla_\mathbf xf({\pmb \theta}^t, \mathbf x+\bm{\varepsilon})\|`$ |
| Adversarial alt. lower | at epoch $`t`$, i.e. $`{\pmb \theta}^t`$ | $`\mathbf x\in \mathcal{D}`$ | $`\sup_{\mathbf x\in \mathcal{D}}\frac{\|f(\theta^t, \mathbf x)-f(\theta^t, \mathbf x+\bm{\varepsilon})\|}{\|\bm{\varepsilon}\|}`$ |

A table of Lipschitz constant notations.

</div>

<span id="tab:lipschitz_notations" label="tab:lipschitz_notations"></span>

### Experimental setup

This section contains comprehensive details on the experiments in the paper — this includes a summary of our strategy for Lipschitz upper bound calculation, models’ architecture descriptions, hyperparameters and optimisation strategy choices for every experimental section in the main and supplementary parts of the paper. We also release our code on [GitHub](https://github.com/gakhromov/lipschitz-continuity-of-nns) for further reproducibility and transparency.

In short, we benchmark our findings in a wide variety of settings, with different choices of (a) *architectures:* fully-connected networks (FCNs), convolutional neural networks (CNNs), Residual networks (ResNets) and Vision Transformers (ViT); (b) *datasets:* CIFAR-10 and CIFAR-100 , MNIST, MNIST1D  (a harder version of usual MNIST), as well as ImageNet; (c) *loss functions:* Cross-Entropy (CE) and Mean-Squared Error (MSE) loss. Unless stated otherwise, the results in the main paper are based on stochastic gradient descent with CE loss and the metrics have been averaged over 4 runs. All plots with shaded regions represent an uncertainty of $`\pm`$ standard deviation from the mean, which is computed across different seeds. When semi-transparent dotted lines are shown, the solid line represents the mean over seeds and each dotted line depicts data from individual seeds.

#### Local Lipschitz estimates vs. pairs of points

As mentioned in Section <a href="#preliminaries-and-setup" data-reference-type="ref" data-reference="preliminaries-and-setup">2</a>, another way to lower bound is to consider Definition <a href="#def:lip-const" data-reference-type="ref" data-reference="def:lip-const">1</a> and restrict the supremum computation to the train set. In essence, we want to compare the following lower bound estimates:
``` math
\begin{aligned}
    C_\text{lower} = \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf_{{\pmb \theta}}\| & & \text{and} & & C_\text{alt.lower} = \sup_{\mathbf x,\mathbf y\in\mathcal{D},\mathbf x\ne\mathbf y} \frac{\|f_{{\pmb \theta}}(\mathbf x)-f_{{\pmb \theta}}(\mathbf y)\|}{\|\mathbf x-\mathbf y\|}\,.
\end{aligned}
```
These bounds only converge when the considered set includes the whole function domain. Therefore it is not trivial to say which bound is better when a numerical estimation on a subset is performed.

To address this issue, we designed a simple experiment where we computed $`C_\text{lower}`$ and $`C_\text{alt.lower}`$ for an FCN ReLU network from Section <a href="#sec:lip_evol" data-reference-type="ref" data-reference="sec:lip_evol">3.1</a> for several training epochs. The results of this experiment, displayed in Table <a href="#tab:straightforward-vs-jacnorm" data-reference-type="ref" data-reference="tab:straightforward-vs-jacnorm">3</a>, provide evidence for the fact that the $`C_\text{lower}`$ estimate is tighter than $`C_\text{alt.lower}`$.

<div id="tab:straightforward-vs-jacnorm">

|  Epoch | $`C_\text{lower}`$ | $`C_\text{alt. lower}`$ |
|-------:|:------------------:|:-----------------------:|
|      0 |       0.369        |          0.165          |
|  1 000 |       1.336        |          0.865          |
|  2 000 |       3.762        |          2.752          |
|  3 000 |       5.775        |          4.257          |
|  4 000 |       7.142        |          5.120          |
|  5 000 |       7.974        |          5.591          |
| 10 000 |       9.928        |          6.361          |
| 50 000 |       19.390       |          8.840          |

Comparison of two Lipschitz lower bound estimates. Computed for one seed.

</div>

<span id="tab:straightforward-vs-jacnorm" label="tab:straightforward-vs-jacnorm"></span>

#### Upper bound calculation

We start by describing our approach to computing the upper bound of the Lipschitz constant, inspired by the AutoLip algorithm introduced by . As briefly discussed in the main part of the paper (Section <a href="#preliminaries-and-setup" data-reference-type="ref" data-reference="preliminaries-and-setup">2</a>), we simply multiply per-layer Lipschitz bounds. For network $`f_{{\pmb \theta}}`$ with $`L`$ layers, defined as $`f_{{\pmb \theta}}:= f^{(L)} \circ \sigma \circ f^{(L-1)} \circ \sigma \circ \dots \circ f^{(1)}`$, where $`\sigma`$ is a 1-Lipschitz non-linear activation function, the upper bound looks as follows:
``` math
C \, \le\,  \prod_{i=1}^L \, \sup_{{\mathbf x^{(i-1)}}\in dom(f^{(i)})}\|\nabla_{\mathbf x^{(i-1)}} f^{(i)}\| \le \prod_{i=1}^L \sup \|\nabla_{\mathbf x^{(i-1)}} f^{(i)}\| =: C_{\text{upper}} \tag{\ref{eq:lip-upper-bound}}
```

More pedantically, one can see this by using Theorem <a href="#lipschitz_as_jac_norm" data-reference-type="ref" data-reference="lipschitz_as_jac_norm">1</a> and applying the chain rule:
``` math
\begin{aligned}
    C &= \sup_{\mathbf x\in dom(f_{{\pmb \theta}})}\| \nabla_\mathbf xf_{{\pmb \theta}}\|  = \sup_{\mathbf x\in dom(f_{{\pmb \theta}})}\| \nabla_{f^{(L-1)}}f^{(L)} \cdot \nabla_{f^{(L-2)}}f^{(L-1)} \cdot \; \dots\; \cdot \nabla_{f^{(1)}}f^{(2)} \cdot \nabla_\mathbf xf^{(1)} \| \nonumber \\
    & \le \sup_{f^{(L-1)}(\mathbf x)}\| \nabla_{f^{(L-1)}}f^{(L)} \| \cdot \;\dots\; \cdot \sup_{f^{(1)}(\mathbf x)} \| \nabla_{f^{(1)}}f^{(2)} \| \cdot \sup_{\mathbf x\in dom(f_{{\pmb \theta}})}\| \nabla_\mathbf xf^{(1)} \| \nonumber \\
    & \le \sup\| \nabla_{f^{(L-1)}}f^{(L)} \| \cdot \; \dots\;\cdot \sup \| \nabla_{f^{(1)}}f^{(2)} \| \cdot \sup\| \nabla_\mathbf xf^{(1)} \| =: C_\text{upper}\,,
\end{aligned}
```

where in the last line we consider the unconstrained supremum.

###### Linear operations.

Each linear layer of the form $`f^{(i)}(\mathbf x) = \mathbf W^{(i)}\, \mathbf x`$ has Lipschitz constant $`\|\mathbf W^{(i)}\|_2`$, since the Jacobian of $`f^{(i)}`$ is simply the weight matrix. Convolutional layers are also linear operators and therefore can be similarly expressed as a linear transformation  (considering that we perform proper flattening of the input image and reshaping in the end). According to , equivalent linear transformations are represented by doubly block Toeplitz matrices which only depend on kernel weights. At the same time, Batch Normalisation layers are also per-feature linear transformations and, thus, the upper bound can be calculated as a maximum across per-feature Lipschitz constants.

###### Activation functions.

All activation functions that are used in this paper — ReLU, Leaky ReLU with slope $`0.01`$, max and average pooling — are also at most 1-Lipschitz and thus are considered 1-Lipschitz for the upper bound computation.

###### Residual layers.

Residual layers of the form $`f(\mathbf x) = g(\mathbf x) + \mathbf x`$, where $`f,g: \mathbb R^n\to\mathbb R^n`$ and $`g`$ is $`C_g`$-Lipschitz continuous, simply have a Lipschitz constant equal to $`C_f=C_g+1`$. One can trivially derive this using the definition of Lipschitz continuity and the triangle inequality:
``` math
\begin{aligned}
    \|f(\mathbf x) - f(\mathbf y)\| & = \|g(\mathbf x) - g(\mathbf y) + \mathbf y- \mathbf x\| \nonumber \\
    & \le \|g(\mathbf x)-g(\mathbf y)\| + \|\mathbf y-\mathbf x\| \tag{Triangle inequality} \\
    & \le C_g\|\mathbf x-\mathbf y\| + \|\mathbf x-\mathbf y\| \nonumber \\
    & = (C_g+1)\|\mathbf x-\mathbf y\|
\end{aligned}
```

###### Attention layers.

According to , standard Attention layers are not Lipschitz continuous (or, in other words, have an unbounded Lipschitz constant) for unconstrained inputs. Therefore the upper bound introduced in Equation <a href="#eq:lip-upper-bound" data-reference-type="ref" data-reference="eq:lip-upper-bound">[eq:lip-upper-bound]</a> is not applicable. Due to this reason, we only compute the lower Lipschitz bound in all experiments with Transformers.

*Remark on computational challenges.* Since equivalent linear weight matrices for convolutional operations assume flat input, the dimensions of this matrix grow rapidly with increasing image size and channel depth. This results in large memory consumption, which we leveraged by converting this matrix into the `scipy` sparse CSR format and using `scipy.sparse.linalg` library to compute the norm. We also found that standard `pytorch` implementation of 2-norm computation works rather slowly for large matrices. We have therefore implemented a Power method to compute 2-norms for linear layers and batched Jacobian matrices, resulting in almost $`10\times`$ speedup in some cases.

#### Model descriptions

##### FCN ReLU networks

A fully Connected Network with ReLU activations (or FCN ReLU for short) consists of a sequence of Linear layers with zero bias term, each followed by a ReLU activation layer, the last layer included. When it is specified that FCN ReLU has a width of 256, there are only two linear transformations involved: first, from an input vector to a hidden vector of size 256, and second, from a hidden layer of size 256 to the output dimension. When a sequence of widths is given (i.e. FCN ReLU with widths 64,64), FCN has several hidden layers, sizes of which are listed from the hidden layer closer to the input to the hidden layer closer to the output.

*Remark on the Dead ReLU problem.* Since we use ReLU as the last layer’s activation, outputs of the model can in some cases become zero-vectors, specifically in scenarios with classification using MSE loss. To address this issue we use a modified version of the FCN model for all MSE experiments — the last ReLU activation is substituted with a Leaky ReLU function with a negative slope of $`0.01`$.

Table <a href="#tab:ff_relu_params" data-reference-type="ref" data-reference="tab:ff_relu_params">4</a> shows a list of FCN ReLU realisations with various hidden layer widths and depths and the respective number of model parameters. All models in this table are configured for the MNIST1D[^5] dataset.

<div id="tab:ff_relu_params">

|       Model name        | Number of parameters |
|:-----------------------:|:--------------------:|
|       FCN ReLU 16       |         800          |
|       FCN ReLU 32       |        1,600         |
|            ⋮            |          ⋮           |
|       FCN ReLU 80       |        4,000         |
|            ⋮            |          ⋮           |
|      FCN ReLU 256       |        12,800        |
|            ⋮            |          ⋮           |
|      FCN ReLU 800       |        40,000        |
|            ⋮            |          ⋮           |
|     FCN ReLU 131072     |      6,553,600       |
|     FCN ReLU 64,64      |        7,296         |
|    FCN ReLU 64,64,64    |        11,392        |
|  FCN ReLU 64,64,64,64   |        15,488        |
| FCN ReLU 64,64,64,64,64 |        19,584        |

Table of the number of parameters of FCN ReLU models of various widths and depths, configured for the MNIST1D dataset.

</div>

<span id="tab:ff_relu_params" label="tab:ff_relu_params"></span>

##### CNN networks

Our version of Convolutional Neural Networks (or CNNs for short) follows an approach similar to . We consider a 5-layer model, with 4 Conv-ReLU-MaxPool blocks, followed by a Linear layer with zero bias. All convolution layers have $`3\times 3`$ kernels with stride=1, padding=1 and zero bias. Kernel output channels for convolution layers follow the pattern $`[w, 2w, 4w, 8w]`$, where $`w`$ is the width of the model. Meanwhile, MaxPooling layers have kernels of sizes $`[1, 2, 2, 8]`$ for the case of CIFAR-10[^6] and $`[1, 2, 2, 7]`$ for the case of MNIST[^7]. This configuration shrinks the input image to a single vector that is then passed to the last Linear layer, yielding the output vector of size 10.

Table <a href="#tab:cnn_params" data-reference-type="ref" data-reference="tab:cnn_params">5</a> displays a list of CNN realisations with various hidden layer widths and the respective number of model parameters. Since configurations for CIFAR-10  and MNIST differ only in the MaxPooling layer size, the number of trainable parameters remains the same for both datasets.

<div id="tab:cnn_params">

| Model name | Number of parameters |
|:----------:|:--------------------:|
|   CNN 5    |        9,985         |
|   CNN 7    |        19,271        |
|   CNN 10   |        38,870        |
|   CNN 11   |        46,915        |
|   CNN 12   |        55,716        |
|   CNN 13   |        65,273        |
|     ⋮      |          ⋮           |
|   CNN 20   |       153,340        |
|     ⋮      |          ⋮           |
|   CNN 60   |      1,367,220       |

Table of the number of parameters of the CNN network of various widths.

</div>

<span id="tab:cnn_params" label="tab:cnn_params"></span>

#### Training strategy

We present experiments that compare models that have a substantially varying number of parameters. To minimise the effect of variability in training, we painstakingly enforce the same learning rate, batch size and optimiser configuration for all models in one sweep. While this choice makes our claims on the behaviour of the Lipschitz constant stronger, we now have the challenge of setting a reasonable stopping criterion to manage variable convergence rates.

For model $`f_{{\pmb \theta}}`$ with parameter vector $`{\pmb \theta}`$ and loss on the training set $`\mathcal L({\pmb \theta}, \mathcal{D})`$, after the end of each epoch we compute $`\| \nabla_{{\pmb \theta}} \mathcal L({\pmb \theta}, \mathcal{D}) \|_2`$, which we call gradient norm for simplicity. In all experiments, unless stated otherwise, we control model training by monitoring the respective gradient norm — if it reaches a small value (ideally zero), our model has negligible parameter change (i.e. $`\|{\pmb \theta}^{t+1}-{\pmb \theta}^{t}\|_2`$ is small) or, in other words, has reached a local minimum. By means of experimentation, we found that stopping models at 0.01 gradient norm value gives good results for most scenarios.

*Possible pitfalls.*<span id="training-strategy-pitfalls" label="training-strategy-pitfalls"></span> In most cases, during the course of training, the gradient norm is at first relatively small (in the range of $`10^{-4}`$ to $`10^{-2}`$), then rapidly increases and then slowly decreases. If our only stopping criteria is minuscule gradient norm, we may end up early stopping models right after a few epochs. To avoid this we also introduce a minimum number of epochs that each model has to train for.

#### Learning rate schedulers

This section thoroughly describes learning rate schedulers (LR schedulers for short) that we use in our experiments. Each scheduler modifies a variable $`\gamma_t`$, which is a coefficient that is multiplied by some base learning rate (i.e. learning rate at time step $`t`$ is $`\eta_t=\gamma_t\cdot\eta`$, where $`\eta`$ is the base learning rate). Note that we perform scheduler updates at every parameter update (which happens several times per epoch), and the schedulers are aware of the dataset length and batch size to adapt to epoch-based settings accordingly.

###### Warmup20000Step25 LR Scheduler (Figure <a href="#fig:warmup20000step25" data-reference-type="ref" data-reference="fig:warmup20000step25">9</a>).

This scheduler linearly scales the learning rate from a factor of $`\frac{1}{20,000}`$ to 1 for 20,000 updates. Next, the scaling coefficient drops by a factor of 0.75 every 25% of the next 10,000 epochs. Afterwards, the coefficient remains at the constant factor of $`0.75^3=0.421875`$.

###### Cont100 LR Scheduler (Figure <a href="#fig:cont100" data-reference-type="ref" data-reference="fig:cont100">11</a>).

This scheduler continuously drops the scaling coefficient by a factor of 0.95 every 100 epochs.

<figure id="fig:cont100">
<figure id="fig:warmup20000step25">
<img src="./figures/lr_scheduler%2BWarmup20000Step25%2Bupd_per_epoch_10.png"" />
<figcaption>Warmup20000Step25 LR scheduler</figcaption>
</figure>
<figure id="fig:cont100">
<img src="./figures/lr_scheduler%2BCont100%2Bupd_per_epoch_10.png"" />
<figcaption>Cont100 LR scheduler</figcaption>
</figure>
<figcaption>Learning rate scaling coefficient <span class="math inline"><em>γ</em><sub><em>t</em></sub></span> against updates for two different learning rate schedulers. In each case there are 10 updates per epoch.</figcaption>
</figure>

#### Description of experimental settings

This section presents a detailed description of each experimental setup. We also include theoretical Double Descent interpolation thresholds for Double Descent experiments, where in each equation $`n`$ stands for the number of training samples, $`K`$ for the number of classes and $`p`$ for the number of parameters in the model.

##### A simple toy example to demonstrate the fidelity of the lower Lipschitz bound (Figure <a href="#fig:visual-example" data-reference-type="ref" data-reference="fig:visual-example">4</a>)

For this experiment, we set our data domain to be $`\mathcal D = [-5, 5]^2`$, and the three equally spaced multinomial Gaussians. Each Gaussian’s mean is $`1.5`$ units away from the origin and has $`\Sigma=I_2`$. We sampled 15 points from each Gaussian, resulting in a dataset of $`45`$ points. The dataset with the true labels is depicted in Figure <a href="#fig:visual-example-data" data-reference-type="ref" data-reference="fig:visual-example-data">12</a>.

<figure id="fig:visual-example-data">
<img src="./figures/visual_example_dataset.png"" />
<figcaption>The dataset and the true labels.</figcaption>
</figure>

For the classifier, we chose an FCN ReLU Network with $`2`$ hidden layers of width $`100`$. We optimised the model on Cross-Entropy loss with Gradient Descent and a constant learning rate of $`0.02`$ for $`100{,}000`$ epochs. To compute the Lipschitz bounds, we sampled $`1{,}000{,}000`$ points on the $`[-5,5]^2`$ grid. Additionally, we also computed the Jacobian norms outside of the data domain, i.e. on the $`[-10,10]^2`$ grid, which resulted in the same estimate of $`144.194`$. The results for the local Lipschitz computation are shown in Figure <a href="#fig:visual-example-lower-lip-space" data-reference-type="ref" data-reference="fig:visual-example-lower-lip-space">13</a>.

<figure id="fig:visual-example-lower-lip-space">
<figure>
<img src="./figures/visual_example_lower_lip.png"" />
<figcaption>Local Lip. for <span class="math inline">𝒟 = [−5, 5]<sup>2</sup></span></figcaption>
</figure>
<figure>
<img src="./figures/visual_example_lower_lip_ood.png"" />
<figcaption>Local Lip. for <span class="math inline">[−10, 10]<sup>2</sup></span></figcaption>
</figure>
<figcaption>Local Lipschitz constants of <span class="math inline"><em>f</em></span> for various input domains.</figcaption>
</figure>

##### Double Descent on MNIST1D, FCN ReLU networks, Cross-Entropy loss (Figure <a href="#fig:lip_dd" data-reference-type="ref" data-reference="fig:lip_dd">22</a>)

For this experiment, we trained a sweep of FCN ReLU models (see <a href="#desc-fcn-relu-network" data-reference-type="ref" data-reference="desc-fcn-relu-network">2.3.1</a>) with widths \[$`16`$, $`32`$, $`64`$, $`80`$, $`96`$, $`128`$, $`256`$, $`512`$, $`1024`$, $`2048`$, $`4096`$, 8192, $`16384`$, $`32768`$, $`65536`$, $`131072`$\] on MNIST1D[^8] with batch size 512 using Cross-Entropy loss and SGD optimiser without momentum. We used a Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.005. We trained our models for at least 10,000 epochs and stopped each model when either 0.01 gradient norm is reached or when 300,000 epochs have passed. We trained 4 seeds for each run. The theoretical threshold for this scenario is at $`n\approx p`$, which corresponds to FCN ReLU 80 (4000 parameters).

*Comment on the hyperparameter choice.* In this experiment, we used a very small learning rate to smoothly fit both under- and over-parametrised models. Therefore we require a significant amount of training epochs to secure convergence for all settings.

*Comment on Figure <a href="#fig:lip_dd" data-reference-type="ref" data-reference="fig:lip_dd">22</a>.* In the figure, the training loss uncertainty for models from width 8192 to 65536 is lower bounded by zero (since training loss cannot be negative) and therefore is depicted as a vertical line in the log-log plot.

##### Double Descent on MNIST1D, FCN ReLU networks, MSE loss (Figure <a href="#fig:lip_dd_mse" data-reference-type="ref" data-reference="fig:lip_dd_mse">23</a>)

This experiment depicts a sweep of FCN ReLU models (see <a href="#desc-fcn-relu-network" data-reference-type="ref" data-reference="desc-fcn-relu-network">2.3.1</a>) with widths \[$`16`$, $`32`$, $`64`$, $`80`$, $`96`$, $`128`$, $`256`$, $`512`$, $`1024`$, $`2048`$, $`4096`$, 8192, $`16384`$, $`32768`$, $`65536`$\], trained on MNIST1D with batch size 512 using MSE loss and SGD optimiser without momentum. We used a Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.001. We trained our models for at least 10,000 epochs and stopped each model when 0.01 gradient norm is reached or when 200,000 epochs have passed. We trained 4 seeds for each run. The theoretical threshold for this scenario is at $`n\cdot K\approx p`$, which corresponds to FCN ReLU 800 (40,000 parameters).

##### Double Descent on CIFAR-10, CNN networks, Cross-Entropy loss (Figure <a href="#fig:lip_dd_cnn_cifar" data-reference-type="ref" data-reference="fig:lip_dd_cnn_cifar">24</a>)

For this experiment, we trained a sweep of CNN models with widths \[$`5`$, $`7`$, $`10`$, $`11`$, $`12`$, $`15`$, $`20`$, $`25`$, $`30`$, $`35`$, $`40`$, $`45`$, $`50`$, $`55`$, $`60`$\] on CIFAR-10[^9] with batch size 128 using Cross-Entropy loss and SGD optimiser without momentum. We used a Cont100 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.01. We trained our models for at least 500 epochs and stopped each model when 0.01 gradient norm is reached or when 5000 epochs have passed. We trained 4 seeds for each run. The theoretical threshold for this scenario is at $`n \approx p`$, which corresponds to somewhere between CNN 11 (46,915 parameters) and CNN 12 (55,716 parameters).

##### Double Descent on CIFAR-100 with 20 superclasses, CNN networks, Cross-Entropy loss (Figure <a href="#fig:lip_dd_cnn_cifar100c20" data-reference-type="ref" data-reference="fig:lip_dd_cnn_cifar100c20">5</a>)

For this experiment, we trained a sweep of CNN models with widths \[$`5`$, $`7`$, $`10`$, $`11`$, $`12`$, $`13`$, $`15`$, $`20`$, $`25`$, $`30`$, $`35`$, $`40`$, $`45`$, $`50`$, $`55`$, $`60`$\] on CIFAR-100[^10] with 20 superclasses, as described by  with batch size 128 using Cross-Entropy loss and SGD optimiser without momentum. We used a Cont100 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.005. We trained our models for at least 500 epochs and stopped each model when 0.01 gradient norm is reached or when 15 000 epochs have passed. We trained 4 seeds for each run. The theoretical threshold for this scenario is at $`n \approx p`$, which corresponds to somewhere between CNN 11 (47 795 parameters) and CNN 12 (56 676 parameters).

##### Double Descent on MNIST, CNN networks, Cross-Entropy loss (Figure <a href="#fig:lip_dd_cnn_mnist" data-reference-type="ref" data-reference="fig:lip_dd_cnn_mnist">25</a>)

For this experiment, we trained a sweep of CNN models with widths \[$`5`$, $`7`$, $`10`$, $`11`$, $`12`$, $`13`$, $`15`$, $`20`$, $`25`$, $`30`$, $`35`$, $`40`$, $`45`$, $`50`$, $`55`$, $`60`$\] on MNIST[^11] with a fixed 10% label shuffling (i.e. the dataset is the same across all seeds and models) with batch size 128 using Cross-Entropy loss and SGD optimiser without momentum. We used a Cont100 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.01. We trained our models for at least 100 epochs and stopped each model when 0.01 gradient norm is reached or when 1000 epochs have passed. We train 4 seeds for each run. The theoretical threshold for this scenario is at $`n \approx p`$, which corresponds to somewhere between CNN 12 (55,500 parameters) and CNN 13 (65,039 parameters). Note that the parameter count is different due to different input size.

##### Double Descent on CIFAR-10, Vision Transformer (ViT) networks (Figure <a href="#fig:lip_dd_vit_cifar10" data-reference-type="ref" data-reference="fig:lip_dd_vit_cifar10">26</a>)

For this experiment, we trained a sweep of ViT models with widths \[$`3`$, $`5`$, $`7`$, $`8`$, $`10`$, $`11`$, $`12`$, $`13`$, $`14`$, $`15`$, $`16`$, $`17`$, $`18`$, $`19`$, $`20`$, $`50`$, $`100`$, $`500`$\]. The width is used as the last dimension of output tensor after linear transformation, as well the dimension of the FeedForward layer (we multiply it by 4 in this case). No dropout is used. The patch size is equal to 8, and we use 6 heads with 6 Transformer blocks. We used the `vit-pytorch` Python package implementation.

All models were trained on CIFAR-10[^12] with batch size 128 using Cross-Entropy loss and SGD optimiser without momentum. We used a Step LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>) and a base learning rate of 0.05. We trained our models for at least 2,000 epochs and stopped each model when 0.01 gradient norm is reached or when 10,000 epochs have passed. We train 4 seeds for each run. The theoretical threshold for this scenario is at $`n \approx p`$, which corresponds to somewhere near ViT 5 (49,099 parameters).

*Remark.* Since the Attention layers are not Lipschitz continuous , or, rather, the upper bound on the Lipschitz constant does not exist in $`\mathbb R^d`$, we only present the results for the lower Lipschitz bounds. In order to provide proper upper bounds, modification to Attention layers are required, as described in detail by .

##### Lipschitz bounds evolution, FCN ReLU, MNIST1D with convex combinations (Fig. <a href="#fig:lip_bounds_evolution" data-reference-type="ref" data-reference="fig:lip_bounds_evolution">2</a>)

To showcase Lipschitz constant evolution we trained 4 FCN ReLU 256 with different seeds on the MNIST1D dataset with batch size 512 for 83,000 epochs. Models were trained using Cross-Entropy loss and SGD optimiser with a base learning rate of 0.005 and Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>). Each model achieved gradient norm of 0.01 up to 2 significant figures. Each epoch consists of 8 parameter updates.

For this scenario, we empirically defined the stable phase to begin from epoch 2500 (or after 20,000 updates). The slopes for the upper, lower, and average Lipschitz bounds are: 0.59, 0.46 and 0.44 respectively and are computed by examining the slope coefficient of a linear regression model fitted to the corresponding values in the log-log scale. The $`R^2`$ values of the fit for the upper, lower, and average Lipschitz bounds are 0.9955, 0.9986 and 0.9980 respectively.

To compute the lower bound on convex combinations of samples from MNIST1D we constructed a set $`S^*`$, which contains: (a) training set $`\mathcal{D}`$ — 4000 samples, (b) test set $`\mathcal{D_{\mathrm{test}}}`$ — 1000 samples, (c) convex combinations $`\lambda \mathbf x_i + (1-\lambda)\mathbf x_j`$ from $`\mathcal{D}`$ — 100,000 samples for each $`\lambda=\{0.1, 0.2, 0.3, 0.4, 0.5\}`$, and (d) convex combinations $`\lambda \mathbf x_i + (1-\lambda)\mathbf x_j`$ from $`\mathcal{D_{\mathrm{test}}}`$ — 100,000 samples for each $`\lambda=\{0.1, 0.2, 0.3, 0.4, 0.5\}`$. Altogether this makes $`S^*`$ contain 1,005,000 samples.

##### Lipschitz bounds evolution, ResNet50, subset of 200,000 ImageNet samples (Figure <a href="#fig:resnet50-evolution" data-reference-type="ref" data-reference="fig:resnet50-evolution">[fig:resnet50-evolution]</a>)

For this experiment, we trained 3 ResNet50 models with different seeds for 90 epochs on full ImageNet with batch size 256, using Cross-Entropy loss and SGD optimiser with momentum of 0.9, weight decay of 0.0001 and base learning rate 0.1. We also used a LR decay scheme where the learning rate is decreased 10 times every 30 epochs. We then evaluated lower Lipschitz bounds for epochs \[$`0`$, $`1`$, $`10`$, $`20`$, $`30`$, $`40`$, $`50`$, $`60`$, $`70`$, $`80`$, $`90`$\] on a fixed random subset of 200,000 images from the ImageNet training set. During Lipschitz evaluation, all training samples are resized to $`256\times256\times3`$, then center-cropped to size $`224\times224\times3`$ and then normalised using $`mean = [0.485, 0.456, 0.406]`$ and $`std = [0.229, 0.224, 0.225]`$. During training, training samples are randomly resized and cropped to $`224\times224\times3`$ and then normalised using the same $`mean`$ and $`std`$.

For this scenario, we empirically defined the stable phase to begin from epoch 60. The slopes for the upper, lower, and average Lipschitz bounds are: 7.32, 0.49 and 0.48 respectively and are computed by examining the slope coefficient of a linear regression model fitted to the corresponding values in the log-log scale. The $`R^2`$ values of the fit for the upper, lower, and average Lipschitz bounds are 0.8441, 0.9718 and 0.8774 respectively.

##### Distribution of the norm of the per-sample Jacobian for ResNet18 trained on ImageNet (Figure <a href="#fig:norm-distr-imagenet-resnet18" data-reference-type="ref" data-reference="fig:norm-distr-imagenet-resnet18">18</a>)

For this experiment we first evaluated the norms of the Jacobian matrices (see <a href="#lipschitz_as_jac_norm" data-reference-type="ref" data-reference="lipschitz_as_jac_norm">1</a>) of ResNet18 for each sample of ImageNet. We took a pretrained ResNet18 from [`pytorch hub`](https://pytorch.org/hub/pytorch_vision_resnet/). We then constructed another dataset, which took a mean of all possible pairs of 1,000 ImageNet samples with the highest Jacobian norm from the previous calculation, and evaluated the distribution once more on the new dataset.

##### Variance upper bounds and Bias-Variance tradeoff (Sections <a href="#bias-variance-tradeoff-argument" data-reference-type="ref" data-reference="bias-variance-tradeoff-argument">[bias-variance-tradeoff-argument]</a> and <a href="#variance_upper_bounds" data-reference-type="ref" data-reference="variance_upper_bounds">3.11</a>)

For this study we used the same models that we have trained for the Double Descent on MNIST1D using FCN ReLU networks trained with MSE setting (see <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a>).

To compute variance bound estimates (see Eq. <a href="#eq:var_upper:1" data-reference-type="ref" data-reference="eq:var_upper:1">[eq:var_upper:1]</a>) in Figure <a href="#fig:bias-var-tradeoff" data-reference-type="ref" data-reference="fig:bias-var-tradeoff">[fig:bias-var-tradeoff]</a>, all expectations and variances are computed as their respective unbiased statistical estimates over 4 seeds. $`\overline{C}`$ is estimated as the norm of the average Jacobian among 4 seeds on the training set:
``` math
\begin{aligned}
    \overline{C} = \sup_{\mathbf x\in \mathcal D} \|\nabla_\mathbf x\overline{f_{{\pmb \theta}}(\mathbf x)}\| & = \sup_{\mathbf x\in \mathcal D} \|\nabla_\mathbf x\mathbb{E}_\zeta[f_{{\pmb \theta}}(\mathbf x, \zeta)]\| = \sup_{\mathbf x\in \mathcal D}  \|\mathbb{E}_\zeta[\nabla_\mathbf xf_{{\pmb \theta}}(\mathbf x, \zeta)]\| \nonumber \\
    & \ge \sup_{\mathbf x\in \mathcal{D}}  \|\mathbb{E}_\zeta[\nabla_\mathbf xf_{{\pmb \theta}}(\mathbf x, \zeta)]\| \approx \sup_{\mathbf x\in \mathcal{D}}  \|\frac{1}{4}\sum_{i=1}^4\left(\nabla_\mathbf xf_{{\pmb \theta}}(\mathbf x, \zeta_i)\right)\|
\end{aligned}
```

##### Effect of the loss function on the Lipschitz constant (Section <a href="#effect-of-loss" data-reference-type="ref" data-reference="effect-of-loss">3.17</a>)

For this study we used the same models that we have trained for the Double Descent on MNIST1D using FCN ReLU networks trained with Cross-Entropy setting (see <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>) and with MSE setting (see <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a>). We plotted the evolution for the first 83,000 epochs for all models.

##### Effect of the optimisation algorithm on the Lipschitz constant (Section <a href="#effect-of-optimiser" data-reference-type="ref" data-reference="effect-of-optimiser">3.18</a>)

For this study, we used the same models that we have trained for the Double Descent on MNIST1D using FCN ReLU networks trained with Cross-Entropy setting (see <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>), as well as a set of 4 additionally trained FCN ReLU models with various seeds that were optimised with standard `pytorch` Adam optimiser ($`\beta_1=0.9, \beta_2=0.999`$). Each of these models was trained on MNIST1D with Cross-Entropy loss, 0.005 learning rate and Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>). Parameter vector was computed as a concatenation of flattened layer weights at each layer.

In Figure <a href="#fig:effect-of-optimiser-same-epoch" data-reference-type="ref" data-reference="fig:effect-of-optimiser-same-epoch">36</a> we showcase models up to epoch 10,000, while in figure <a href="#fig:effect-of-optimiser-last-epoch" data-reference-type="ref" data-reference="fig:effect-of-optimiser-last-epoch">37</a> models are stopped at 83,000 and 1,100 epochs for the SGD and Adam case respectively. In the former plot, both models achieved a gradient norm of at most 0.01 up to 2 significant figures at the end of their training.

*Comment on Figure <a href="#fig:effect-of-optimiser-same-epoch" data-reference-type="ref" data-reference="fig:effect-of-optimiser-same-epoch">36</a>.* We display training up to 10,000 epochs even though Adam reached low gradient norm much earlier due to slower rate of SGD — by 1,100 epochs SGD is still in the early training phase.

##### Effect of depth of the network on its Lipschitz constant (Section <a href="#effect-of-depth" data-reference-type="ref" data-reference="effect-of-depth">3.19</a>)

For this experiment we trained 5 types of FCN ReLU models with increasing depth. In particualr, we considered FCN ReLU 64; FCN ReLU 64,64; FCN ReLU 64,64,64; FCN ReLU 64,64,64,64 and FCN ReLU 64,64,64,64,64. Each model was trained on MNIST1D with batch size 512 using Cross-Entropy loss and the SGD optimiser, 0.005 learning rate and Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>). We trained our models for at least 10,000 epochs and stopped each model when either 0.01 gradient norm is reached or when 300,000 epochs have passed. We trained 4 seeds for each run.

For this scenario, we computed slopes for Lipschitz bounds starting from depth 2. The slopes for the upper, lower, and average Lipschitz bounds for *trained networks* are: 3.33, 2.03, 1.19 respectively and are computed by examining the slope coefficient of a linear regression model fitted to the corresponding values in the log-log scale. The $`R^2`$ values of the fit for the upper, lower, and average Lipschitz bounds are: 0.9749, 0.9483 and 0.8798 respectively. For networks *at initialisation*, slopes from depth 2 for the upper, lower, and average Lipschitz bounds are 0.30, -2.52 and -2.84 respectively.

##### Effect of the number of training samples on the Lipschitz constant (Section <a href="#effect-of-num-train-samples" data-reference-type="ref" data-reference="effect-of-num-train-samples">3.20</a>)

For this experiment we trained 4 types of FCN ReLU 256 models on different sizes of MNIST1D: 4000, 1000, 500 and 100 training samples. Sampling was performed by taking a random subsample from the main dataset. Each model was trained with batch size 512 using Cross-Entropy loss and the SGD optimiser, 0.005 learning rate and Warmup20000Step25 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>). We trained our models for at least 10,000 epochs and stopped each model when either 0.01 gradient norm is reached or when 300,000 epochs have passed. We trained 4 seeds for each run.

##### Effect of shuffling labels (Section <a href="#sec:label-noise" data-reference-type="ref" data-reference="sec:label-noise">5</a>)

For this experiment, we trained a sweep of CNN models with widths \[$`5`$, $`7`$, $`10`$, $`11`$, $`12`$, $`13`$, $`14`$, $`15`$, $`16`$, $`17`$, $`18`$, $`19`$, $`20`$, $`30`$, $`40`$\] on CIFAR-10 with various amounts of shuffled labels ($`\alpha`$ parameter): 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% and 100%. Each shuffle is incremental, meaning that all shuffles from a dataset with a smaller $`\alpha`$ are contained in the dataset with a larger $`\alpha`$. For each dataset, subsets and shuffles are fixed among seeds. Each model was trained with batch size 128 using Cross-Entropy loss and the SGD optimiser, 0.01 learning rate and Cont100 LR scheduler (see <a href="#lr-schedulers" data-reference-type="ref" data-reference="lr-schedulers">2.5</a>), where the minimum learning rate was limited to $`0.01\cdot0.001=\,`$<!-- -->1e-5. We trained the models for at least 1 000 epochs and stopped each model when either 0.01 gradient norm is reached or when 20 000 epochs have passed.

*Comment on the minimum number of epochs.* In comparison to the similar Double Descent setup (see <a href="#setup-dd-cifar10" data-reference-type="ref" data-reference="setup-dd-cifar10">2.6.4</a>) we use a larger number of minimum epochs in this experiment, since smaller models require more updates to escape the initialisation well, especially in the presence of label noise (see <a href="#training-strategy-pitfalls" data-reference-type="ref" data-reference="training-strategy-pitfalls">[training-strategy-pitfalls]</a>).

##### Effect of dropout and weight decay (Sections <a href="#effect-of-dropout" data-reference-type="ref" data-reference="effect-of-dropout">3.21</a> and <a href="#effect-of-weight-decay" data-reference-type="ref" data-reference="effect-of-weight-decay">3.23</a>)

For the following set of experiments, we used the approach of , who carefully implements the approach by . We refer to the original paper and the code repository cited before for implementation details. The only difference in our experimental setup is the inclusion of dropout layers (for the weight decay scenario, we do not use dropout). A 2D version of the Dropout layer is inserted before each ResNet block (i.e. a group of layers with a skip connection), leaving the first Convolution layer and BatchNorm intact. We also add a 1D Dropout layer before the last linear map. All dropout layers use the same probability $`p`$. We use the following values of $`p`$ for different runs: $`[0.0, 0.1, 0.2, 0.25, 0.3, 0.4]`$. For all Dropout experiments, we use the default value of 1e-4 for weight decay and train all models for 500 epochs.

For the weight decay experiments, we use the original setup and only change the weight decay parameter. We used the following set of values: \[0.0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 1e-3, 2.5e-3\]. All models were trained for 500 epochs.

### Additional experiments

#### Linearisation of Neural Networks and the effect of training

Using the first order of the Taylor expansion on our Neural Network $`f(\mathbf x;
{\pmb \theta})`$, we get the following linearisation: $`f(\mathbf x; {\pmb \theta})\approx f(\mathbf x; {\pmb \theta}_0) + \langle{\pmb \theta}-{\pmb \theta}_0, \nabla_{\pmb \theta}f(\mathbf x; {\pmb \theta}_0)\rangle`$. To see the effects of training on the lower Lipschitz constant bound, we can take the derivative w.r.t. input of the expression above and compute the 2-norm:
``` math
\begin{aligned}
    \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf(\mathbf x; {\pmb \theta})\| & \approx \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf(\mathbf x; {\pmb \theta}_0) + \langle{\pmb \theta}-{\pmb \theta}_0, \frac{\partial^2}{\partial \mathbf x\partial {\pmb \theta}}f(\mathbf x; {\pmb \theta}_0)\rangle\| \\
    & \le \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf(\mathbf x; {\pmb \theta}_0)\| + \|{\pmb \theta}-{\pmb \theta}_0\| \cdot \sup_{\mathbf x\in\mathcal{D}}\|\frac{\partial^2}{\partial \mathbf x\partial {\pmb \theta}}f(\mathbf x; {\pmb \theta}_0)\|\,.
\end{aligned}
```

We compute the bound above for an FCN ReLU 256 network trained on MNIST1D with CE loss (see Appendix <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>). It is worth noting that even in this experiment with a small network ($`12{,}800`$ params) and a small dataset ($`4{,}000`$ train points wih $`10`$ classes, input dimension is $`40`$), computing the second derivative becomes very expensive, as we have to evaluate a $`10\times 12{,}800 \times 40`$ tensor for all $`4{,}000`$ train points (that is more than half a billion gradient evaluations). We therefore restricted our computation to a subset of $`470`$ training points that we managed to compute given our time constraints. We compute the norm of this tensor in its matrix representation, where we flatten the output-parameter dimension (for this example, we have a $`128{,}000 \times 40`$ matrix).

The results are as follows: the lower bound at the last epoch (LHS) is $`24.228`$, while the lower bound at initialisation is almost $`65`$ times smaller: $`0.369`$. The second derivative term turns out to be only $`4.685`$, which shows that the final Lipschitz constant changes significanly during training (due to large parameter vector change).

*Remark*. The mean $`\pm`$ std. of $`\|\frac{\partial^2}{\partial \mathbf x\partial {\pmb \theta}}f(\mathbf x; {\pmb \theta}_0)\|`$ is $`3.127 \pm 0.619`$, showing that the estimate is very similar for most training points. This behaviour is expected, as we the network has not yet trained and is smooth around the whole domain due to the random initialisation.

#### Using adversarially perturbed samples for the lower Lipschitz bound computation

One way to improve the lower Lispchitz bound estimate (see Equation <a href="#eq:lower_bound" data-reference-type="ref" data-reference="eq:lower_bound">[eq:lower_bound]</a>) is to consider adversarial perturbations of the input instead of the inputs themselves. For this experiment, we perturb input samples using a slight modification of a PGD attack  with $`\epsilon=0.5`$, where the learning rate of 10.0 is decayed by a factor of 0.95 for each step in the algorithm. We use 1,000 iterations of PGD for each sample. We also compute the following Lipschitz estimates:
``` math
\begin{aligned}
    C_{\text{lower},\epsilon=0} & := \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf_{{\pmb \theta}}(\mathbf x)\|_2 \,, \\
    C_{\text{lower},\epsilon=0.5} & := \sup_{\mathbf x\in\mathcal{D}}\|\nabla_\mathbf xf_{{\pmb \theta}}(\mathbf x+\bm{\varepsilon})\|_2 \,, \\
    C_{\text{adv.straightforward},\epsilon=0.5} & := \sup_{\mathbf x\in\mathcal{D}} \frac{\|f_{{\pmb \theta}}(\mathbf x+\bm{\varepsilon})-f_{{\pmb \theta}}(\mathbf x)\|_2}{\|\bm{\varepsilon}\|_2}\,,
\end{aligned}
```
where $`\bm{\varepsilon}`$ is a noise vector, generated by the PGD attack ($`\|\bm{\varepsilon}\|_2 \le \epsilon`$).

Table <a href="#tab:adv-lower-bound" data-reference-type="ref" data-reference="tab:adv-lower-bound">6</a> shows the computed bounds for CNN models from the Double Descent experiment on CIFAR-10 (see Appendix <a href="#setup-dd-cifar10" data-reference-type="ref" data-reference="setup-dd-cifar10">2.6.4</a>) on one fixed seed. The results clearly show that adversarial perturbations do not always result in tighter lower bounds, and if they do, the increase in the estimate is not substantial.

<div id="tab:adv-lower-bound">

|  | CNN 5 | CNN 7 | CNN 10 | CNN 20 | CNN 30 | CNN 40 |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| $`C_{\text{lower},\epsilon=0}`$ | **148.29** | 633.68 | **756.37** | 172.15 | 123.61 | **106.35** |  |
| $`C_{\text{lower},\epsilon=0.5}`$ | 139.54 | **658.03** | 739.00 | **173.07** | **129.13** | 102.14 |  |
| $`C_{\text{adv.straightforward},\epsilon=0.5}`$ | 118.21 | 507.00 | 601.16 | 140.54 | 102.74 | 86.84 |  |

Lipschitz bounds for various CNN models, trained on CIFAR-10. Adversarial examples were generated using a PGD attack with $`\epsilon=0.5`$.

</div>

<span id="tab:adv-lower-bound" label="tab:adv-lower-bound"></span>

A similar behaviour can be observed for other values of $`\epsilon`$. Figure <a href="#fig:adversarial-lip-dd" data-reference-type="ref" data-reference="fig:adversarial-lip-dd">14</a> shows how close the $`C_\text{lower}`$ bounds lie for different attack strengths, while the loss on perturbed test values increases significantly.

<figure id="fig:adversarial-lip-dd">
<figure>
<img src="./figures/double_descent_lip_clean_adv_CIFAR10.png"" />
<figcaption>Lip. bounds on perturbed train samples</figcaption>
</figure>
<figure>
<img src="./figures/double_descent_test_loss_clean_adv_CIFAR10.png"" />
<figcaption>Loss on perturbed test samples</figcaption>
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with test loss with increasing hidden layer width, in the case of CNN networks on perturbed CIFAR-10 samples.</figcaption>
</figure>

#### Evaluating Lipschitz bounds for out-of-distribution (OOD) samples

To study the effects of out-of-distribution (OOD) samples on the Lipschitz constant, we took the models from the MNSIT1D, CE Double Descent study (Appendix <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>) and computed the Lipschitz constant bounds on OOD train samples, as well as test loss on OOD test samples. To generate an OOD sample, we added standard Gaussian noise with scale $`r`$ to the sample. The results are shown in Figure <a href="#fig:ood-lip-dd" data-reference-type="ref" data-reference="fig:ood-lip-dd">15</a>, which confirm that lower Lipschitz bounds do not change significanly when evaluated at OOD points.

<figure id="fig:ood-lip-dd">
<figure>
<img src="./figures/double_descent_lip_ood%2BMNIST1D%2BFF_ReLU%2BCrossEntropy.png"" />
<figcaption>Lipschitz bounds on OOD samples</figcaption>
</figure>
<figure>
<img src="./figures/double_descent_train_test_ood%2BMNIST1D%2BFF_ReLU%2BCrossEntropy.png"" />
<figcaption>Train and test loss on OOD samples</figcaption>
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with test loss with increasing hidden layer width, in the case of FCN ReLU networks on OOD MNIST1D samples.</figcaption>
</figure>

#### Evaluating the tightness of the theoretical bound

To compare the bounds derived in Appendix <a href="#power-law-trend-in-lipschitz-evolution" data-reference-type="ref" data-reference="power-law-trend-in-lipschitz-evolution">4.1</a>, we trained one FCN ReLU model with 6 hidden layers (widths 100, 100, 80, 80, 60 and 60) on MNIST1D with SGD using batch size 1,000 (4 updates per epoch). We also used a learning rate of 0.04 and a Cont50 LR scheduler (similar to Cont100, but the decay applies every 50 epochs, or every 200 updates in this case).

Figure <a href="#fig:theor-lip-evol-bnd-sgd" data-reference-type="ref" data-reference="fig:theor-lip-evol-bnd-sgd">16</a> shows the results of the evaluation. We denote Bound 2 and Bound 1 as follows:
``` math
\begin{aligned}
    \text{Bound 1}(\tau) & = \frac{2}{r} C_{{\pmb \theta}}^\text{discrete}(\tau) \sum_{t=0}^{\tau-1}\|{\pmb \theta}^{t+1}-{\pmb \theta}^{t}\|_2 + C_\mathbf x({\pmb \theta}^0)\,, \\
    \text{Bound 2}(\tau) & = \frac{2}{r} C_{{\pmb \theta}}^\text{discrete}(\tau) B\eta \tau + C_\mathbf x({\pmb \theta}^0)\,,
\end{aligned}
```
where $`C_{{\pmb \theta}}^\text{discrete}(\tau)=\sup_{{\pmb \theta}\in\{{\pmb \theta}^0,\dots,{\pmb \theta}^\tau\}}\sup_{\mathbf x\in \mathcal{D}} \|\nabla_{{\pmb \theta}} f({\pmb \theta}, \mathbf x) \|`$ denotes the Lipschitz constant with respect to model parameters up to checkpoint $`\tau`$, estimated at the train samples.

<figure id="fig:theor-lip-evol-bnd-sgd">
<figure>
<img src="./figures/lip_theor_bounds_evolution_MNIST1D_FF_ReLU_100_100_80_80_60_60_CrossEntropy_SGD_0.04_Cont50_loglog.png"" />
<figcaption>Lipschitz bounds</figcaption>
</figure>
<figure>
<img src="./figures/lip_theor_bounds_evolution_from_1000_MNIST1D_FF_ReLU_100_100_80_80_60_60_CrossEntropy_SGD_0.04_Cont50_loglog.png"" />
<figcaption>Lipschitz after 1000 updates</figcaption>
</figure>
<figcaption>Evaluation of the theoretical bounds on the Lipschitz evolution and comparison with the actual upper and lower Lipschitz estimates. Computed for FCN ReLU 100,100,80,80,60,60 trained on MNIST1D with SGD with batch size 1,000 (4 updates per epoch).</figcaption>
</figure>

The results show that our both bounds follow a trend similar to the other Lipschitz estimates. Moreover, at the beginning of training, our bound comes out to be smaller than the upper bound estimate.

#### Lipschitz evolution of a Vision Transformer on a subset of ImageNet

In this experiment, I evaluate the evolution of lower Lipschitz bounds for the Vision Transformer on a subset of ImageNet. As previously discussed, Attention layers do not have a theoretical Lipschitz upper bound and hence I only present the lower and average Lipschitz bounds. For this experiment, a subset of 50 000 train points from ImageNet was taken and only one seed is considered due to computational and time limitations. I used a `SimpleViT` model from the `vit_pytorch` package, where the patch size is set to 16, the last dimension after linear transformation is 384, depth is set 12, number of heads to 6, and the MLP dimension to 1 536. The model was pretrained on full ImageNet. Figure <a href="#fig:lip-evol-vit-resnet-imagenet" data-reference-type="ref" data-reference="fig:lip-evol-vit-resnet-imagenet">17</a> shows the results, with a comparison to ResNet50 bound on the same 50 000 sample subset.

<figure id="fig:lip-evol-vit-resnet-imagenet">
<img src="./figures/lip_evolution_vit_resnet_imagenet_loglog_True.png"" />
<figcaption>Lower Lipschitz bounds evolution for ViT and ResNet50 on a <span class="math inline">50 000</span> samples ImageNet subset.</figcaption>
</figure>

#### Estimation of the potential error of evaluating lower Lipschitz bound by considering a subsample of ImageNet

Due to computational limitations, in the ResNet50 evolution experiment (see Section <a href="#larger-network-dataset-setting" data-reference-type="ref" data-reference="larger-network-dataset-setting">3.2</a>) we evaluate the lower Lipschitz bound on a subset of ImageNet. This decision would naturally produce a weaker estimate of the Lipschitz constant, compared to the full dataset evaluation, raising concerns on the representativeness of the lower bound in the first place. Therefore we conducted a small investigation into the magnitude of potential error due to ImageNet subsampling for the ResNet18 case.

In this analysis, we took the distribution of Jacobian norms for pretrained ResNet18, evaluated for the whole ImageNet training dataset (see <a href="#setup-convex-combinations-resnet18" data-reference-type="ref" data-reference="setup-convex-combinations-resnet18">2.6.10</a> for more details), and estimated the lower bound for various random subsets of computed norms. Results are depicted in Table <a href="#tab:imagenet-subsets-error" data-reference-type="ref" data-reference="tab:imagenet-subsets-error">7</a>. According to the obtained statistics, doubling the subset size decreases the relative error by around 5%, suggesting that moderate errors can be achieved for rather small subsets. These results motivated our decision of using 200,000 as our subsample size — a plausible trade-off between computational burden and estimation accuracy.

<div id="tab:imagenet-subsets-error">

| Subset size | Estimate | St.dev. | Percentage difference to true estimate |
|:-----------:|:--------:|:-------:|:--------------------------------------:|
|   50,000    |  220.50  |  23.72  |                 20.87%                 |
|   100,000   |  234.95  |  23.49  |                 15.69%                 |
|   200,000   |  249.28  |  21.24  |                 10.54%                 |
|   300,000   |  257.02  |  18.85  |                 7.76%                  |
|   400,000   |  262.15  |  16.86  |                 5.92%                  |

Estimates of the lower bound for trained ResNet18 on the different subsets of ImageNet. Estimates are averaged over 1000 random subsets.

</div>

<span id="tab:imagenet-subsets-error" label="tab:imagenet-subsets-error"></span>

#### The convex combination study for ResNet18

In this experiment, we take the top $`1,000`$ samples that have the highest Jacobian norm, consider their convex combinations, and use that as a basis to evaluate the lower Lipschitz bounds. In fact, in Figure <a href="#fig:norm-distr-imagenet-resnet18" data-reference-type="ref" data-reference="fig:norm-distr-imagenet-resnet18">18</a>, we show the entire distribution of norms for these “hard” convex combinations as well as the entire ImageNet training set.

<figure id="fig:norm-distr-imagenet-resnet18">
<img src="./figures/ResNet18s_norms_distr_ImageNet_and_CC.png"" />
<figcaption>Distribution of the norm of the per-sample Jacobian for pretrained <strong>ResNet18</strong>, computed on the entire ImageNet and <span class="math inline">1, 000, 000</span> hard convex combinations on ImageNet (See Appendix <a href="#setup-convex-combinations-resnet18" data-reference-type="ref" data-reference="setup-convex-combinations-resnet18">2.6.10</a> for more).</figcaption>
</figure>

We find that while the distribution indeed shifts towards larger per-sample Jacobian norms for the hard convex combinations, the shift is not even a multiplicative factor of $`2\times`$ more. This shift pales in comparison to the upper bound which is over tens of orders of magnitudes higher. Overall, this strengthens our claim that the lower bound is much more faithful to the effective Lipschitz value and can hence serve better to explore various phenomenon observed in over-parameterized neural networks. Lastly, similar distribution plots for other models and datasets can be found in Appendix <a href="#jacobian-norm-distributions-other-models-datasets" data-reference-type="ref" data-reference="jacobian-norm-distributions-other-models-datasets">3.15</a>.

#### Lipschitz evolution for other settings

In this section, we present evolution plots for settings that were mentioned in Section <a href="#sec:lip_evol" data-reference-type="ref" data-reference="sec:lip_evol">3.1</a>. We start with evolution plots for FCN networks with MSE loss and then continue with CNN evolution on CIFAR-10 and MNIST. Table <a href="#tab:slopes-evolution" data-reference-type="ref" data-reference="tab:slopes-evolution">8</a> shows the values of the corresponding slopes with linear regression $`R^2`$ metrics for each setup.

<div id="tab:slopes-evolution">

| Evolution plot | Slopes | $`R^2`$ metric |
|:--:|:--:|:--:|
| FCN, MNIST1D, CE loss (Fig. <a href="#fig:lip_bounds_evolution" data-reference-type="ref" data-reference="fig:lip_bounds_evolution">2</a>) | 0.59, **0.46**, 0.44 | 0.9955, **0.9986**, 0.9980 |
| FCN, MNIST1D, MSE loss (Fig. <a href="#fig:lip-bounds-evolution-mnist1d-mse" data-reference-type="ref" data-reference="fig:lip-bounds-evolution-mnist1d-mse">19</a>) | 0.80, **0.62**, 0.52 | 0.9982, **0.9914**, 0.9991 |
|  |  |  |
| CNN, CIFAR-10, CE loss (Fig. <a href="#fig:lip-bounds-evolution-cifar10" data-reference-type="ref" data-reference="fig:lip-bounds-evolution-cifar10">20</a>) | 0.22, **0.29**, 0.29 | 0.9461, **0.9327**, 0.9460 |
|  |  |  |
| ResNet50, subset of ImageNet, CE loss (Fig. <a href="#fig:resnet50-evolution" data-reference-type="ref" data-reference="fig:resnet50-evolution">[fig:resnet50-evolution]</a>) | 7.32, **0.49**, 0.48 | 0.8441, **0.9718**, 0.8774 |

Estimates of the slopes and linear regression $`R^2`$ metrics for the upper, **lower** and average norm Lipschitz bounds for various evolution plots.

</div>

<span id="tab:slopes-evolution" label="tab:slopes-evolution"></span>

From Table <a href="#tab:slopes-evolution" data-reference-type="ref" data-reference="tab:slopes-evolution">8</a>, one can clearly see how the slope for the upper bound is larger for some more complex models and datasets. Consequently, a simple upper bound is an excessively loose estimator of the Lipschitz constant — despite its theoretical full input domain coverage, this estimator has little practical value due to excessive overestimation. As discussed in detail in Section <a href="#sec:lip_evol" data-reference-type="ref" data-reference="sec:lip_evol">3.1</a>, we therefore suggest paying close attention to the lower bound estimator. It would intriguing to see how tighter upper bound estimates compare to our lower and upper bounds, but we leave this investigation for future work.

<figure id="fig:lip-bounds-evolution-mnist1d-mse">
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_MNIST1D_LeakyOutput_FF_ReLU_256_MSE_SGD_0.001_Warmup20000Step25.png"" />
<figcaption>Linear scale plot</figcaption>
</figure>
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_MNIST1D_LeakyOutput_FF_ReLU_256_MSE_SGD_0.001_Warmup20000Step25_loglog.png"" />
<figcaption>Log-log scale plot</figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for <strong>FCN ReLU 256</strong>. The model was trained on <strong>MNIST1D</strong> using <strong>MSE loss</strong> and SGD optimiser. Results are averaged over 4 runs. Stable phase is considered to start after epoch 75,000. We took this model form the Double Descent experiment (see Appendix <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a>).</figcaption>
</figure>

<figure id="fig:lip-bounds-evolution-cifar10">
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_CIFAR10_CNN_20_CrossEntropy_SGD_0.01_Cont100.png"" />
<figcaption>Linear scale plot</figcaption>
</figure>
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_CIFAR10_CNN_20_CrossEntropy_SGD_0.01_Cont100_loglog.png"" />
<figcaption>Log-log scale plot</figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for <strong>CNN 20</strong>. The model was trained on <strong>CIFAR-10</strong> using Cross-Entropy loss and SGD optimiser. Results are averaged over 4 runs. Stable phase is considered to start after epoch 240. We used the same setup as in the Double Descent experiment (see Appendix <a href="#setup-dd-cifar10" data-reference-type="ref" data-reference="setup-dd-cifar10">2.6.4</a>).</figcaption>
</figure>

<figure id="fig:lip-bounds-evolution-mnist">
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_MNIST_CNN_20_CrossEntropy_SGD_0.01_Cont100.png"" />
<figcaption>Linear scale plot</figcaption>
</figure>
<figure>
<img src="./figures/lip_bounds_evolution_powerlaw_MNIST_CNN_20_CrossEntropy_SGD_0.01_Cont100_loglog.png"" />
<figcaption>Log-log scale plot</figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for <strong>CNN 20</strong>. The model was trained on <strong>MNIST with 10% labels shuffled</strong> using Cross-Entropy loss and SGD optimiser. Results are averaged over 4 runs. Stable phase is considered to start after epoch 100. We took this model form the Double Descent experiment (see Appendix <a href="#setup-dd-mnist" data-reference-type="ref" data-reference="setup-dd-mnist">2.6.6</a>).</figcaption>
</figure>

#### Lipschitz Double Descent for other settings

This section showcases Lipschitz constant (left) and train / test loss (right) Double Descent plots for more model class and dataset settings.

*Remark on Figure <a href="#fig:lip_dd" data-reference-type="ref" data-reference="fig:lip_dd">22</a>:* Like the test loss, the upper Lipschitz for FCN ReLU networks seems to continue to increase after the second descent, which could potentially be tied to the Triple Descent phenomenon , where the test loss shows another peak for $`p\approx n^2`$. We leave this interesting observation for future research.

*Remark on Figure <a href="#fig:lip_dd_vit_cifar10" data-reference-type="ref" data-reference="fig:lip_dd_vit_cifar10">26</a>:* For the case of ViT models on CIFAR-10, we also plot the lower Lipschitz constant estimates for a set of $`1{,}000{,}000`$ CIFAR-10 convex combinations (denoted as $`S^{**}`$) to further show the fidelity of the lower Lipschitz bound, even in a Double Descent setting.

<figure id="fig:lip_dd">
<figure>
<img src="./figures/double_descent_lip%2BMNIST1D%2BFF_ReLU_X%2BCrossEntropy%2BSGD_0.005_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figure>
<img src="./figures/double_descent_train_test%2BMNIST1D%2BFF_ReLU_X%2BCrossEntropy%2BSGD_0.005_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing hidden layer width, in the case of <strong>FCN ReLU networks</strong> on <strong>MNIST1D</strong>. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>.</figcaption>
</figure>

<figure id="fig:lip_dd_mse">
<figure>
<img src="./figures/double_descent_lip%2BMNIST1D%2BLeakyOutput_FF_ReLU_X%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" style="height:14em" />
</figure>
<figure>
<img src="./figures/double_descent_train_test%2BMNIST1D%2BLeakyOutput_FF_ReLU_X%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" style="height:14em" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing hidden layer width, in the case of <strong>FCN ReLU networks</strong> on <strong>MNIST1D</strong> with <strong>MSE loss</strong>. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a>.</figcaption>
</figure>

<figure id="fig:lip_dd_cnn_cifar">
<figure>
<img src="./figures/double_descent_lip%2BCIFAR10%2BCNN_X%2BCrossEntropy%2BSGD_0.01_Cont100%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figure>
<img src="./figures/double_descent_train_test%2BCIFAR10%2BCNN_X%2BCrossEntropy%2BSGD_0.01_Cont100%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing hidden layer width, in the case of <strong>CNN networks</strong> on <strong>CIFAR-10</strong>. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-cifar10" data-reference-type="ref" data-reference="setup-dd-cifar10">2.6.4</a>.</figcaption>
</figure>

<figure id="fig:lip_dd_cnn_mnist">
<figure>
<img src="./figures/double_descent_lip%2BMNIST%2BCNN_X%2BCrossEntropy%2BSGD_0.01_Cont100%2Balpha_0.1%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figure>
<img src="./figures/double_descent_train_test%2BMNIST%2BCNN_X%2BCrossEntropy%2BSGD_0.01_Cont100%2Balpha_0.1%2Bseed_X%2Btarget_0.01.png"" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing model width, in the case of <strong>CNN networks</strong> on <strong>MNIST with 10% of labels shuffled</strong> with Cross-Entropy loss. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-mnist" data-reference-type="ref" data-reference="setup-dd-mnist">2.6.6</a>.</figcaption>
</figure>

<figure id="fig:lip_dd_vit_cifar10">
<figure>
<img src="./figures/double_descent_lip_cc_vit_CIFAR10.png"" />
</figure>
<figure>
<img src="./figures/double_descent_train_test_loss_vit_CIFAR10.png"" />
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test losses with increasing parameter count, in the case of <strong>Vision Transformer networks</strong> on <strong>CIFAR-10</strong> with Cross-Entropy loss. Results are averaged over 4 runs. More details about the networks and the training strategy are listed in Appendix <a href="#setup-dd-vit-cifar" data-reference-type="ref" data-reference="setup-dd-vit-cifar">2.6.7</a>.</figcaption>
</figure>

#### More experiments on the Lipschitz constant in the Double Descent setting

This section includes experiments, where we compute Lipschitz constant bounds for a set of models from another study . Results are produced from training a series of fully connected networks on CIFAR-10 and MNIST with MSE loss. This solidifies our findings on Lipschitz’s Double Descent trends.

<figure id="fig:extra_lip_bounds_and_optim">
<figure>
<img src="./figures/double_descent_other_models.png"" />
<figcaption>Models trained on CIFAR-10</figcaption>
</figure>
<figure>
<img src="./figures/mse_fcn_mnist.png"" />
<figcaption>Models trained on MNIST</figcaption>
</figure>
<figcaption>Plot of Lipschitz constant bounds for fully connected networks with 1 hidden layer, trained using SGD with MSE loss.</figcaption>
</figure>

#### Bias-Variance trade-off evaluation

<div class="wrapfigure">

r0.43 <img src="./figures/test_loss%2Bvar%2BMNIST1D%2BLeakyOutput_FF_ReLU%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Btarget_0.01.png"" />

</div>

In Section <a href="#bias-variance-tradeoff-argument" data-reference-type="ref" data-reference="bias-variance-tradeoff-argument">[bias-variance-tradeoff-argument]</a>, we showed the equation for the bias-variance trade-off for the expected test loss. We then proceeded with upper bounding the variance, ignoring the effect of bias. In Figure <a href="#fig:bias-var-test-loss" data-reference-type="ref" data-reference="fig:bias-var-test-loss">[fig:bias-var-test-loss]</a> we display the results of the empirical bias-variance decomposition of the test loss for the MNIST1D Double Descent scenario (see Appendix <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a> for more details). The plot reveals that the Double Descent shape is mostly governed by the variance term, whereas the bias is almost constant across widths.

#### Double Descent for Lipschitz constrained networks

In this experiment, inspired by the work of , we replicate the Double Descent scenario with FCN ReLU networks on MNIST1D, where we regularise the product of spectral norms of the weight matrices to be at most $`C_\text{upper}^\text{constr}`$. We do this by renormalising each individual weight matrix spectral norm to be at most $`\sqrt[\text{depth}]{C_\text{upper}^\text{constr}}`$. In order to preserve the expressivity of highly constrained networks, we multiply network output logits by a temperature $`\tau`$, as described by . For $`C_\text{upper}^\text{constr}=10`$, we use $`\tau=10`$, for $`C_\text{upper}^\text{constr}=1`$, we use $`\tau=40`$, and for no constraint $`\tau=1`$ (i.e. no temperature).

To optimise the models, we used SGD with batch size 512 and LR 0.001, with a LR scheduler Cont100. The results are shown in Figure <a href="#fig:dd-lip-constr" data-reference-type="ref" data-reference="fig:dd-lip-constr">28</a>. As expected, the test loss follows the Lipschitz constant bounds.

<figure id="fig:dd-lip-constr">
<figure>
<img src="./figures/double_descent_lip_constr_lip.png"" />
<figcaption>Lipschitz constant bounds</figcaption>
</figure>
<figure>
<img src="./figures/double_descent_lip_constr_train_test_loss_with_tau.png"" />
<figcaption>Train and test error</figcaption>
</figure>
<figcaption>Comparison of various Lipschitz constant bounds with train and test loss with increasing hidden layer width, in the case of Lipschitz-constrained FCN ReLU networks on MNIST1D with different <span class="math inline"><em>C</em><sub>upper</sub><sup>constr</sup></span> constraint.</figcaption>
</figure>

#### Optimising the upper bound for the Variance

As discussed in Section <a href="#bias-variance-tradeoff-argument" data-reference-type="ref" data-reference="bias-variance-tradeoff-argument">[bias-variance-tradeoff-argument]</a> (and more comprehensively in Appendix <a href="#bias-variance-tradeoff-argument-extended" data-reference-type="ref" data-reference="bias-variance-tradeoff-argument-extended">4.2</a>), the variance of the learned function is related to the Lipschitz constant by following equation:
``` math
\begin{aligned}
   \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\, (\overline{C} ^2+\overline{C_{\zeta}}^2 )\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \|\mathbf x\|^2 \, + \, 3\, \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf{0}, \zeta)) \tag{\ref{eq:var_upper:1}}
\end{aligned}
```

<div class="wrapfigure">

r0.43 <img src="./figures/bounds_verif_extra%2BMNIST1D%2BLeakyOutput_FF_ReLU%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Btarget_0.01.png"" /> <span id="fig:optimised-var-bounds" label="fig:optimised-var-bounds"></span>

</div>

According to Figure <a href="#fig:bias-var-tradeoff" data-reference-type="ref" data-reference="fig:bias-var-tradeoff">[fig:bias-var-tradeoff]</a>, variance bounds are rather loose compared to the variance itself. We contribute this difference to the large value of dataset radius $`\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \|\mathbf x\|^2`$ which can potentially be optimised by choosing a better $`\mathbf x'`$ in bound equation <a href="#eq:var_upper:0" data-reference-type="ref" data-reference="eq:var_upper:0">[eq:var_upper:0]</a>. For this case, the optimal minimum for $`\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \|\mathbf x-\mathbf x'\|^2`$ is $`\mathbf x'=\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}}[\mathbf x]=\overline{\mathbf x}`$, which we utilised to recompute the bound in Figure <a href="#fig:optimised-var-bounds" data-reference-type="ref" data-reference="fig:optimised-var-bounds">[fig:optimised-var-bounds]</a>.

In the case of MNIST1D, the bound is only almost negligibly better due to $`\overline{\mathbf x}`$ being close to a zero vector for this particular dataset. We leave further analysis of improving the upper variance bound tightness for future work.

#### ResNet50 upper bound on more checkpoints

<div class="wrapfigure">

r0.5 <img src="./figures/lip_evolution_resnet50_upper_loglog_True.png"" />

</div>

Since computing the Lipschitz upper bound is significantly less expensive than the lower bound for the case of ResNet50 evolution, we have additionally evaluated the upper bound on a finer set of checkpoints for one seed to present a more complete picture of the upper bound evolution. According to the results in Figure <a href="#fig:resnet50-upper-evolution-more-checkpoints" data-reference-type="ref" data-reference="fig:resnet50-upper-evolution-more-checkpoints">[fig:resnet50-upper-evolution-more-checkpoints]</a>, the upper bound slope after epoch 60 is 7.04 with $`R^2`$ 0.9624, supporting the representativeness of our previous estimation.

#### Jacobian norm distributions for other models and datasets.

In Section <a href="#larger-network-dataset-setting" data-reference-type="ref" data-reference="larger-network-dataset-setting">3.2</a> we have shown a distribution of the norm of the per-sample Jacobian for pretrained ResNet18. Here we provide additional plots for other types of models and datasets. In particular, we showcase the distribution plots for FCNs trained on MNIST1D with Cross-Entropy and MSE losses, CNNs trained on CIFAR-10 and MNIST with 10% label noise and, finally, ResNets on CIFAR-10 (we used pretrained models from ). All norms are evaluated on the datasets used for training. Each dotted line represents the maximum norm, or, in other words, the lower Lipschitz bound estimate.

In comparison to just the lower Lipschitz bound estimate, the Jacobian norm distribution plot provides more information on the continuity of the function in the training domain. If the distribution has a long right tail, like in the case of ResNets in Figure <a href="#fig:dist-norm-cifar10-resnet" data-reference-type="ref" data-reference="fig:dist-norm-cifar10-resnet">31</a>, the corresponding function has to be rather smooth in the vicinity of most samples, while being more vulnerable to drastic changes for a smaller subset of inputs. It would be fascinating to see if the skewness of this distribution bears a connection to the difficulty of generating adversarial examples, but we leave this direction for future research.

<figure id="fig:dist-norm-mnist1d-mse">
<div class="minipage">
<img src="./figures/norms%2BMNIST1D%2BFF_ReLU_64%2BFF_ReLU_256.png"" />
</div>
<div class="minipage">
<img src="./figures/norms%2BMNIST1D%2BLeakyOutput_FF_ReLU_64%2BLeakyOutput_FF_ReLU_256.png"" />
</div>
<figcaption>Distribution of the norm of the per-sample Jacobian for FCN ReLU 64 and FCN ReLU 256, trained on MNIST1D with <strong>MSE loss</strong>. See Appendix <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a> for training details.</figcaption>
</figure>

<figure id="fig:dist-norm-mnist">
<div class="minipage">
<img src="./figures/norms%2BCIFAR10%2BCNN_7%2BCNN_20.png"" />
</div>
<div class="minipage">
<img src="./figures/norms%2BMNIST%2BCNN_7%2BCNN_20.png"" />
</div>
<figcaption>Distribution of the norm of the per-sample Jacobian for CNN 7 and CNN 20, trained on <strong>MNIST with 10% label noise</strong>. See Appendix <a href="#setup-dd-mnist" data-reference-type="ref" data-reference="setup-dd-mnist">2.6.6</a> for training details.</figcaption>
</figure>

<figure id="fig:dist-norm-cifar10-resnet">
<img src="./figures/norms%2BCIFAR10%2BResNet_20%2BResNet_56.png"" />
<figcaption>Distribution of the norm of the per-sample Jacobian for ResNet 20 and ResNet 56, trained on CIFAR-10. We refer to <span class="citation" data-cites="Idelbayev18a"></span> for training details.</figcaption>
</figure>

#### Lower Lipschitz bound as a regulariser

Although it is arguably easier to use the upper Lipschitz bound as a regulariser due to its lower computational complexity, it could not be the best choice for larger models, where compensating for the exponential increase in the upper Lipschitz estimate by the $`\lambda`$ hyperparameter might be tricky. In Figure <a href="#fig:lip-reg-evol" data-reference-type="ref" data-reference="fig:lip-reg-evol">32</a>, we show the results of training an FCN ReLU network on MNIST1D, while adding $`\lambda \cdot C_\text{lower}`$ a regularisation term. The $`C_\text{lower}`$ estimate is recomputed every epoch using the complete training set.

Although the computational efficiency of this approach leaves much to be desired, we can still see a positive effect on the test loss, *despite almost no change in the upper Lipschitz bounds*.

<figure id="fig:lip-reg-evol">
<figure>
<img src="./figures/lip_bounds_evolution_lip_reg_LeakyOutput_FF_ReLU_256.png"" />
<figcaption>Lipschitz constant bounds</figcaption>
</figure>
<figure>
<img src="./figures/lip_bounds_evolution_train_test_loss_reg_LeakyOutput_FF_ReLU_256.png"" />
<figcaption>Train and test loss</figcaption>
</figure>
<figcaption>Plot of Lipschitz constant bounds and train/test loss by training epoch for FCN ReLU network with width 256 trained on MNIST1D. Networks were regularised using various values of the <span class="math inline"><em>λ</em></span> hyperparameter.</figcaption>
</figure>

#### Effect of the loss function: CE vs MSE

Surprisingly, training networks with MSE loss results in marginally lower Lipschitz bounds than in the case of Cross-Entropy, as shown in Figure <a href="#fig:effect-of-loss-ce-mse" data-reference-type="ref" data-reference="fig:effect-of-loss-ce-mse">33</a>. We contribute this behaviour to the constraints that MSE imposes on the function output in the case of classification. Since in the MSE scenario an ideal model should output a zero vector with only one entry of value $`1`$, model’s outputs are restricted to unit vectors for the domain of training samples. At the same time, Cross-Entropy loss does not impose this constraint as output logits are implicitly Softmax-ed. Figure <a href="#fig:effect-of-loss-ce-mse-cesoftmax" data-reference-type="ref" data-reference="fig:effect-of-loss-ce-mse-cesoftmax">34</a> shows that applying Softmax to the output of the Cross-Entropy network shrinks the Lipschitz constant dramatically.

<figure id="fig:effect-of-loss">
<figure id="fig:effect-of-loss-ce-mse">
<img src="./figures/lip_bounds_and_loss_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_MSE_SGD_loglog.png"" />
<figcaption>Lipschitz bounds comparison: Cross-Entropy vs MSE</figcaption>
</figure>
<figure id="fig:effect-of-loss-ce-mse-cesoftmax">
<img src="./figures/lip_bounds_and_loss_extra_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_MSE_SGD_loglog.png"" />
<figcaption>Lower Lipschitz bounds for CE (with and without Softmax) and MSE</figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for FCN ReLU network with 1 hidden layer with 256 neurons, trained using <strong>Cross-Entropy and MSE</strong>. Both models were trained on MNIST1D with SGD, using the same learning rate and LR scheduler. Results are averaged over 4 runs. More details are in Appendix <a href="#setup-effect-of-loss" data-reference-type="ref" data-reference="setup-effect-of-loss">2.6.12</a>.</figcaption>
</figure>

#### Effect of the optimisation algorithm: SGD vs Adam

When a network is trained using the Adam optimiser, Lipschitz constant bounds escalate dramatically compared to the results from SGD. This finding supports the fact that Adam finds solutions that generalise significantly worse, despite great training performance . Note that this trend remains even if we account for Adam’s faster convergence, i.e. compare networks at their respective last epochs of training, which are not the same due to various convergence rates (see Figure <a href="#fig:effect-of-optimiser" data-reference-type="ref" data-reference="fig:effect-of-optimiser">38</a>).

<figure id="fig:effect-of-optimiser">
<figure id="fig:effect-of-optimiser-same-epoch">
<img src="./figures/lip_bounds_and_optim_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_CrossEntropy_Adam_loglog.png"" />
<figcaption>Lipschitz bounds evolution for SGD and Adam <strong>at the same epoch</strong></figcaption>
</figure>
<figure id="fig:effect-of-optimiser-last-epoch">
<img src="./figures/diff_stop_epoch_lip_bounds_and_optim_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_CrossEntropy_Adam_loglog.png"" />
<figcaption>Lipschitz bounds evolution for SGD and Adam <strong>at the end of training</strong></figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for FCN ReLU network with 1 hidden layer with 256 neurons, trained using <strong>SGD and Adam</strong>. Both models were trained on MNIST1D with Cross-Entropy loss, using the same LR and LR scheduler. Results are averaged over 4 runs. We end training when the gradient norm reaches the critical value of 0.01. More details are in Appendix <a href="#setup-effect-of-optimiser" data-reference-type="ref" data-reference="setup-effect-of-optimiser">2.6.13</a>.</figcaption>
</figure>

We suggest that this behaviour can be explained by observing how far models travel from their initial parameters. Figure <a href="#fig:effect-of-optimiser-param-distance" data-reference-type="ref" data-reference="fig:effect-of-optimiser-param-distance">41</a> shows the evolution of parameter distances (i.e. $`param\_dist_\tau = \sum_{t=1}^\tau \|{\pmb \theta}^{t}-{\pmb \theta}^{t-1}\|_2`$, where $`t`$ iterates through saved model checkpoints), which has a similar trend to Lipschitz bounds. In fact, we show that Lipschitz constant can be indeed expressed in terms of parameter distance in our theoretical analysis in Section <a href="#power-law-trend-in-lipschitz-evolution" data-reference-type="ref" data-reference="power-law-trend-in-lipschitz-evolution">4.1</a>. We leave a thorough exploration of this facet for future work. As a bonus we also show that parameter distance also exhibits Double Descent, see Figure <a href="#fig:effect-of-optimiser-double-descent" data-reference-type="ref" data-reference="fig:effect-of-optimiser-double-descent">44</a>.

<figure id="fig:effect-of-optimiser-param-distance">
<figure id="fig:effect-of-optimiser-param-distance-same-epoch">
<img src="./figures/param_dist_optimiser_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_CrossEntropy_Adam_loglog.png"" />
<figcaption>Parameter displacement evolution for SGD and Adam <strong>at the same epoch</strong></figcaption>
</figure>
<figure id="fig:effect-of-optimiser-param-distance-last-epoch">
<img src="./figures/diff_stop_epoch_param_dist_optimiser_MNIST1D_FF_ReLU_256_CrossEntropy_SGD_CrossEntropy_Adam_loglog.png"" />
<figcaption>Parameter displacement evolution for SGD and Adam <strong>at the end of training</strong></figcaption>
</figure>
<figcaption>Parameter displacement evolution for FCN ReLU network with 1 hidden layer with 256 neurons, trained using <strong>SGD and Adam</strong>. Both models were trained on MNIST1D with Cross-Entropy loss, using the same LR and LR scheduler. Results are averaged over 4 runs. We end training when the gradient norm reaches the critical value of 0.01. More details are in Appendix <a href="#setup-effect-of-optimiser" data-reference-type="ref" data-reference="setup-effect-of-optimiser">2.6.13</a>.</figcaption>
</figure>

<figure id="fig:effect-of-optimiser-double-descent">
<figure id="fig:effect-of-optimiser-double-descent-ce">
<img src="./figures/double_descent_param_dist%2BMNIST1D%2BFF_ReLU_X%2BCrossEntropy%2BSGD_0.005_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
<figcaption>Double Descent with Cross-Entropy loss</figcaption>
</figure>
<figure id="fig:effect-of-optimiser-double-descent-mse">
<img src="./figures/double_descent_param_dist%2BMNIST1D%2BLeakyOutput_FF_ReLU_X%2BMSE%2BSGD_0.001_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" />
<figcaption>Double Descent with MSE loss</figcaption>
</figure>
<figcaption>Parameter distance at the last epoch for models from the Double Descent on MNIST1D setting with Cross-Entropy (Fig. <a href="#fig:lip_dd" data-reference-type="ref" data-reference="fig:lip_dd">22</a>) and MSE (Fig. <a href="#fig:lip_dd_mse" data-reference-type="ref" data-reference="fig:lip_dd_mse">23</a>) losses. More details are in Appendices <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a> and <a href="#setup-dd-mnist1d-mse" data-reference-type="ref" data-reference="setup-dd-mnist1d-mse">2.6.3</a>.</figcaption>
</figure>

#### Effect of depth

To study how depth affects the Lipschitz constant of the network we trained 5 fully-connected networks on MNIST1D with the same learning parameters. According to Figure <a href="#fig:effect-of-depth-trained" data-reference-type="ref" data-reference="fig:effect-of-depth-trained">45</a>, all Lipschitz bounds for trained models start to increase with each subsequent layer after 2 layers, following a trend close to power law — $`R^2`$ linear regression metrics are 0.9749, 0.9483 and 0.8798 for upper, lower, and average Lipschitz bounds respectively. The slopes for the corresponding Lipschitz bounds are 3.33, 2.03 and 1.19, indicating a superlinear trend.

An interesting fact is that the aforementioned trend for lower Lipschitz bounds does not hold for networks at initialisation, both for the case of increasing depth (Figure <a href="#fig:effect-of-depth-at-init" data-reference-type="ref" data-reference="fig:effect-of-depth-at-init">46</a>) and increasing width (Figure <a href="#fig:init-by-width" data-reference-type="ref" data-reference="fig:init-by-width">51</a>). Consequently, we see how the effect of feature learning gets manifested in the bounds of the Lipschitz constant and that looking solely at initialisation (as in the style of the lazy regime ) would be insufficient. To visualise this ‘trend flipping’ behaviour we also present evolution plots for the upper and lower Lipschitz constant bounds in Figure <a href="#fig:lip-evol-depth" data-reference-type="ref" data-reference="fig:lip-evol-depth">50</a>.

<figure id="fig:effect-of-depth">
<figure id="fig:effect-of-depth-trained">
<img src="./figures/summary_lip_depth_MNIST1D_FF_ReLU_CrossEntropy_SGD_0.005_Warmup20000Step25_loglog.png"" />
<figcaption>Trained networks</figcaption>
</figure>
<figure id="fig:effect-of-depth-at-init">
<img src="./figures/summary_lip_depth_at_init_MNIST1D_FF_ReLU_CrossEntropy_SGD_0.005_Warmup20000Step25_loglog.png"" />
<figcaption>At initialisation</figcaption>
</figure>
<figcaption>Lipschitz constant bounds for FCN ReLU network with various number of hidden layer with 64 neurons for parameters at initialisation and after training. Results are averaged over 4 runs. More details are in Appendix <a href="#setup-effect-of-depth" data-reference-type="ref" data-reference="setup-effect-of-depth">2.6.14</a>.</figcaption>
</figure>

<figure id="fig:lip-evol-depth">
<figure id="fig:lip-evol-depth-lower">
<img src="./figures/lip_depth_lower_bound_MNIST1D_FF_ReLU_CrossEntropy_SGD_0.005_Warmup20000Step25_loglog.png"" />
<figcaption>Lower Lipschitz bound</figcaption>
</figure>
<figure id="fig:lip-evol-depth-upper">
<img src="./figures/lip_depth_upper_bound_MNIST1D_FF_ReLU_CrossEntropy_SGD_0.005_Warmup20000Step25_loglog.png"" />
<figcaption>Upper Lipschitz bound</figcaption>
</figure>
<figcaption>Lipschitz constant bounds evolution for FCN ReLU networks with various number of hidden layer with 64 neurons for parameters. Results are averaged over 4 runs. More details are in Appendix <a href="#setup-effect-of-depth" data-reference-type="ref" data-reference="setup-effect-of-depth">2.6.14</a>.</figcaption>
</figure>

<figure id="fig:init-by-width">
<img src="./figures/double_descent_lip_at_init%2BMNIST1D%2BFF_ReLU_X%2BCrossEntropy%2BSGD_0.005_Warmup20000Step25%2Balpha_0.0%2Bseed_X%2Btarget_0.01.png"" style="width:50.0%" />
<figcaption>Lipschitz constant bounds for FCN ReLU network with increasing hidden layer width at initialisation. Results are averaged over 4 runs. Here we used models from the Double Descent experiment, see details in Appendix <a href="#setup-dd-mnist1d-ce" data-reference-type="ref" data-reference="setup-dd-mnist1d-ce">2.6.2</a>.</figcaption>
</figure>

#### Effect of the number of training samples

Increasing the number of samples in the dataset results in a corresponding sublinear increase in all Lipschitz bounds. Figure <a href="#fig:effect-of-num-samples" data-reference-type="ref" data-reference="fig:effect-of-num-samples">52</a> shows the Lipschitz bounds for FCN ReLU 256 networks trained on various random subsets of MNIST1D. Using linear regression we estimated the slope of upper, lower and average Lipschitz bounds to be 0.53, 0.37 and 0.36 respectively, and the $`R^2`$ metrics are 0.9680, 0.9932 and 0.9928, implying a strong sublinear trend in all Lipschitz bounds. These results suggest that as the number of samples increases, the complexity of the function rises to fit a larger set of points. It would be interesting to precisely tease out this behaviour in terms of relevant theoretical quantities, but we leave that for future work.

<figure id="fig:effect-of-num-samples">
<img src="./figures/summary_lip_dataset_len_FF_ReLU_256_CrossEntropy_SGD_0.005_Warmup20000Step25_loglog.png"" style="width:50.0%" />
<figcaption>Lipschitz constant bounds for FCN ReLU network with 1 hidden layer with 256 neurons trained for various subsets of MNIST1D. Results are averaged over 4 runs. More details are in Appendix <a href="#setup-effect-of-num-train-samples" data-reference-type="ref" data-reference="setup-effect-of-num-train-samples">2.6.15</a>.</figcaption>
</figure>

#### Effect of dropout

Intuitively, increasing the probability value in dropout further regularises the model, which should result in lower Lipschitz bounds. To that this hold even for the $`C_\text{lower}`$ metric, we conduct an experiment with a ResNet20 model, trained on CIFAR-10. We impute Dropout layers before each ResNet block (i.e. a group of Convolution and BatchNorm layers that has a skip connection) and before the last Linear layer. The details of the experiment are described in Appendix <a href="#setup-resnet-dropout" data-reference-type="ref" data-reference="setup-resnet-dropout">2.6.17</a>. Figure <a href="#fig:effect-of-dropout" data-reference-type="ref" data-reference="fig:effect-of-dropout">53</a> shows the results, which indicate that our lower bound metric indeed follows the expected decreasing trend with increasing dropout regularisation.

<figure id="fig:effect-of-dropout">
<img src="./figures/lip_bounds_dropout.png"" style="width:50.0%" />
<figcaption>Lipschitz constant bounds for ResNet20, trained on CIFAR-10 with various levels of dropout. More details are in Appendix <a href="#setup-resnet-dropout" data-reference-type="ref" data-reference="setup-resnet-dropout">2.6.17</a>.</figcaption>
</figure>

#### Effect of weight decay

Similar to the previous experiment, we also test the $`C_\text{lower}`$ metric against increasing regularisation via weight decay. Once again, we expect the Lipschitz constant to decrease with stronger regularisation. The results in Figure <a href="#fig:effect-of-wd" data-reference-type="ref" data-reference="fig:effect-of-wd">55</a> are in line with our expectations: the lower Lipschitz bound indeed decreases (with some small noise) with higher values of weight decay.

<figure id="fig:effect-of-wd">
<img src="./figures/lip_bounds_weight_decay.png"" style="width:50.0%" />
<figcaption>Lipschitz constant bounds for ResNet20, trained on CIFAR-10 with various levels of weight decay. More details are in Appendix <a href="#setup-resnet-dropout" data-reference-type="ref" data-reference="setup-resnet-dropout">2.6.17</a>.</figcaption>
</figure>

#### Lower Lipschitz bound samples with correct and incorrect predicted labels

In this experiment, we compute the lower Lipschitz bounds on misclassified and correctly classified labels separately for the case of ResNet20 on CIFAR-10 with 0 dropout and 1e-4 weight decay. As the results demonstrate, incorrectly classified samples indeed more frequently have higher values of the Jacobian norm, compared to the points with correct classes. However, since the lower bound is computed as the supremum, basing the calculation on either of the sets alone would yield a similar value. Moreover, there is simply no correlation between the loss and the Jacobian norm (the value is 0.03), which indicates that there is no clear benefit in only considering misclassified samples for lower Lipschitz bound estimation.

<figure id="fig:effect-of-wd">
<img src="./figures/lip_distr_resnet20_dropout_p_0.0_wd_1e-4_stats.png"" style="width:50.0%" />
<figcaption>Jacobian norm bounds for points with correct and incorrect labels, predicted by a ResNet20, trained on CIFAR-10 with zero dropout and 1e-4 weight decay. More details are in Appendix <a href="#setup-resnet-dropout" data-reference-type="ref" data-reference="setup-resnet-dropout">2.6.17</a>.</figcaption>
</figure>

### Theoretical proofs

#### Theoretical analysis of Lipschitz evolution

Let us denote $`f({\pmb \theta}, \mathbf x): \mathbb R^p\times\mathbb R^d \mapsto \mathbb R^K`$ as our network function, where $`{\pmb \theta}`$ is our parameter vector. We also denote $`{\mathcal{L}}({\pmb \theta}, \mathcal{D})`$ as our loss and $`\mathcal{D}`$ as our training set. We are interested in finding a bound for $`C`$ for the trained model at time step $`T`$:
``` math
\forall \mathbf x, \mathbf x' \in \mathcal D: \|f({\pmb \theta}^T, \mathbf x) - f({\pmb \theta}^T, \mathbf x')\| \le C \|\mathbf x-\mathbf x'\| \le C \underbrace{\sup_{\mathbf x,\mathbf x'\in \mathcal D}\|\mathbf x-\mathbf x'\|}_{=r} \\
```

###### Initial and Final points based analysis.

Let us introduce a simple upper bound on the LHS by adding and subtracting the network at initialisation:
``` math
\begin{aligned}
    \|f({\pmb \theta}^T,\mathbf x) - f({\pmb \theta}^T,\mathbf x')\| & = \|f({\pmb \theta}^T,\mathbf x) - f({\pmb \theta}^0,\mathbf x) + f({\pmb \theta}^0,\mathbf x) - f({\pmb \theta}^0,\mathbf x') + f({\pmb \theta}^0,\mathbf x') - f({\pmb \theta}^T,\mathbf x')\| \\ 
    & \le \|f({\pmb \theta}^T,\mathbf x) - f({\pmb \theta}^0,\mathbf x)\| + \|f({\pmb \theta}^0,\mathbf x) - f({\pmb \theta}^0,\mathbf x')\| + \|f({\pmb \theta}^0,\mathbf x') - f({\pmb \theta}^T,\mathbf x')\| \\
    & \le 2 C_{{\pmb \theta}} \| {\pmb \theta}^0 - {\pmb \theta}^T \| + C_\mathbf x({\pmb \theta}^0) \| \mathbf x-\mathbf x' \| \\
    & \le 2 C_{{\pmb \theta}} \| {\pmb \theta}^0 - {\pmb \theta}^T \| + C_\mathbf x({\pmb \theta}^0) r \,,
\end{aligned}
```
where $`C_\mathbf x({\pmb \theta}^0)`$ is the Lipschitz constant in the input space for the model at initialisation and $`C_{\pmb \theta}`$ is the Lipschitz constant for the network in the parameter space. The latter quantity is unfortunately hard to compute, since it requires to search through the space of both parameters and inputs to find the maximum norm. Moreover, it might not be as tight.

###### Intermediate-points based analysis.

We can tackle the above issue by applying the same trick iteratively to get a local Lipschitz constant in the parameter space:
``` math
\begin{aligned}
    \|f({\pmb \theta}^T,\mathbf x) - f({\pmb \theta}^T,\mathbf x')\| & = \|f({\pmb \theta}^T,\mathbf x) + \sum_{t=0}^{T-1}\left(f({\pmb \theta}^t,\mathbf x) - f({\pmb \theta}^t,\mathbf x) + f({\pmb \theta}^t,\mathbf x') - f({\pmb \theta}^t,\mathbf x')\right) - f({\pmb \theta}^T,\mathbf x')\| \\ 
    & \le \sum_{t=0}^{T-1}\|f({\pmb \theta}^{t+1},\mathbf x) - f({\pmb \theta}^t,\mathbf x)\| + \sum_{t=0}^{T-1}\|f({\pmb \theta}^t,\mathbf x') - f({\pmb \theta}^{t+1},\mathbf x')\| + \|f({\pmb \theta}^0,\mathbf x) - f({\pmb \theta}^0,\mathbf x')\| \\
    & \le 2 C_{{\pmb \theta}}^\text{discrete} \sum_{t=0}^{T-1}\| {\pmb \theta}^{t+1} - {\pmb \theta}^t \| + C_\mathbf x({\pmb \theta}^0) \| \mathbf x-\mathbf x' \| \\
    & \le 2 C_{{\pmb \theta}}^\text{discrete} \sum_{t=0}^{T-1}\| {\pmb \theta}^{t+1} - {\pmb \theta}^t \| + C_\mathbf x({\pmb \theta}^0)r \,,
\end{aligned}
```
where $`C_{{\pmb \theta}}^\text{discrete}:= \sup_{{\pmb \theta}\in\{{\pmb \theta}^0,\dots,{\pmb \theta}^T\}}\sup_{\mathbf x\in dom(f)} \|\nabla_{{\pmb \theta}} f({\pmb \theta}, \mathbf x) \|`$, i.e., the supremum of parameter-wise Lipschitz constants for a discrete set of checkpoints. In comparison to $`C_{{\pmb \theta}}`$, $`C_{{\pmb \theta}}^\text{discrete}`$ is only evaluated for ($`{\pmb \theta}^0,\dots,{\pmb \theta}^T`$), reducing the search space in the parameter dimension (thus it is marked as discrete).

We can further simplify the equation by considering the GD update rule: $`{\pmb \theta}^{t+1} = {\pmb \theta}^t - \eta_t \nabla_{{\pmb \theta}^t}{\mathcal{L}}({\pmb \theta}^t, \mathcal{D})`$ and introducing the bounded gradients assumption (i.e. $`\|\nabla_{\pmb \theta}{\mathcal{L}}({\pmb \theta}, \mathbf x)\| \le B`$). Note that this constraint can be easily fulfilled by using gradient clipping. The term $`\eta_t`$ denotes the learning rate at time step $`t`$. Let $`\eta`$ be the maximum learning rate throughout the epochs. Then we have the following:
``` math
\begin{aligned}
    \|f({\pmb \theta}^T,\mathbf x) - f({\pmb \theta}^T,\mathbf x')\| & \le 2 C_{{\pmb \theta}}^\text{discrete} \sum_{t=0}^{T-1}\| {\pmb \theta}^{t+1} - {\pmb \theta}^t \| + C_\mathbf x({\pmb \theta}^0)r \\
    & \le 2 C_{{\pmb \theta}}^\text{discrete} \sum_{t=0}^{T-1}\eta_t\| \nabla_{{\pmb \theta}^t}{\mathcal{L}}({\pmb \theta}^t, \mathcal{D}) \| + C_\mathbf x({\pmb \theta}^0)r  \le 2 C_{{\pmb \theta}}^\text{discrete} \sum_{t=0}^{T-1}\eta B + C_\mathbf x({\pmb \theta}^0)r \\
    &= \left( \frac{2}{r} C_{{\pmb \theta}}^\text{discrete} B\eta T + C_\mathbf x({\pmb \theta}^0) \right)r \\
\end{aligned}
```
Therefore the Lipschitz constant grows in proportion to the number of steps:
``` math
C \propto \left(\frac{2}{r}C_{\pmb \theta}^\text{discrete} B\eta \right) T
```

#### Bias-Variance tradeoff

Let us denote the neural network function as $`f_{{\pmb \theta}}(\mathbf x, \mathcal{D}, \zeta)`$ and the ground-truth function as $`\mathbf y^{\star}(\mathbf x)`$. Here, $`\mathcal{D}`$ denotes the training set and $`\zeta`$ indicates the noise in the function due to the choice of random initialisation and the noise introduced by a stochastic optimiser, like stochastic gradient descent (SGD). In other words, one can take $`\zeta`$ as denoting the random seed used in practice. Then let us assume we have the square loss, i.e., $`\ell(\mathbf x; f_{{\pmb \theta}}) =  \|\mathbf y^*(\mathbf x) - f_{{\pmb \theta}}(\mathbf x, \mathcal{D}, \zeta)\|^2`$. We can write the loss evaluated on a test set, $`\mathcal{D_{\mathrm{test}}}`$, i.e., the test loss, as follows:
``` math
{\mathcal{L}}({\pmb \theta}, \mathcal{D_{\mathrm{test}}}, \zeta) = \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - f_{{\pmb \theta}}(\mathbf x, \mathcal{D}, \zeta)\|^2\right]
```
In practice, we typically average the test loss over several random seeds, hence inherently involving an expectation over the noise $`\zeta`$. We derive a bias-variance tradeoff  that rests upon this as the noise source. Also, we consider the fixed-design variant of the bias-variance tradeoff and as a result, we will not average over the choice of the training set sampled from the distribution. In any case, for a suitably large training set size, this is expected not to introduce a lot of fluctuations and in particular, for the phenomenon at hand, i.e. Double Descent, the training set is generally considered to be fixed. Hereafter, for convenience, we will suppress the dependence of the network function on the training set.

Now we do the usual trick of adding and subtracting the expected neural network function over the noise source. Hence, we can rewrite the above as:

``` math
\begin{aligned}
    {\mathcal{L}}({\pmb \theta}, \mathcal{D_{\mathrm{test}}}, \zeta) &= \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] + \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] -  f_{{\pmb \theta}}(\mathbf x, \zeta)\|^2\right]\\
    &= \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)]\|^2\right] + \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}}\left[\|\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] -  f_{{\pmb \theta}}(\mathbf x, \zeta)\|^2\right] \\
    & + 2 \,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}}\left[\left(\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)]\right)^\top \, \left(\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] - f_{{\pmb \theta}}(\mathbf x, \zeta))\right)\right]
\end{aligned}
```

Next, we take the expectation of the above test loss with respect to the noise source $`\zeta`$ — mirroring the empirical practice of reporting results averaged over multiple seeds. It is easy to see that when taking the expectation, the cross-term vanishes and we are left with the following expression:

``` math
\begin{aligned}
    \mathbb{E}_{\zeta}\, {\mathcal{L}}({\pmb \theta}, \mathcal{D_{\mathrm{test}}}, \zeta) &= \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)]\|^2\right] + \mathbb{E}_{\zeta} \, \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}}\left[\|\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] -  f_{{\pmb \theta}}(\mathbf x, \zeta)\|^2\right] \\
   & = \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)]\|^2\right] + \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}}\mathbb{E}_{\zeta} \left[\|\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] -  f_{{\pmb \theta}}(\mathbf x, \zeta)\|^2\right]\\
   & = \mathbb{E}_{\mathbf x\sim \mathcal{D_{\mathrm{test}}}} \left[\|\mathbf y^*(\mathbf x) - \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)]\|^2\right] + \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta))
\end{aligned}
```
Overall, this results in the bias-variance trade-off under our setting.

###### Upper-bounding the Variance term.

Now, we want to a do a finer analysis of the variance term by involving the Lipschitz constant of the network function.

``` math
\begin{aligned}
    & \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta))  = \mathbb{E}_{\zeta} \left[\|\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] -  f_{{\pmb \theta}}(\mathbf x, \zeta)\|^2\right]\\
     & = \mathbb{E}_{\zeta} \left[\| \underbrace{\mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] - \mathbb{E}_{\zeta}[f_{{\pmb \theta}}(\mathbf x',\zeta)]}_{a}  + \underbrace{\mathbb{E}_{\zeta}[f_{{\pmb \theta}}(\mathbf x',\zeta)] - f_{{\pmb \theta}}(\mathbf x',\zeta)}_{b} + \underbrace{f_{{\pmb \theta}}(\mathbf x',\zeta) -  f_{{\pmb \theta}}(\mathbf x, \zeta)}_{c}\|^2\right]
\end{aligned}
```
where, we have considered some auxiliary point $`\mathbf x'`$, and added and subtracted some terms. For $`n`$ vectors, $`\mathbf x_1, \cdots, \mathbf x_n`$, we can utilize the simple inequality:
``` math
\|\mathbf x_1 + \cdots + \mathbf x_n\|^2 \leq n \sum\limits_{i=1}^n \|\mathbf x_i\|^2
```

which follows from $`n`$ applications of the Cauchy-Schwarz inequality. Hence, the variance above can be upper-bounded as:
``` math
\begin{aligned}
    \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\,  \| \mathbb{E}_{\zeta} [f_{{\pmb \theta}}(\mathbf x, \zeta)] - \mathbb{E}_{\zeta}[f_{{\pmb \theta}}(\mathbf x',\zeta)]\|^2 + 3 \,\mathbb{E}_{\zeta}\, \|\mathbb{E}_{\zeta}[f_{{\pmb \theta}}(\mathbf x',\zeta)] - f_{{\pmb \theta}}(\mathbf x',\zeta)\|^2 \\
    &+ 3\, \mathbb{E}_{\zeta}\, \|f_{{\pmb \theta}}(\mathbf x',\zeta) -  f_{{\pmb \theta}}(\mathbf x, \zeta) \|^2
\end{aligned}
```

We can think of $`\mathbb{E}_{\zeta} f_{{\pmb \theta}}(\mathbf x, \zeta)`$ as the ensembled function mapping, and denote it by saying $`\overline{f_{{\pmb \theta}}}(\mathbf x):=\mathbb{E}_{\zeta} f_{{\pmb \theta}}(\mathbf x, \zeta)`$, and let’s assume that it is $`\overline{C}`$-Lipschitz. On the other hand, let’s say that each individual function $`f_{{\pmb \theta}}(\mathbf x, \zeta)`$ has Lipschitz constant $`C_{\zeta}`$. Hence we can further reduce the upper bound to
``` math
\begin{aligned}
    \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\, \overline{C}^2 \, \|\mathbf x-\mathbf x'\|^2 + 3\, \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf x', \zeta)) + 3\, \mathbb{E}_{\zeta} \, C_\zeta^2 \|\mathbf x-\mathbf x'\|^2 \,.
\end{aligned}
```
Now, we bring back the outer expectation with respect to samples from the test set, i.e., $`\mathbf x\sim\mathcal{D_{\mathrm{test}}}`$:
``` math
\begin{aligned}
   \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \overline{C}^2 \, \|\mathbf x-\mathbf x'\|^2 + 3\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf x', \zeta)) + 3\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \mathbb{E}_{\zeta} \, C_\zeta^2 \|\mathbf x-\mathbf x'\|^2
\end{aligned}
```
Notice that while the Lipschitz constant of the neural network function do depend on the training data, the above expectation is with respect to samples from the test set. Hence, we can take the Lipschitz constants that appear above outside of the expectation. Besides, the middle term on the right-hand side has no dependency on the test sample $`\mathbf x\sim\mathcal{D_{\mathrm{test}}}`$ and so the expectation goes away. Overall, this yields,
``` math
\begin{aligned}
   \label{eq:var_upper:0}
   \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}\limits_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\, (\overline{C}^2 +\overline{C_{\zeta}}^2 )\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \|\mathbf x-\mathbf x'\|^2 + 3\, \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf x', \zeta)) \tag{Var bound 0}
\end{aligned}
```

where, for simplicity, we have denoted the Lipschitz constant $`C_{\zeta}`$ averaged over the random seeds $`\zeta`$, as $`\overline{C_{\zeta}}`$. We can simplify the above upper bounds by taking $`\mathbf x'=\mathbf{0}`$ as the vector of all zeros, resulting in:
``` math
\begin{aligned}
   \mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \mathrm{Var}_\zeta(f_{{\pmb \theta}}(\mathbf x, \zeta)) & \leq 3\, (\overline{C} ^2+\overline{C_{\zeta}}^2 )\,\mathbb{E}_{\mathbf x\sim\mathcal{D_{\mathrm{test}}}} \, \|\mathbf x\|^2 + 3\, \mathrm{Var}_{\zeta}(f_{{\pmb \theta}}(\mathbf{0}, \zeta)) \label{eq:var_upper:1}\tag{Var bound 1}
\end{aligned}
```

[^1]: Equal Contribution. Correspondence to [`gkhromov@ethz.ch`](mailto:gkhromov@ethz.ch), [`ssidak@ethz.ch`](mailto:ssidak@ethz.ch). Our code is publicly available on [GitHub](https://github.com/gakhromov/lipschitz-continuity-of-nns).

[^2]: i.e., linearisation in the function space, $`f(\mathbf x; {\pmb \theta})\approx f(\mathbf x; {\pmb \theta}_0) + \langle{\pmb \theta}-{\pmb \theta}_0, \nabla_{\pmb \theta}f(\mathbf x; {\pmb \theta}_0)\rangle`$

[^3]: Computing the lower Lipschitz on the entire ImageNet would require evaluating $`\sim 1.2`$ million Jacobian matrices of size $`1{,}000\times 150{,}528`$, for a single seed and a single checkpoint. As a reference, training a ResNet50 for $`90`$ epochs with a $`1024`$ batch size requires $`\sim 106{,}000`$ gradient evaluations. Hence, **lower Lipschitz estimation at a single checkpoint is almost as expensive as one entire training run**. This justifies the expediency of employing subsampling. Moreover, the variance of this procedure seems minimal (Appendix <a href="#estimation-of-error-subsample-imagenet" data-reference-type="ref" data-reference="estimation-of-error-subsample-imagenet">3.6</a>).

[^4]: This is an idealised depiction of DD, since often the initial descent requires considering very small networks and may not even be visible . The more prominent DD signature is the test loss peak at the interpolation threshold and the ensuing non-monotonicity.

[^5]: MNIST1D input image is a vector of size $`40`$.

[^6]: CIFAR-10 dataset input image has shape $`32\times32\times3`$.

[^7]: MNIST dataset input image has shape $`28\times28\times3`$.

[^8]: MNIST1D has 4000 training samples.<span id="fn:mnist1d" label="fn:mnist1d"></span>

[^9]: CIFAR-10 has 50,000 training samples.

[^10]: CIFAR-100 has 50 000 training samples.

[^11]: MNIST has 60,000 training samples.

[^12]: CIFAR-10 has 50,000 training samples.
