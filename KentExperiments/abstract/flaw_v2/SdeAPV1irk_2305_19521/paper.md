# Incremental Randomized Smoothing Certification

## Abstract

The CRISPR (clustered regularly interspaced short palindromic repeat)–Cas9 (CRISPR-associated nuclease 9) system is poised to transform developmental biology by providing a simple, efficient method to precisely manipulate the genome of virtually any developing organism. This RNA-guided nuclease (RGN)-based approach already has been effectively used to induce targeted mutations in multiple genes simultaneously, create conditional alleles, and generate endogenously tagged proteins. Illustrating the adaptability of RGNs, the genomes of >20 different plant and animal species as well as multiple cell lines and primary cells have been successfully modified. Here we review the current and potential uses of RGNs to investigate genome function during development

# Introduction

Ensuring the robustness of deep neural networks (DNNs) to input perturbations is gaining increased attention from both users and regulators in various application domains . Out of many techniques for obtaining robustness certificaties, statistical methods currently offer the greatest scalability. Randomized smoothing (RS) is a popular statistical certification method by constructing a smoothed model $`g`$ from a base network $`f`$ under noise . To certify the model $`g`$ on an input, RS certification checks if the estimated lower bound on the probability of the top class is greater than the upper bound on the probability of the runner-up class (with high confidence). RS certification computes the certified accuracy metric of the DNN on the set of test inputs as a proxy for the DNN robustness. However, despite its effectiveness, RS-based certification can be computationally expensive as it requires DNN inference on a large number of corruptions per input.

The high cost of certification complicates the DNN deployment process, which has become increasingly iterative: the networks are often modified post-training to improve their execution time and/or accuracy. Especially, deploying DNNs on real-world systems with bounded computing resources (e.g., edge devices or GPUs with limited memory), has led to various techniques for approximating DNNs. Common approximation techniques include quantization – reducing the numerical precision of weights , and pruning – removing weights that have minimal impact on accuracy .

Common to all of these approximations is that the network behavior (e.g., the classifications) remains the same on most inputs, its architecture does not change, and many weights are only slightly changed. When a user seeks to select a robust and accurate DNN from these possible approximations, RS needs to be performed to compute the robustness of all candidate networks.

For instance, in the context of approximation tuning, there are multiple choices for approximation where different quantization or pruning strategies are applied at different layers. Tools such as use approximations iteratively and test the network at each step. To ensure DNN robustness when using such tools, one would need to check certified accuracy, computed using RS on test data in each step. However, performing RS to compute certified accuracy from scratch can take hours as shown in our experiments even for a single network (with only 500 test images).

Therefore, a major encumbrance of almost all existing RS-based certification practices in the above setting, is that *the expensive certification needs to be re-run from scratch* for each approximate network. Overcoming this main limitation requires addressing the following fundamental problem:

#### This Work.

We present the first incremental RS-based certification framework called Incremental Randomized Smoothing (IRS) to answer this question. The primary objective of our work is to improve the sample complexity of the certification process of similar networks on a predefined test set. Improved sample complexity results in overall speedup in certification, and it reduces the energy requirement and memory footprint of the certification. Given a network $`f`$ and its smoothed version $`g`$, and a modified network $`f^p`$ with its smoothed version $`g^p`$, IRS incrementally certifies the robustness of $`g^p`$ by reusing the information from the execution of RS certification on $`g`$.

IRS optimizes the process of certifying the robustness of smoothed classifier $`g^p`$ on an input $`x`$, by estimating *the disparity* $`\zeta_{x}`$ – the upper bound on the probability that outputs of $`f`$ and $`f^p`$ are distinct. Our new algorithm is based on three key insights about disparity:

1.  Common approximations yield small $`\zeta_{x}`$ values – for instance, it is below 0.01 for int8 quantization for multiple large networks in our experiments.

2.  Estimating $`\zeta_{x}`$ through binomial confidence interval requires fewer samples as it is close to 0 – it is, therefore, less expensive to certify with this probability than directly working with lower and upper probability bounds in the original RS algorithm.

3.  We can leverage $`\zeta_{x}`$ alongside the bounds in the certified radius of $`g`$ around $`x`$ to compute the certified radius for $`g^p`$ – thus soundly reusing the samples from certifying $`g`$.

We extensively evaluate the performance of IRS when applying several common DNN approximations such as pruning and quantization on state-of-the-art DNNs on CIFAR10 (ResNet-20, ResNet-110) and ImageNet (ResNet-50) datasets.

The main contributions of this paper are:

- We propose a novel concept of incremental RS certification of the robustness of the updated smoothed classifier by reusing the certification guarantees for the original smoothed classifier.

- We design the first algorithm IRS for incremental RS that efficiently computes the certified radius of the updated smoothed classifier.

- We present an extensive evaluation of the performance of IRS speedups of up to 4.1x over the standard non-incremental RS baseline on state-of-the-art classification models.

 IRS code is available at <https://github.com/uiuc-arc/Incremental-DNN-Verification>.

# Background

#### Randomized Smoothing.

Let $`f : \mathbb{R}^m \to \mathcal{Y}`$ be an ordinary classifier. A smoothed classifier $`g : \mathbb{R}^m \to \mathcal{Y}`$ can be obtained from calculating the most likely result of $`f(x+\epsilon)`$ where $`\epsilon \sim \mathcal{N}(0, \sigma^2 I)`$.
``` math
%g(x) := \argmax_{c \in \labels} \mathbb{P}_{\epsilon \sim \normal} (f(x+\epsilon) = c)
g(x) := \mathop{\mathrm{arg\,max}}_{c \in \mathcal{Y}} \mathbb{P}_{\epsilon} (f(x+\epsilon) = c)
```

The smoothed network $`g`$ satisfies following guarantee for a single network $`f`$:

<div class="theorem">

<span id="thm:rs" label="thm:rs"></span> \[From \] Suppose $`c_A \in \mathcal{Y}`$, $`\underline{p_A}, \overline{p_B}\in [0, 1]`$. if
``` math
\mathbb{P}_\epsilon (f(x+\epsilon) = c_A) \geq \underline{p_A}\geq \overline{p_B}\geq \max_{c \neq c_A} \mathbb{P}_\epsilon (f(x+\epsilon) = c),
```

then $`g(x+\delta) = c_A`$ for all $`\delta`$ satisying $`\|\delta\|_2 \leq \frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B}))`$, where $`\Phi^{-1}`$ denotes the inverse of the standard Gaussian CDF.

</div>

<figure id="alg:rs">
<figure id="alg:rs">
<p><strong>Inputs:</strong> <span class="math inline"><em>f</em></span>: DNN, <span class="math inline"><em>σ</em></span>: standard deviation, <span class="math inline"><em>x</em></span>: input to the DNN, <span class="math inline"><em>n</em><sub>0</sub></span>: number of samples to predict the top class, <span class="math inline"><em>n</em></span>: number of samples for computing <span class="math inline">$\underline{p_A}$</span>, <span class="math inline"><em>α</em></span>: confidence parameter</p>
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
<span class="math inline"><em>c</em><em>o</em><em>u</em><em>n</em><em>t</em><em>s</em><sub>0</sub> ← SampleUnderNoise(<em>f</em>, <em>x</em>, <em>n</em><sub>0</sub>, <em>σ</em>)</span><br />
<span class="math inline"><em>ĉ</em><sub><em>A</em></sub> ← top index in <em>c</em><em>o</em><em>u</em><em>n</em><em>t</em><em>s</em><sub>0</sub></span><br />
<span class="math inline"><em>c</em><em>o</em><em>u</em><em>n</em><em>t</em><em>s</em> ← SampleUnderNoise(<em>f</em>, <em>x</em>, <em>n</em>, <em>σ</em>)</span><br />
<span class="math inline">$\underline{p_A}\gets \text{LowerConfidenceBound}(counts[\hat{c}_A], n, 1 - \alpha)$</span><br />
<strong>If</strong> <span><span class="math inline">$\underline{p_A}&gt; \frac{1}{2}$</span></span><br />
<br />
Return prediction <span class="math inline"><em>ĉ</em><sub><em>A</em></sub></span> and radius <span class="math inline">$\sigma \cdot\Phi^{-1}(\underline{p_A})$</span><br />
Else<br />
<br />
Return ABSTAIN<br />
EndIf</p>
</div>
<figcaption>RS certification <span class="citation" data-cites="DBLP:conf/icml/CohenRK19"></span></figcaption>
</figure>
<figcaption>RS certification <span class="citation" data-cites="DBLP:conf/icml/CohenRK19"></span></figcaption>
</figure>

Computing the exact probabilities $`P_\epsilon(f(x + \epsilon) = c)`$ is generally intractable. Thus, for practical applications, CERTIFY (Algorithm <a href="#alg:rs" data-reference-type="ref" data-reference="alg:rs">2</a>) utilizes sampling: First, it takes $`n_0`$ samples to determine the majority class, then $`n`$ samples to compute a lower bound $`\underline{p_A}`$ to the success probability with confidence $`1-\alpha`$ via the Clopper-Pearson lemma . If $`\underline{p_A}> 0.5`$, we set $`\overline{p_B}= 1 - \underline{p_A}`$ and obtain radius $`R = \sigma\cdot\Phi^{-1}(\underline{p_A})`$ via Theorem <a href="#thm:rs" data-reference-type="ref" data-reference="thm:rs">[thm:rs]</a>, else we return ABSTAIN.

#### DNN approximation.

DNN weights need to be quantized to the appropriate datatype for deploying them on various edge devices. DNN approximations are used to compress the model size at the time of deployment, to allow inference speedup and energy savings without significant accuracy loss. While IRS can work with most of these approximations, for the evaluation, we focus on quantization and pruning as these are the most common ones .

# Incremental Randomized Smoothing

<figure id="fig:workflow">
<img src="./figures/workflow.png"" style="width:11.5cm" />
<figcaption>Workflow of IRS from left to right. IRS takes the classifier <span class="math inline"><em>f</em></span> and input <span class="math inline"><em>x</em></span>. IRS reuses the <span class="math inline">$\underline{p_A}$</span> and <span class="math inline">$\overline{p_B}$</span> estimates computed for <span class="math inline"><em>f</em></span> on <span class="math inline"><em>x</em></span> by RS. IRS estimate <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span> from <span class="math inline"><em>f</em></span> and <span class="math inline"><em>f</em><sup><em>p</em></sup></span>. For the smoothed classifier <span class="math inline"><em>g</em><sup><em>p</em></sup></span> obtained from any of the approximate classifiers <span class="math inline"><em>f</em><sup><em>p</em></sup></span> it computes the certified radius by combining <span class="math inline">$\underline{p_A}$</span> and <span class="math inline">$\overline{p_B}$</span> with <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span>.</figcaption>
</figure>

Figure <a href="#fig:workflow" data-reference-type="ref" data-reference="fig:workflow">3</a> illustrates the high-level idea behind the workings of IRS. It takes as input the classifier $`f`$, the updated classifier $`f^p`$, and an input $`x`$. Let $`g`$ and $`g^p`$ denote the smoothed network obtained from $`f`$ and $`f^p`$ using RS respectively. IRS reuses the $`\underline{p_A}`$ and $`\overline{p_B}`$ estimates computed for $`g`$ to compute the certified radius for $`g^p`$.

## Motivation

**Insight 1: Similarity in approximate networks** We observe that for many practical approximations,

<div id="tab:zeta">

|         |            |           |
|:--------|-----------:|----------:|
|         |    CIFAR10 |  ImageNet |
|         | ResNet-110 | ResNet-50 |
| int8    |      0.009 |     0.006 |
| prune10 |      0.010 |     0.008 |

Average $`\zeta_{x}`$ with $`n=1000`$ samples for various approximations.

</div>

<span id="tab:zeta" label="tab:zeta"></span>

$`f`$ and $`f^p`$ produce the same result on most inputs. In this experiment, we estimate the disparity between $`f`$ and $`f^p`$ on Gaussian corruptions of the input $`x`$. We compute a lower confidence bound $`\zeta_{x}`$ such that $`\mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon)) \leq \zeta_{x}`$ for $`\epsilon \sim \mathcal{N}(0, \sigma^2 I)`$.

Table <a href="#tab:zeta" data-reference-type="ref" data-reference="tab:zeta">1</a> presents empirical average $`\zeta_{x}`$ for int8 quantization and pruning $`10\%`$ lowest magnitude weights for some of the networks in our experiments computed over $`500`$ inputs. We compute $`\zeta_{x}`$ value as the binomial confidence upper limit using method with $`n=1000`$ samples with $`\sigma=1`$. The results show that the $`\zeta_{x}`$ value is quite small in all the cases.

**Insight 2: Sample reduction through $`\zeta_{x}`$ estimation** We demonstrate that $`\zeta_{x}`$ estimation for approximate networks is more efficient than running certification from scratch. Fig. <a href="#fig:motivation" data-reference-type="ref" data-reference="fig:motivation">4</a> shows that for the fixed target error $`\chi`$, confidence $`(1 - \alpha)`$ and estimation technique, the number of samples required for estimation peaks, when the actual parameter value is around $`0.5`$ and is smallest around the boundaries. For example, when $`\chi = 0.5\%`$ and $`\alpha = 0.01`$ estimating the unknown binomial proportion will take $`41,500`$ samples if the actual parameter value is $`0.05`$ while achieving the same target error and confidence takes $`216,900`$ samples ($`5.22`$x higher) if the actual parameter value is $`0.5`$. As observed in the previous section, $`\zeta_{x}`$’s value for many practical approximations is close to 0.

Leveraging the observation shown in Fig. <a href="#fig:motivation" data-reference-type="ref" data-reference="fig:motivation">4</a> and given actual value $`\zeta_{x}`$ is close to 0, estimating $`\zeta_{x}`$ with existing binomial proportion estimation techniques is efficient and requires a smaller number of samples. In Appendix <a href="#app:sigma" data-reference-type="ref" data-reference="app:sigma">9.7</a>, we show the distribution of $`\underline{p_A}`$ and $`\overline{p_B}`$ for various cases. We see that $`\underline{p_A}`$ and $`\overline{p_B}`$ do not always lie close to 0 or 1 and have a more dispersed distribution. Thus, estimating those requires more samples. Prior work has theoretically shown that the expected length of the confidence interval for Clopper-Pearson follows a similar trend as in Fig. <a href="#fig:motivation" data-reference-type="ref" data-reference="fig:motivation">4</a>. This theoretical result supports our observation. We show in Appendix <a href="#sec:app_binomial" data-reference-type="ref" data-reference="sec:app_binomial">9.1</a> that this observation is not contingent on a specific estimation method and holds for other popular estimation techniques, e.g., , .

<figure id="fig:motivation">
<div class="center">
<img src="./figures/plot_binomial_proportion.png"" style="width:40.0%" />
</div>
<figcaption>The number of samples for the Clopper-Pearson method to achieve a target error <span class="math inline"><em>χ</em></span> with confidence <span class="math inline">(1 − <em>α</em>)</span>. </figcaption>
</figure>

**Insight 3: Computing the approximate network’s certified radius using $`\zeta_{x}`$** For certification of the approximate network $`g^p`$, our main insight is that estimating $`\zeta_{x}`$ and using that value to compute the certified radius is more efficient than computing RS certified radius from scratch. The next theorem shows how to use estimated value of $`\zeta_{x}`$ to certify $`g^p`$ (the proof is in Appendix <a href="#sec:proofs" data-reference-type="ref" data-reference="sec:proofs">9.2</a>):

<div class="restatable">

theoremirs <span id="thm:irs" label="thm:irs"></span> If a classifier $`f^p`$ is such that for all $`x \in \mathbb{R}^m,  \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon)) \leq \zeta_{x}`$, and classifier $`f`$ satisfies $`\mathbb{P}_\epsilon (f(x+\epsilon) = c_A) \geq \underline{p_A}\geq \overline{p_B}\geq \max_{c \neq c_A} \mathbb{P}_\epsilon (f(x+\epsilon) = c)`$ and $`\underline{p_A}-\zeta_{x}\geq \overline{p_B}+\zeta_{x}`$ then $`g^p`$ satisfies $`g^p(x+\delta) = c_A`$ for all $`\delta`$ satisying $`\|\delta\|_2 \leq \frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}-\zeta_{x}) - \Phi^{-1}(\overline{p_B}+\zeta_{x}))`$

</div>

Theorem <a href="#thm:rs" data-reference-type="ref" data-reference="thm:rs">[thm:rs]</a> considers standard RS for a single network. Our Theorem <a href="#alg:irs" data-reference-type="ref" data-reference="alg:irs">6</a> shows how to use the estimated value of $`\zeta_{x}`$ to transfer the certification guarantees across two networks $`f`$ and $`f^p`$.

## IRS Certification Algorithm

<figure id="alg:irs">
<figure id="alg:irs">
<p><strong>Inputs:</strong> <span class="math inline"><em>f</em><sup><em>p</em></sup></span>: DNN obtained from approximating <span class="math inline"><em>f</em></span>, <span class="math inline"><em>σ</em></span>: standard deviation, <span class="math inline"><em>x</em></span>: input to the DNN, <span class="math inline"><em>n</em><sub><em>p</em></sub></span>: number of Gaussian samples used for certification, <span class="math inline">𝒞<sub><em>f</em></sub></span>: stores the information to be reused from certification of <span class="math inline"><em>f</em></span>, <span class="math inline"><em>α</em></span> and <span class="math inline"><em>α</em><sub><em>ζ</em></sub></span>: confidence parameters, <span class="math inline"><em>γ</em></span>: threshold hyperparameter to switch between estimation methods</p>
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
<span class="math inline"><em>ĉ</em><sub><em>A</em></sub> ← top index in 𝒞<sub><em>f</em></sub>[<em>x</em>]</span><br />
<span class="math inline">$\underline{p_A}\gets \text{lower confidence of $f$ from } \mathcal{C}_f[x]$</span><br />
<strong>If</strong> <span><span class="math inline">$\underline{p_A}&lt; \gamma$</span></span><br />
<span class="math inline"><em>ζ</em><sub><em>x</em></sub> ← EstimateZeta(<em>f</em><sup><em>p</em></sup>, <em>σ</em>, <em>x</em>, <em>n</em><sub><em>p</em></sub>, 𝒞<sub><em>f</em></sub>, <em>α</em><sub><em>ζ</em></sub>)</span><br />
<strong>If</strong> <span><span class="math inline">$\underline{p_A}-\zeta_{x}&gt; \frac{1}{2}$</span></span><br />
<br />
Return prediction <span class="math inline"><em>ĉ</em><sub><em>A</em></sub></span> and radius <span class="math inline">$\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x})$</span><br />
EndIf<br />
Else<br />
<span class="math inline"><em>c</em><em>o</em><em>u</em><em>n</em><em>t</em><em>s</em> ← SampleUnderNoise(<em>f</em><sup><em>p</em></sup>, <em>x</em>, <em>n</em><sub><em>p</em></sub>, <em>σ</em>)</span><br />
<span class="math inline"><em>p</em><sup>′</sup><sub><em>A</em></sub> ← LowerConfidenceBound(</span><br />
<span class="math inline"><em>c</em><em>o</em><em>u</em><em>n</em><em>t</em><em>s</em>[<em>ĉ</em><sub><em>A</em></sub>], <em>n</em><sub><em>p</em></sub>, 1 − (<em>α</em> + <em>α</em><sub><em>ζ</em></sub>))</span><br />
<strong>If</strong> <span><span class="math inline">$p'_A &gt; \frac{1}{2}$</span></span><br />
<br />
Return prediction <span class="math inline"><em>ĉ</em><sub><em>A</em></sub></span> and radius <span class="math inline"><em>σ</em><em>Φ</em><sup>−1</sup>(<em>p</em><sup>′</sup><sub><em>A</em></sub>)</span><br />
EndIf<br />
EndIf<br />
<br />
Return ABSTAIN</p>
</div>
<figcaption>IRS algorithm: Certification with cache</figcaption>
</figure>
<figcaption>IRS algorithm: Certification with cache</figcaption>
</figure>

The Algorithm <a href="#alg:irs" data-reference-type="ref" data-reference="alg:irs">6</a> presents the pseudocode for the IRS algorithm, which extends RS certification from Algorithm <a href="#alg:rs" data-reference-type="ref" data-reference="alg:rs">2</a>. The algorithm takes the modified classifier $`f^p`$ and certifies the robustness of $`g^p`$ around $`x`$. The input $`n_p`$ denotes the number of Gaussian corruptions used by the algorithm.

The IRS algorithm utilizes a cache $`\mathcal{C}_f`$, which stores information obtained from the RS execution of the classifier $`f`$ for each input $`x`$. The cached information is crucial for the operation of IRS. $`\mathcal{C}_f`$ stores the top predicted class index $`\hat{c}_A`$ and its lower confidence bound $`\underline{p_A}`$ for $`f`$ on input $`x`$.

The standard RS algorithm takes a conservative value of $`\overline{p_B}`$ by letting $`\overline{p_B}= 1-\underline{p_A}`$. This works reasonably well in practice and simplifies the computation of certified radius $`\frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}) - \Phi^{-1}(\overline{p_B}))`$ to $`\sigma \Phi^{-1}(\underline{p_A})`$. We make a similar choice in IRS, which simplifies the certified radius calculation from $`\frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}-\zeta_{x}) - \Phi^{-1}(\overline{p_B}+\zeta_{x}))`$ of Theorem <a href="#thm:irs" data-reference-type="ref" data-reference="thm:irs">[thm:irs]</a> to $`\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x})`$ as we state in the next theorem (the proof is in Appendix <a href="#sec:proofs" data-reference-type="ref" data-reference="sec:proofs">9.2</a>):

<div class="restatable">

theoremsimplify <span id="thm:simplify" label="thm:simplify"></span> If $`\underline{p_A}-\zeta_{x}\geq \frac{1}{2},`$ then $`\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x}) \leq \frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}-\zeta_{x}) - \Phi^{-1}(\overline{p_B}+\zeta_{x}))`$

</div>

As per our insight 2 (Section <a href="#sec:motivation" data-reference-type="ref" data-reference="sec:motivation">3.1</a>), binomial confidence interval estimation requires fewer samples for binomial with actual probability close to $`0`$ or $`1`$. IRS can take advtange of this when $`\underline{p_A}`$ is not close to $`1`$. However, when $`\underline{p_A}`$ is close to $`1`$ then there is no benefit of using $`\zeta_{x}`$-based certified radius for $`g^p`$. Therefore, the algorithm uses a threshold hyperparameter $`\gamma`$ close to $`1`$ that is used to switch between certified radius from Theorem <a href="#thm:irs" data-reference-type="ref" data-reference="thm:irs">[thm:irs]</a> and standard RS from Theorem <a href="#thm:rs" data-reference-type="ref" data-reference="thm:rs">[thm:rs]</a>.

If the $`\underline{p_A}`$ is less than the threshold $`\gamma`$, then an estimate of $`\zeta_{x}`$ for classifier $`f^p`$ and the classifier $`f`$ is computed using the $`\text{EstimateZeta}`$ function. We discuss $`\text{EstimateZeta}`$ procedure in the next section. If the $`\underline{p_A}-\zeta_{x}`$ is greater than $`\frac{1}{2}`$, then the top predicted class in the cache is returned as the prediction with the radius $`\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x})`$ as computed in Theorem <a href="#thm:simplify" data-reference-type="ref" data-reference="thm:simplify">[thm:simplify]</a>.

In case, $`\underline{p_A}`$ is greater than the threshold $`\gamma`$, similar to standard RS, the IRS algorithm draws $`n^p`$ samples of $`f^p(x + \epsilon)`$ by running $`n^p`$ noise-corrupted copies of $`x`$ through the classifier $`f^p`$. The function $`\text{SampleUnderNoise}(f^p, x, n_p, \sigma)`$ in the pseudocode draws $`n_p`$ samples of noise, $`\epsilon_1 \dots \epsilon_{n_p} \sim \mathcal{N}(0, \sigma^2 I)`$, runs each $`x + \epsilon_i`$ through the classifier $`f^p`$, and returns a vector of class counts. If the lower confidence bound is greater than $`\frac{1}{2}`$, the top predicted class is returned as the prediction with a radius based on the lower confidence bound $`\underline{p_A}`$.

If the function does certify the input in both of the above cases, it returns ABSTAIN.

The hyperparameters $`\alpha`$ and $`\alpha_\zeta`$ denote confidence of IRS results. The IRS algorithm result is correct with confidence at least $`1-(\alpha+\alpha_\zeta)`$. For the case $`\underline{p_A}\geq \gamma`$, this holds since we follow the same steps as standard RS. The function $`\text{LowerConfidenceBound}(counts[\hat{c}_A], n_p, 1 - (\alpha+\alpha_\zeta ))`$ in the pseudocode returns a one-sided $`1 - (\alpha+\alpha_\zeta )`$ lower confidence interval for the Binomial parameter $`p`$ given a sample $`counts[\hat{c}_A] \sim Binomial(n_p, p)`$. We next state the theorem that shows the confidence of IRS results in the other case when $`\underline{p_A}< \gamma`$ (the proof is in Appendix <a href="#sec:proofs" data-reference-type="ref" data-reference="sec:proofs">9.2</a>):

<div class="restatable">

theoremestimate <span id="thm:estimate" label="thm:estimate"></span> If $`\mathbb{P}_\epsilon(f(x+\epsilon) = f^p(x+\epsilon)) > 1-\zeta_{x}`$ with confidence at least $`1-\alpha_\zeta`$. If classifier $`f`$ satisfies $`\mathbb{P}_\epsilon (f(x+\epsilon) = c_A) \geq \underline{p_A}`$ with confidence at least $`1-\alpha`$. Then for classifier $`f^p`$, $`\mathbb{P}_\epsilon (f^p(x+\epsilon) = c_A) \geq \underline{p_A}-\zeta_{x}`$ with confidence at least $`1-(\alpha+\alpha_\zeta)`$

</div>

## Estimating the Upper Confidence Bound $`\zeta_{x}`$

In this section, we present our method for estimating $`\zeta_{x}`$ such that $`\mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon)) \leq \zeta_{x}`$ with high confidence (Algorithm <a href="#alg:eqlb" data-reference-type="ref" data-reference="alg:eqlb">8</a>). We use the Clopper-Pearson method to estimate the upper confidence bound $`\zeta_{x}`$.

<figure id="alg:eqlb">
<figure id="alg:eqlb">
<p><strong>Inputs:</strong> <span class="math inline"><em>f</em><sup><em>p</em></sup></span>: DNN obtained from approximating <span class="math inline"><em>f</em></span>, <span class="math inline"><em>σ</em></span>: standard deviation, <span class="math inline"><em>x</em></span>: input to the DNN, <span class="math inline"><em>n</em><sub><em>p</em></sub></span>: number of Gaussian samples used for estimating <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span>, <span class="math inline">𝒞<sub><em>f</em></sub></span>: stores the information to be reused from certification of <span class="math inline"><em>f</em></span>, <span class="math inline"><em>α</em><sub><em>ζ</em></sub></span>: confidence parameter<br />
<strong>Output:</strong> Estimated value of <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span></p>
<div class="algorithmic">
<p>ALGORITHM BLOCK (caption below)</p>
<p><br />
<span class="math inline"><em>n</em><sub><em>Δ</em></sub> ← 0</span><br />
<span class="math inline"><em>seeds</em>←</span> seeds for original samples from <span class="math inline">𝒞<sub><em>f</em></sub>[<em>x</em>]</span><br />
<span class="math inline"><em>predictions</em> ← <em>f</em></span>’s predictions on samples from <span class="math inline">𝒞<sub><em>f</em></sub>[<em>x</em>]</span><br />
<strong>For</strong> <span><span class="math inline"><em>i</em> ∈ {1, …<em>n</em><sub><em>p</em></sub>}</span></span><br />
<span class="math inline"><em>ϵ</em> ∼ 𝒩(0, <em>σ</em><sup>2</sup><em>I</em>)</span> using <span class="math inline"><em>seeds</em>[<em>i</em>]</span><br />
<span class="math inline"><em>c</em><sub><em>f</em></sub> ← <em>predictions</em>[<em>i</em>]</span><br />
<span class="math inline"><em>c</em><sub><em>f</em><sup><em>p</em></sup></sub> ← <em>f</em><sup><em>p</em></sup>(<em>x</em> + <em>ϵ</em>)</span><br />
<span class="math inline"><em>n</em><sub><em>Δ</em></sub> ← <em>n</em><sub><em>Δ</em></sub> + <em>I</em>(<em>c</em><sub><em>f</em></sub> ≠ <em>c</em><sub><em>f</em><sup><em>p</em></sup></sub>)</span><br />
EndFor<br />
<br />
Return UpperConfidenceBound(<span class="math inline"><em>n</em><sub><em>Δ</em></sub></span>, <span class="math inline"><em>n</em><sub><em>p</em></sub></span>, <span class="math inline">1 − <em>α</em><sub><em>ζ</em></sub></span>)</p>
</div>
<figcaption>Estimate <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span></figcaption>
</figure>
<figcaption>Estimate <span class="math inline"><em>ζ</em><sub><em>x</em></sub></span></figcaption>
</figure>

We store the $`\textit{seeds}`$ used for randomly generating Gaussian samples while certifying the function $`f`$ in the cache, and we reuse these seeds to generate the same Gaussian samples. $`\textit{seeds}[i]`$ stores the seed used for generating $`i`$-th sample in the RS execution of $`f`$, and $`\textit{predictions}[i]`$ stores the prediction of $`f`$ on the corrsponding $`x+\epsilon`$. We evaluate $`f^p`$ on each corruption $`\epsilon`$ generated from $`\textit{seeds}`$ and match them to predictions by $`f`$. $`c_f`$ and $`c_{f^p}`$ represent the top class prediction by $`f`$ and $`f^p`$ respectively. $`n_\Delta`$ is the count of the number of corruptions $`\epsilon`$ such that $`f`$ and $`f^p`$ do not match on $`x+\epsilon`$.

The function UpperConfidenceBound($`n_\Delta`$, $`n_p`$, $`1-\alpha_\zeta`$) in the pseudocode returns a one-sided $`1-\alpha_\zeta`$ upper confidence interval for the Binomial parameter $`p`$ given a sample $`n_\Delta \sim Binomial(n_p, p)`$. We compute this upper confidence bound using the Clopper-Pearson method. This is similar to how the lower confidence bound is computed in the standard RS Algorithm <a href="#alg:rs" data-reference-type="ref" data-reference="alg:rs">2</a>. It is sound since the Clopper-Pearson method is conservative.

Reusing the seeds for generating noisy samples does not change the certified radius and is 2x faster compared to naive Monte Carlo estimation of $`\zeta_{x}`$ with fresh Gaussian samples. Storing the seeds used in the cache results in a small memory overhead (less than 2MBs for our largest benchmark). We use the same Gaussian samples for estimations of $`\underline{p_A}`$ and $`\zeta_{x}`$. This is equivalent to estimating two functions, $`p(X)`$ and $`q(X)`$, of a random variable $`X`$, where the same set of samples of $`X`$ can be employed for their respective estimations. Theorem <a href="#thm:estimate" data-reference-type="ref" data-reference="thm:estimate">[thm:estimate]</a> makes no assumptions about the independence of estimating $`\underline{p_A}`$ and $`\zeta_{x}`$, thus we can soundly reuse the same Gaussian samples for both estimations.

# Experimental Methodology

We evaluate IRS on CIFAR-10  and ImageNet . On each dataset, we use several classifiers, each with a different $`\sigma`$’s. For an experiment that adds Gaussian corruptions with $`\sigma`$ to the input, we use the network that is trained with Gaussian augmentation with variance $`\sigma^2`$. On CIFAR-10 we use the base classifier a 20-layer and 110-layer residual network. On ImageNet our base classifier is a ResNet-50.

We evaluate IRS on multiple approximations. We consider float16 (fp16), bfloat16 (bf16), and int8 quantizations (Section <a href="#sec:exp_quant" data-reference-type="ref" data-reference="sec:exp_quant">5.1</a>). We show the effectiveness of IRS on pruning approximation in Section <a href="#sec:exp_prune" data-reference-type="ref" data-reference="sec:exp_prune">5.3</a>. For int8 quantization, we use dynamic per-channel quantization mode. from  library. For float16 and bfloat16 quantization, we change the data type of the DNN weights from float32 to the respective types. We perform float32, float16, and bfloat16 inferences on the GPU and int8 inferences on CPU since Pytorch does not support int8 quantization for GPUs yet . For the pruning experiment, we perform the lowest weight magnitude (LWM) pruning. The top-1 accuracy of the networks used in the evaluation and the approximate networks is discussed in Appendix <a href="#sec:eval_nets" data-reference-type="ref" data-reference="sec:eval_nets">9.3</a>.

We ran experiments on a 48-core Intel Xeon Silver 4214R CPU with 2 NVidia RTX A5000 GPUs. IRS is implemented in Python and uses PyTorch 2.0.1. .

We use confidence parameters $`\alpha=0.001`$ for the certification of $`g`$, and $`\alpha_\zeta = 0.001`$ for the estimation of $`\zeta_{x}`$. To establish a fair comparison, we set the baseline confidence with $`\alpha_b = \alpha + \alpha_\zeta = 0.002`$. This choice ensures that both the baseline and IRS, provide certified radii with equal confidence. We use grid search to choose an effective value for $`\gamma`$. A detailed description of our hyperparameter search and its results are described in Section <a href="#sec:ablation" data-reference-type="ref" data-reference="sec:ablation">5.4</a>.

We compute the certified radius $`r`$ when the certification algorithm did not abstain and returned the correct class with radius $`r`$, for both IRS (Algorithm <a href="#alg:irs" data-reference-type="ref" data-reference="alg:irs">6</a>) and the baseline (Algorithm <a href="#alg:rs" data-reference-type="ref" data-reference="alg:rs">2</a>). In other cases, we say that the certified radius $`r=0`$. We compute the *average certified radius* (ACR) by taking the mean of certified radii computed for inputs in the test set. Higher ACR indicates stronger robustness certification guarantees.

**Speedup.** IRS is applicable while certifying multiple similar networks, where it can reuse the certification of one of the networks for faster certification of all other similar networks. We demonstrate the effectiveness of IRS by comparing IRS’s certification time for these other similar networks with the baseline certification from scratch. We do not include the certification time of the first network in the comparison as it adds the same time for both IRS and baseline.

# Experimental Results

We now present our main evaluation results. We consider the float32 representation of the DNN as $`f`$ and a particular approximation as $`f^p`$. However, IRS can be used with any similar $`f`$ and $`f^p`$s, e.g., where $`f`$ is an int8 quantized network and $`f^p`$ is the float32 network. In all of our experiments, we follow a specific procedure:

1.  We certify the smoothed classifier $`g`$ using standard RS with a sample size of $`n`$.

2.  We approximate the base classifier $`f`$ with $`f^p`$.

3.  Using the IRS, we certify smoothed classifier $`g^p`$ by employing Algorithm <a href="#alg:irs" data-reference-type="ref" data-reference="alg:irs">6</a> and utilizing the cached information $`\mathcal{C}_f`$ obtained from the certification of $`g`$.

We compare IRS to the baseline that uses standard non-incremental RS (Algorithm <a href="#alg:rs" data-reference-type="ref" data-reference="alg:rs">2</a>), to certify $`g^p`$. Our results compare ACR and certification time between IRS and the baseline for various $`n_p`$ values.

## Effectiveness of IRS

<div id="tab:avg_speedup">

| Dataset | Architecture | $`ACR`$ | $`T_{\textit{base}}`$ | $`T_\textit{IRS{}}`$ | $`T_\textit{saved}`$ |
|:---|:---|:---|---:|---:|---:|
| CIFAR10 | ResNet-110 | 0.56 | 2.91h | 0.71h | 2.12h |
|  |  | 0.85 | 15.39h | 11.32h | 4.07h |
| ImageNet | ResNet-50 | 0.875 | 27.82h | 22.45h | 5.36h |
|  |  | 0.90 | 53.93h | 40.58h | 13.35h |

Time taken by IRS and baseline for certifying int8 quantized .

</div>

<span id="tab:avg_speedup" label="tab:avg_speedup"></span>

We compare the ACR and the certification time of the baseline and IRS for the common int8 quantization. We use $`n=10^5`$ samples for certification of $`g`$. For certifying $`g^p`$, we consider $`n_p`$ values from $`\{5\%, \dots 50\%\}`$ of $`n`$ and $`\sigma=1`$. We perform experiments on $`500`$ images and compute the total time for certifying $`g^p`$.

<figure id="fig:reduction">
<figure id="fig:reduction_a">
<img src="./figures/mean_radius_vs_time_cifar10.png"" />
<figcaption>ResNet-110 on CIFAR-10</figcaption>
</figure>
<figure id="fig:reduction_b">
<img src="./figures/mean_radius_vs_time_imagenet.png"" />
<figcaption>ResNet-50 on ImageNet</figcaption>
</figure>
<figcaption>Total certification time versus ACR with <span class="math inline"><em>σ</em> = 1.0</span>.</figcaption>
</figure>

Figure <a href="#fig:reduction" data-reference-type="ref" data-reference="fig:reduction">11</a> presents the comparison between IRS and RS for int8 quantization. The x-axis displays the ACR and the y-axis displays the certification time. The plot consists of 10 markers each for the IRS and the baseline representing a specific value of $`n_p`$. Expectedly, the higher the value of $`n_p`$, the higher the average time and ACR. The marker coordinate denotes the ACR and the time for an experiment. In all the cases, IRS consistently takes less certification time to obtain the same ACR.

Figure <a href="#fig:reduction_a" data-reference-type="ref" data-reference="fig:reduction_a">9</a>, for ResNet-110 on CIFAR10, shows that IRS reduced the certification time from 2.91 hours (baseline) to 0.71 hours, resulting in time savings of 2.12 hours (4.1x faster). Moreover, we see that IRS achieves an ACR of more than 0.565, whereas the baseline does not reach this ACR for any of the $`n_p`$ values in our experiments.

Figure <a href="#fig:reduction_b" data-reference-type="ref" data-reference="fig:reduction_b">10</a>, for ResNet-50 on ImageNet, for certifying an ACR of 0.875, IRS substantially reduced certification time from 27.82 hours (baseline) to 22.45 hours, saving approximately 5.36 hours (1.24x faster). Additionally, IRS achieved an ACR of 0.90 and reduced the certification time from 53.93 hours (baseline) to 40.58 hours, resulting in substantial time savings of 13.35 hours (1.33x faster).

## IRS speedups on different quantizations

<div id="tab:avg_speedup">

<table>
<caption>Average IRS speedup for combinations of quantizations and <span class="math inline"><em>σ</em></span>’s.</caption>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Architecture</td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td colspan="3" style="text-align: center;">Quantization</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">1.37x</td>
<td style="text-align: right;">1.29x</td>
<td style="text-align: right;">1.3x</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-20</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.79x</td>
<td style="text-align: right;">1.7x</td>
<td style="text-align: right;">1.77x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">2.85x</td>
<td style="text-align: right;">2.41x</td>
<td style="text-align: right;">2.65x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">1.42x</td>
<td style="text-align: right;">1.35x</td>
<td style="text-align: right;">1.29x</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-110</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.97x</td>
<td style="text-align: right;">1.74x</td>
<td style="text-align: right;">1.77x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">3.02x</td>
<td style="text-align: right;">2.6x</td>
<td style="text-align: right;">2.6x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.2x</td>
<td style="text-align: right;">1.14x</td>
<td style="text-align: right;">1.19x</td>
</tr>
<tr>
<td style="text-align: left;">ImageNet</td>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">1.43x</td>
<td style="text-align: right;">1.31x</td>
<td style="text-align: right;">1.43x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;">2.04x</td>
<td style="text-align: right;">1.69x</td>
<td style="text-align: right;">1.80x</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:avg_speedup" label="tab:avg_speedup"></span>

Next, we study if IRS can handle other kinds of quantization. We perform experiments for 10 different values of $`n_p`$ along with distinct approximations, and 3 values of $`\sigma`$. Since this would take months of experiment time with $`n`$ and $`n_p`$ values from Section <a href="#sec:exp_quant" data-reference-type="ref" data-reference="sec:exp_quant">5.1</a>, for the rest of the experiments we use smaller values for these parameters. In these experiments, we compute the relative speedup due to IRS in comparison to the baseline. We use $`n=10^4`$ for samples for certification of $`g`$. For certifying $`g^p`$, we consider $`n_p`$ values from $`\{1\%, \dots 10\%\}`$ of $`n`$. For CIFAR10, we consider $`\sigma \in \{0.25, 0.5, 1.0`$}, and for ImageNet, we consider $`\sigma \in \{0.5, 1.0, 2.0\}`$ as in the previous work. We validated that the speedups for int8 quantization in this section for ResNet-50-ImageNet and ResNet-110-CIFAR10 are similar to those studied in Section <a href="#sec:exp_quant" data-reference-type="ref" data-reference="sec:exp_quant">5.1</a>.

To quantify IRS’s average speedup over the baseline, we employ an approximate area under the curve (AOC) analysis. Specifically, we plot the certification time against the ACR. In most cases, IRS certifies a larger ACR compared to the baseline, resulting in regions on the x-axis where IRS exists but the baseline does not. To ensure a conservative estimation, we calculate the speedup only within the range where both IRS and the baseline exist. We determine the speedup by computing the ratio of the AOC for IRS to the AOC for the baseline within this common range. Table <a href="#tab:avg_speedup" data-reference-type="ref" data-reference="tab:avg_speedup">3</a> summarizes the average speedups for all quantization experiments.

We observe that IRS gets a larger speedup for smoothing with larger $`\sigma`$ since on average the $`\underline{p_A}`$ values are smaller. Appendix <a href="#app:sigma" data-reference-type="ref" data-reference="app:sigma">9.7</a> presents a further justification for this observation. Appendix <a href="#app:eval" data-reference-type="ref" data-reference="app:eval">[app:eval]</a> presents further experiments with all combinations of DNNs, $`\sigma`$, and quantizations.

<div id="tab:avg_acc_speedup">

<table>
<caption>Average IRS certification accuracy speedup for combinations of radius, quantizations, and <span class="math inline"><em>σ</em></span>’s on ImageNet ResNet-50.</caption>
<tbody>
<tr>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td style="text-align: left;">Radius</td>
<td colspan="3" style="text-align: center;">Quantization</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">3.92x</td>
<td style="text-align: right;">1.85x</td>
<td style="text-align: right;">3.61x</td>
</tr>
<tr>
<td style="text-align: left;">0.5</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">2.7x</td>
<td style="text-align: right;">1.6x</td>
<td style="text-align: right;">2.67x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.75</td>
<td style="text-align: right;">1.2x</td>
<td style="text-align: right;">1.12x</td>
<td style="text-align: right;">1.23x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">4.79x</td>
<td style="text-align: right;">3.07x</td>
<td style="text-align: right;">4.44x</td>
</tr>
<tr>
<td style="text-align: left;">1.0</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">2.6x</td>
<td style="text-align: right;">2.25x</td>
<td style="text-align: right;">2.6x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.5</td>
<td style="text-align: right;">1.53x</td>
<td style="text-align: right;">1.38x</td>
<td style="text-align: right;">1.54x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">5.3x</td>
<td style="text-align: right;">3.31x</td>
<td style="text-align: right;">4.06x</td>
</tr>
<tr>
<td style="text-align: left;">2.0</td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;">2.79x</td>
<td style="text-align: right;">2.46x</td>
<td style="text-align: right;">3.94x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">3.0</td>
<td style="text-align: right;">1.51x</td>
<td style="text-align: right;">1.48x</td>
<td style="text-align: right;">1.80x</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:avg_acc_speedup" label="tab:avg_acc_speedup"></span>

## IRS speedups on Pruned Models

<div id="tab:prune">

<table>
<caption>Average IRS speedup for combinations of pruning ratio and <span class="math inline"><em>σ</em></span>’s.</caption>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Architecture</td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td colspan="3" style="text-align: center;">Prune</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;">5%</td>
<td style="text-align: right;">10%</td>
<td style="text-align: right;">20%</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">1.3x</td>
<td style="text-align: right;">1.25x</td>
<td style="text-align: right;">0.99x</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-20</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.63x</td>
<td style="text-align: right;">1.39x</td>
<td style="text-align: right;">1.13x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">2.5x</td>
<td style="text-align: right;">2.09x</td>
<td style="text-align: right;">1.39x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">1.35x</td>
<td style="text-align: right;">1.24x</td>
<td style="text-align: right;">1.04x</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-110</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.83x</td>
<td style="text-align: right;">1.6x</td>
<td style="text-align: right;">1.23x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">2.7x</td>
<td style="text-align: right;">2.25x</td>
<td style="text-align: right;">1.63x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">1.19x</td>
<td style="text-align: right;">1.04x</td>
<td style="text-align: right;">0.87x</td>
</tr>
<tr>
<td style="text-align: left;">ImageNet</td>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">1.36x</td>
<td style="text-align: right;">1.15x</td>
<td style="text-align: right;">0.87x</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;">1.87x</td>
<td style="text-align: right;">1.54x</td>
<td style="text-align: right;">1.01x</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:prune" label="tab:prune"></span>

In this experiment, we study IRS’s ability to certify beyond quantized models. We employ $`l_1`$ unstructured pruning, which prunes the fraction of the lowest $`l_1`$ magnitude weights from the DNN. Table <a href="#tab:prune" data-reference-type="ref" data-reference="tab:prune">5</a> presents the average IRS speedup for DNNs obtained by pruning $`5\%, 10\%`$ and $`20\%`$ weights. The speedups range from 0.99x to 2.7x. As the DNN is pruned more aggressively, it’s expected that IRS’s speedup will be lower. This is due to higher values of $`\zeta_{x}`$ associated with aggressive pruning. In Appendix <a href="#sec:zeta_app" data-reference-type="ref" data-reference="sec:zeta_app">9.4</a>, we provide average $`\zeta_{x}`$ values for all approximations. Compared to pruning, quantization typically yields smaller $`\zeta_{x}`$ values, making IRS more effective for quantization.

<div id="tab:n_ablation">

<table>
<caption>Average IRS speedup for combinations of <span class="math inline"><em>n</em></span>, <span class="math inline"><em>σ</em></span>’s, and quantizations for ResNet-20 on CIFAR10.</caption>
<tbody>
<tr>
<td style="text-align: left;"><span class="math inline"><em>n</em></span></td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td colspan="3" style="text-align: center;">Quantization</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: left;">1.37x</td>
<td style="text-align: right;">1.29x</td>
<td style="text-align: right;">1.3x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline">10<sup>4</sup></span></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: left;">1.79x</td>
<td style="text-align: right;">1.7x</td>
<td style="text-align: right;">1.77x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: left;">2.85x</td>
<td style="text-align: right;">2.41x</td>
<td style="text-align: right;">2.65x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: left;">1.22x</td>
<td style="text-align: right;">1.11x</td>
<td style="text-align: right;">1.27x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline">10<sup>5</sup></span></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: left;">1.73x</td>
<td style="text-align: right;">1.4x</td>
<td style="text-align: right;">1.86x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: left;">3.88x</td>
<td style="text-align: right;">2.40x</td>
<td style="text-align: right;">4.31x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: left;">1.12x</td>
<td style="text-align: right;">0.93x</td>
<td style="text-align: right;">1.15x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"><span class="math inline">10<sup>6</sup></span></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: left;">1.97x</td>
<td style="text-align: right;">1.04x</td>
<td style="text-align: right;">2.25x</td>
<td style="text-align: right;"></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: left;">4.58x</td>
<td style="text-align: right;">1.25x</td>
<td style="text-align: right;">5.85x</td>
<td style="text-align: right;"></td>
</tr>
</tbody>
</table>

</div>

<span id="tab:n_ablation" label="tab:n_ablation"></span>

## Ablation Studies

Next, we show the effect of $`\gamma`$ on ACR. In Appendix <a href="#app:n" data-reference-type="ref" data-reference="app:n">9.5</a> we show IRS speedup on distinct values of $`n`$.

<div id="tab:gamma">

|            |           |            |           |
|:-----------|:----------|:-----------|:----------|
| $`\gamma`$ | CIFAR10   | CIFAR10    | ImageNet  |
|            | ResNet-20 | ResNet-110 | ResNet-50 |
| 0.9        | 0.438     | 0.436      | 0.458     |
| 0.95       | 0.442     | 0.439      | 0.464     |
| 0.975      | 0.445     | 0.441      | 0.465     |
| 0.99       | **0.446** | **0.443**  | 0.466     |
| 0.995      | 0.445     | 0.442      | **0.467** |
| 0.999      | 0.444     | 0.442      | 0.464     |

ACR for each $`\gamma`$.

</div>

<span id="tab:gamma" label="tab:gamma"></span>

For each DNN architecture, we chose the hyperparameter $`\gamma`$ by running IRS to certify a small subset of the validation set images for certifying the int8 quantized DNN and comparing the ACR. The choice of $`\gamma`$ has no effect on certification time, as we perform $`n_p`$ inferences in both cases, $`\underline{p_A} < \gamma`$ and $`\underline{p_A} > \gamma`$. We use the same $`\gamma`$ for each DNN irrespective of the approximation and $`\sigma`$. We use the grid search to choose the best value of gamma from the set $`\{0.9, 0.95, 0.975, 0.99, 0.999\}`$. Table <a href="#tab:gamma" data-reference-type="ref" data-reference="tab:gamma">7</a> presents the ACR obtained for each $`\gamma`$. We chose $`\gamma`$ as $`0.99`$ for CIFAR10 networks and $`0.995`$ for the ImageNet networks since they result in the highest ACR.

# Related Work

The scalability of traditional program verification has been significantly improved by incremental verification, which has been applied on an industrial scale . Incremental program analysis tasks achieve faster analysis of individual commits by reusing partial results , constraints , and precision information  from previous runs.

Several methods have been introduced in recent years to certify the properties of DNNs deterministically  and probabilisticly . Researchers used incremental certification speed up DNN certification  – these works apply complete and incomplete deterministic certification using formal logic cannot scale to e.g., ImageNet. In contrast, we propose incremental probabilistic certification with Randomized Smoothing, which enables much greater scalability.

introduced the addition of Gaussian noise to achieve $`l_2`$-robustness results. Several extensions to this technique utilize different types of noise distributions and radius calculations to determine certificates for general $`l_p`$-balls. and derived recipes for determining certificates for $`p = 1, 2`$, and $`\infty`$. , , and presented extensions to discrete perturbations such as $`l_0`$-perturbations, while , , , and explored extensions to graphs, patches, and point cloud manipulations. presented theoretical derivations for the application of both continuous and discrete smoothing measures, while improved certificates by using gradient information. used ensembles to improve the certificate.

Beyond norm-balls certificates, and presented how geometric operations such as rotation or translation can be certified via Randomized Smoothing. and demonstrated how the certificates can be extended from the setting of classification to regression (and object detection) and segmentation, respectively. For classification, extended certificates from just the top-1 class to the top-k classes, while certified the confidence of the classifier, not just the top-class prediction. used Randomized Smoothing to defend against data poisoning attacks. These RS extensions (using different noise distributions, perturbations, and geometric operations) are orthogonal to the standard RS approach from . While these extensions have been shown to improve the overall bredth of RS, IRS is complementary to these extensions.

# Limitations

We showed that IRS is effective at certifying the smoothed version of the approximated DNN. However, there are certain limitations to the effectiveness of IRS. First, the IRS algorithm requires a cache with the top predicted class index, its lower confidence bound, and the seeds for Gaussian corruptions obtained from the RS execution of the original classifier. However, storing this additional information is reasonable since it has negligible memory overhead and is a byproduct of certification (as trustworthy ML matures, we anticipate that this information will be shipped with pre-certified networks for reproducibility purposes).

The smoothing parameter $`\sigma`$ used in IRS affects its efficiency, with larger values of $`\sigma`$ generally leading to better results. As a consequence, we observed a smaller speedup when using a smaller value of $`\sigma`$ (e.g., 0.25 on CIFAR10) compared to a larger value (e.g., 1 on CIFAR10). The value of $`\sigma`$ offers a trade-off between robustness and accuracy. By choosing a larger $`\sigma`$, one can improve robustness but it may lead to a loss of accuracy in the model.

IRS targets fast certification while maintaining a sufficiently large radius. Therefore, we considered $`n_p`$ smaller than $`50\%`$ of $`n`$ for our evaluation. However, IRS certified radius can be smaller than the non-incremental RS, provided the user has a larger sample budget. In our experiment in Appendix <a href="#app:np" data-reference-type="ref" data-reference="app:np">9.6</a> we test IRS on larger $`n_p`$ and observe that IRS is better than baseline for $`n_p`$ less than $`70\%`$ of $`n`$. This is particularly advantageous when computational resources are limited.

# Conclusion

We propose IRS, the first incremental approach for probabilistic DNN certification. IRS leverages the certification guarantees obtained from the smoothed model to certify a smoothed approximated model with very few samples. Reusing the computation of original guarantees significantly reduces the computational cost of certification while maintaining strong robustness guarantees. IRS speeds up certification up to 4.1x over the standard non-incremental RS baseline on state-of-the-art classification models. We anticipate that IRS can be particularly useful for approximate tuning when the users need to analyze the robustness of multiple similar networks. Further, one can easily ship the certification cache to allow other users to further modify these networks based on their specific device and application needs and recertify the new network. We believe that our approach paves the way for efficient and effective certification of DNNs in real-world applications.

# ACKNOWLEDGMENTS

We thank the anonymous reviewers for their comments. This research was supported in part by NSF Grants No. CCF-1846354, CCF-2217144, CCF-2238079, CCF-2313028, CCF-2316233, CNS-2148583, USDA NIFA Grant No. NIFA-2024827 and Google Research Scholar award.

# References

<div class="thebibliography">

Alan Agresti and Brent A. Coull Approximate is better than “exact” for interval estimation of binomial proportions *The American Statistician*, 52 (2): 119–126, 1998. . URL <https://doi.org/10.1080/00031305.1998.10480550>. **Abstract:** Abstract For interval estimation of a proportion, coverage probabilities tend to be too large for "exact" confidence intervals based on inverting the binomial test and too small for the interval based on inverting the Wald large-sample normal test (i.e., sample proportion ± z-score × estimated standard error). Wilson’s suggestion of inverting the related score test with null rather than estimated standard error yields coverage probabilities close to nominal confidence levels, even for very small sample sizes. The 95% score interval has similar behavior as the adjusted Wald interval obtained after adding two "successes" and two "failures" to the sample. In elementary courses, with the score and adjusted Wald methods it is unnecessary to provide students with awkward sample size guidelines. (@doi:10.1080/00031305.1998.10480550)

Filippo Amato, Alberto López, Eladia María Peña-Méndez, Petr Vaňhara, Aleš Hampl, and Josef Havel Artificial neural networks in medical diagnosis *Journal of Applied Biomedicine*, 11 (2): 47–58, 2013. **Abstract:** An extensive amount of information is currently available to clinical specialists, ranging from details of clinical symptoms to various types of biochemical data and outputs of imaging devices. Each type of data provides information that must be evaluated and assigned to a particular pathology during the diagnostic process. To streamline the diagnostic process in daily routine and avoid misdiagnosis, artificial intelligence methods (especially computer aided diagnosis and artificial neural networks) can be employed. These adaptive learning algorithms can handle diverse types of medical data and integrate them into categorized outputs. In this paper, we briefly review and discuss the philosophy, capabilities, and limitations of artificial neural networks in medical diagnosis through selected examples. (@AMATO201347)

Wolfgang Balzer, Masanobu Takahashi, Jun Ohta, and Kazuo Kyuma Weight quantization in boltzmann machines *Neural Networks*, 4 (3): 405–409, 1991. (@balzer1991weight)

Dirk Beyer, Stefan Löwe, Evgeny Novikov, Andreas Stahlbauer, and Philipp Wendler Precision reuse for efficient regression verification In *Proceedings of the 2013 9th Joint Meeting on Foundations of Software Engineering*, ESEC/FSE 2013, page 389–399, New York, NY, USA, 2013. Association for Computing Machinery. ISBN 9781450322379. . URL <https://doi.org/10.1145/2491411.2491429>. **Abstract:** Continuous testing during development is a well-established technique for software-quality assurance. Continuous model checking from revision to revision is not yet established as a standard practice, because the enormous resource consumption makes its application impractical. Model checkers compute a large number of verification facts that are necessary for verifying if a given specification holds. We have identified a category of such intermediate results that are easy to store and efficient to reuse: abstraction precisions. The precision of an abstract domain specifies the level of abstraction that the analysis works on. Precisions are thus a precious result of the verification effort and it is a waste of resources to throw them away after each verification run. In particular, precisions are reasonably small and thus easy to store; they are easy to process and have a large impact on resource consumption. We experimentally show the impact of precision reuse on industrial verification problems created from 62 Linux kernel device drivers with 1119 revisions. (@10.1145/2491411.2491429)

Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al End to end learning for self-driving cars *arXiv preprint arXiv:1604.07316*, 2016. **Abstract:** We trained a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands. This end-to-end approach proved surprisingly powerful. With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways. It also operates in areas with unclear visual guidance such as in parking lots and on unpaved roads. The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal. We never explicitly trained it to detect, for example, the outline of roads. Compared to explicit decomposition of the problem, such as lane marking detection, path planning, and control, our end-to-end system optimizes all processing steps simultaneously. We argue that this will eventually lead to better performance and smaller systems. Better performance will result because the internal components self-optimize to maximize overall system performance, instead of optimizing human-selected intermediate criteria, e.g., lane detection. Such criteria understandably are selected for ease of human interpretation which doesn’t automatically guarantee maximum system performance. Smaller networks are possible because the system learns to solve the problem with the minimal number of processing steps. We used an NVIDIA DevBox and Torch 7 for training and an NVIDIA DRIVE(TM) PX self-driving car computer also running Torch 7 for determining where to drive. The system operates at 30 frames per second (FPS). (@bojarski2016end)

Aleksandar Bojchevski, Johannes Gasteiger, and Stephan Günnemann Efficient robustness certificates for discrete data: Sparsity-aware randomized smoothing for graphs, images and more 2023. **Abstract:** Existing techniques for certifying the robustness of models for discrete data either work only for a small class of models or are general at the expense of efficiency or tightness. Moreover, they do not account for sparsity in the input which, as our findings show, is often essential for obtaining non-trivial guarantees. We propose a model-agnostic certificate based on the randomized smoothing framework which subsumes earlier work and is tight, efficient, and sparsity-aware. Its computational complexity does not depend on the number of discrete categories or the dimension of the input (e.g. the graph size), making it highly scalable. We show the effectiveness of our approach on a wide variety of models, datasets, and tasks – specifically highlighting its use for Graph Neural Networks. So far, obtaining provable guarantees for GNNs has been difficult due to the discrete and non-i.i.d. nature of graph data. Our method can certify any GNN and handles perturbations to both the graph structure and the node attributes. (@bojchevski2023efficient)

Rudy Bunel, Jingyue Lu, Ilker Turkaslan, Pushmeet Kohli, P Torr, and P Mudigonda Branch and bound for piecewise linear neural network verification *Journal of Machine Learning Research*, 21 (2020), 2020. **Abstract:** The success of Deep Learning and its potential use in many safety-critical applications has motivated research on formal verification of Neural Network (NN) models. In this context, verification involves proving or disproving that an NN model satisfies certain input-output properties. Despite the reputation of learned NN models as black boxes, and the theoretical hardness of proving useful properties about them, researchers have been successful in verifying some classes of models by exploiting their piecewise linear structure and taking insights from formal methods such as Satisifiability Modulo Theory. However, these methods are still far from scaling to realistic neural networks. To facilitate progress on this crucial area, we exploit the Mixed Integer Linear Programming (MIP) formulation of verification to propose a family of algorithms based on Branch-and-Bound (BaB). We show that our family contains previous verification methods as special cases. With the help of the BaB framework, we make three key contributions. Firstly, we identify new methods that combine the strengths of multiple existing approaches, accomplishing significant performance improvements over previous state of the art. Secondly, we introduce an effective branching strategy on ReLU non-linearities. This branching strategy allows us to efficiently and successfully deal with high input dimensional problems with convolutional network architecture, on which previous methods fail frequently. Finally, we propose comprehensive test data sets and benchmarks which includes a collection of previously released testcases. We use the data sets to conduct a thorough experimental comparison of existing and new algorithms and to provide an inclusive analysis of the factors impacting the hardness of verification problems. (@bunel2020branch)

Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy Tvm: An automated end-to-end optimizing compiler for deep learning In *Proceedings of the 13th USENIX Conference on Operating Systems Design and Implementation*, OSDI’18, page 579–594, USA, 2018. USENIX Association. ISBN 9781931971478. **Abstract:** There is an increasing need to bring machine learning to a wide diversity of hardware devices. Current frameworks rely on vendor-specific operator libraries and optimize for a narrow range of server-class GPUs. Deploying workloads to new platforms - such as mobile phones, embedded devices, and accelerators (e.g., FPGAs, ASICs) - requires significant manual effort. We propose TVM, a compiler that exposes graph-level and operator-level optimizations to provide performance portability to deep learning workloads across diverse hardware back-ends. TVM solves optimization challenges specific to deep learning, such as high-level operator fusion, mapping to arbitrary hardware primitives, and memory latency hiding. It also automates optimization of low-level programs to hardware characteristics by employing a novel, learning-based cost modeling method for rapid exploration of code optimizations. Experimental results show that TVM delivers performance across hardware back-ends that are competitive with state-of-the-art, hand-tuned libraries for low-power CPU, mobile GPU, and server-class GPUs. We also demonstrate TVM’s ability to target new accelerator back-ends, such as the FPGA-based generic deep learning accelerator. The system is open sourced and in production use inside several major companies. (@10.5555/3291168.3291211)

Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy Learning to optimize tensor programs In *Proceedings of the 32nd International Conference on Neural Information Processing Systems*, NIPS’18, page 3393–3404, Red Hook, NY, USA, 2018. Curran Associates Inc. **Abstract:** We introduce a learning-based framework to optimize tensor programs for deep learning workloads. Efficient implementations of tensor operators, such as matrix multiplication and high dimensional convolution, are key enablers of effective deep learning systems. However, existing systems rely on manually optimized libraries such as cuDNN where only a narrow range of server class GPUs are well-supported. The reliance on hardware-specific operator libraries limits the applicability of high-level graph optimizations and incurs significant engineering costs when deploying to new hardware targets. We use learning to remove this engineering burden. We learn domain-specific statistical cost models to guide the search of tensor operator implementations over billions of possible program variants. We further accelerate the search by effective model transfer across workloads. Experimental results show that our framework delivers performance competitive with state-of-the-art hand-tuned libraries for low-power CPU, mobile GPU, and server-class GPU. (@10.5555/3327144.3327258)

C. J. Clopper and E. S. Pearson The use of confidence or fiducial limits illustrated in the case of the binomial *Biometrika*, 26 (4): 404–413, 1934. ISSN 00063444. URL <http://www.jstor.org/stable/2331986>. **Abstract:** THE USE OF CONFIDENCE OR FIDUCIAL LIMITS ILLUSTRATED IN THE CASE OF THE BINOMIAL Get access C. J. CLOPPER, B.Sc., C. J. CLOPPER, B.Sc. Search for other works by this author on: Oxford Academic Google Scholar E. S. PEARSON, D.Sc. E. S. PEARSON, D.Sc. Search for other works by this author on: Oxford Academic Google Scholar Biometrika, Volume 26, Issue 4, December 1934, Pages 404–413, https://doi.org/10.1093/biomet/26.4.404 Published: 01 December 1934 (@10.2307/2331986)

Jeremy M. Cohen, Elan Rosenfeld, and J. Zico Kolter Certified adversarial robustness via randomized smoothing In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA*, volume 97 of *Proceedings of Machine Learning Research*, pages 1310–1320. PMLR, 2019. URL <http://proceedings.mlr.press/v97/cohen19c.html>. **Abstract:** We show how to turn any classifier that classifies well under Gaussian noise into a new classifier that is certifiably robust to adversarial perturbations under the $\\}ell_2$ norm. This "randomized smoothing" technique has been proposed recently in the literature, but existing guarantees are loose. We prove a tight robustness guarantee in $\\}ell_2$ norm for smoothing with Gaussian noise. We use randomized smoothing to obtain an ImageNet classifier with e.g. a certified top-1 accuracy of 49% under adversarial perturbations with $\\}ell_2$ norm less than 0.5 (=127/255). No certified defense has been shown feasible on ImageNet except for smoothing. On smaller-scale datasets where competing approaches to certified $\\}ell_2$ robustness are viable, smoothing delivers higher certified accuracies. Our strong empirical results suggest that randomized smoothing is a promising direction for future research into adversarially robust classification. Code and models are available at http://github.com/locuslab/smoothing. (@DBLP:conf/icml/CohenRK19)

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei Imagenet: A large-scale hierarchical image database In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009. **Abstract:** The explosion of image data on the Internet has the potential to foster more sophisticated and robust models and algorithms to index, retrieve, organize and interact with images and multimedia data. But exactly how such data can be harnessed and organized remains a critical problem. We introduce here a new database called "ImageNet", a large-scale ontology of images built upon the backbone of the WordNet structure. ImageNet aims to populate the majority of the 80,000 synsets of WordNet with an average of 500–1000 clean and full resolution images. This will result in tens of millions of annotated images organized by the semantic hierarchy of WordNet. This paper offers a detailed analysis of ImageNet in its current state: 12 subtrees with 5247 synsets and 3.2 million images in total. We show that ImageNet is much larger in scale and diversity and much more accurate than the current image datasets. Constructing such a large-scale database is a challenging task. We describe the data collection scheme with Amazon Mechanical Turk. Lastly, we illustrate the usefulness of ImageNet through three simple applications in object recognition, image classification and automatic object clustering. We hope that the scale, accuracy, diversity and hierarchical structure of ImageNet can offer unparalleled opportunities to researchers in the computer vision community and beyond. (@deng2009imagenet)

Krishnamurthy (Dj) Dvijotham, Jamie Hayes, Borja Balle, Zico Kolter, Chongli Qin, Andras Gyorgy, Kai Xiao, Sven Gowal, and Pushmeet Kohli A framework for robustness certification of smoothed classifiers using f-divergences In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=SJlKrkSFPH>. **Abstract:** Formal verification techniques that compute provable guarantees on properties of machine learning models, like robustness to norm-bounded adversarial perturbations, have yielded impressive results. Although most techniques developed so far requires knowledge of the architecture of the machine learning model and remains hard to scale to complex prediction pipelines, the method of randomized smoothing has been shown to overcome many of these obstacles. By requiring only black-box access to the underlying model, randomized smoothing scales to large architectures and is agnostic to the internals of the network. However, past work on randomized smoothing has focused on restricted classes of smoothing measures or perturbations (like Gaussian or discrete) and has only been able to prove robustness with respect to simple norm bounds. In this paper we introduce a general framework for proving robustness properties of smoothed machine learning models in the black-box setting. Specifically, we extend randomized smoothing procedures to handle arbitrary smoothing measures and prove robustness of the smoothed classifier by using $f$-divergences. Our methodology achieves state-of-the-art}certified robustness on MNIST, CIFAR-10 and ImageNet and also audio classification task, Librispeech, with respect to several classes of adversarial perturbations. (@Dvijotham2020A)

Emile Fiesler, Amar Choudry, and H John Caulfield Weight discretization paradigm for optical neural networks In *Optical interconnections and networks*, volume 1281, pages 164–173. SPIE, 1990. **Abstract:** Neural networks are a primary candidate architecture for optical computing. One of the major problems in using neural networks for optical computers is that the information holders: the interconnection strengths (or weights) are normally real valued (continuous), whereas optics (light) is only capable of representing a few distinguishable intensity levels (discrete). In this paper a weight discretization paradigm is presented for back(ward error) propagation neural networks which can work with a very limited number of discretization levels. The number of interconnections in a (fully connected) neural network grows quadratically with the number of neurons of the network. Optics can handle a large number of interconnections because of the fact that light beams do not interfere with each other. A vast amount of light beams can therefore be used per unit of area. However the number of different values one can represent in a light beam is very limited. A flexible, portable (machine independent) neural network software package which is capable of weight discretization, is presented. The development of the software and some experiments have been done on personal computers. The major part of the testing, which requires a lot of computation, has been done using a CRAY X-MP/24 super computer. (@fiesler1990weight)

Marc Fischer, Maximilian Baader, and Martin Vechev Certified defense to image transformations via randomized smoothing In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 8404–8417. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/5fb37d5bbdbbae16dea2f3104d7f9439-Paper.pdf>. **Abstract:** We extend randomized smoothing to cover parameterized transformations (e.g., rotations, translations) and certify robustness in the parameter space (e.g., rotation angle). This is particularly challenging as interpolation and rounding effects mean that image transformations do not compose, in turn preventing direct certification of the perturbed image (unlike certification with $\\}ell^p$ norms). We address this challenge by introducing three different kinds of defenses, each with a different guarantee (heuristic, distributional and individual) stemming from the method used to bound the interpolation error. Importantly, we show how individual certificates can be obtained via either statistical error bounds or efficient online inverse computation of the image transformation. We provide an implementation of all methods at this https URL. (@NEURIPS2020_5fb37d5b)

Marc Fischer, Maximilian Baader, and Martin Vechev Scalable certified segmentation via randomized smoothing 2022. **Abstract:** We present a new certification method for image and point cloud segmentation based on randomized smoothing. The method leverages a novel scalable algorithm for prediction and certification that correctly accounts for multiple testing, necessary for ensuring statistical guarantees. The key to our approach is reliance on established multiple-testing correction mechanisms as well as the ability to abstain from classifying single pixels or points while still robustly segmenting the overall input. Our experimental evaluation on synthetic data and challenging datasets, such as Pascal Context, Cityscapes, and ShapeNet, shows that our algorithm can achieve, for the first time, competitive accuracy and certification guarantees on real-world segmentation tasks. We provide an implementation at https://github.com/eth-sri/segmentation-smoothing. (@fischer2022scalable)

Marc Fischer, Christian Sprecher, Dimitar I. Dimitrov, Gagandeep Singh, and Martin T. Vechev Shared certificates for neural network verification In Sharon Shoham and Yakir Vizel, editors, *Computer Aided Verification - 34th International Conference, CAV 2022, Haifa, Israel, August 7-10, 2022, Proceedings, Part I*, volume 13371 of *Lecture Notes in Computer Science*, pages 127–148. Springer, 2022. . URL <https://doi.org/10.1007/978-3-031-13185-1_7>. **Abstract:** Abstract Existing neural network verifiers compute a proof that each input is handled correctly under a given perturbation by propagating a symbolic abstraction of reachable values at each layer. This process is repeated from scratch independently for each input (e.g., image) and perturbation (e.g., rotation), leading to an expensive overall proof effort when handling an entire dataset. In this work, we introduce a new method for reducing this verification cost without losing precision based on a key insight that abstractions obtained at intermediate layers for different inputs and perturbations can overlap or contain each other. Leveraging our insight, we introduce the general concept of shared certificates, enabling proof effort reuse across multiple inputs to reduce overall verification costs. We perform an extensive experimental evaluation to demonstrate the effectiveness of shared certificates in reducing the verification cost on a range of datasets and attack specifications on image classifiers including the popular patch and geometric perturbations. We release our implementation at https://github.com/eth-sri/proof-sharing . (@DBLP:conf/cav/FischerSDSV22)

Jonathan Frankle and Michael Carbin The lottery ticket hypothesis: Finding sparse, trainable neural networks In *Proc. International Conference on Learning Representations (ICLR)*, 2019. **Abstract:** Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance. We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective. We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy. (@DBLP:conf/iclr/FrankleC19)

Zhidong Gao, Rui Hu, and Yanmin Gong Certified robustness of graph classification against topology attack with randomized smoothing In *GLOBECOM 2020 - 2020 IEEE Global Communications Conference*, pages 1–6, 2020. . **Abstract:** Graph classification has practical applications in diverse fields. Recent studies show that graph-based machine learning models are especially vulnerable to adversarial perturbations due to the non i.i. d nature of graph data. By adding or deleting a small number of edges in the graph, adversaries could greatly change the graph label predicted by a graph classification model. In this work, we propose to build a smoothed graph classification model with certified robustness guarantee. We have proven that the resulting graph classification model would output the same prediction for a graph under l \<sub xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"\>0\</sub\> bounded adversarial perturbation. We also evaluate the effectiveness of our approach under graph convolutional network (GCN) based multi-class graph classification model. (@9322576)

Miklós Z. Horváth, Mark Niklas Mueller, Marc Fischer, and Martin Vechev Boosting randomized smoothing with variance reduced classifiers In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=mHu2vIds_-b>. **Abstract:** Randomized Smoothing (RS) is a promising method for obtaining robustness certificates by evaluating a base model under noise. In this work, we: (i) theoretically motivate why ensembles are a particularly suitable choice as base models for RS, and (ii) empirically confirm this choice, obtaining state-of-the-art results in multiple settings. The key insight of our work is that the reduced variance of ensembles over the perturbations introduced in RS leads to significantly more consistent classifications for a given input. This, in turn, leads to substantially increased certifiable radii for samples close to the decision boundary. Additionally, we introduce key optimizations which enable an up to 55-fold decrease in sample complexity of RS for predetermined radii, thus drastically reducing its computational overhead. Experimentally, we show that ensembles of only 3 to 10 classifiers consistently improve on their strongest constituting model with respect to their average certified radius (ACR) by 5% to 21% on both CIFAR10 and ImageNet, achieving a new state-of-the-art ACR of 0.86 and 1.11, respectively. We release all code and models required to reproduce our results at https://github.com/eth-sri/smoothing-ensembles. (@horvth2022boosting)

ISO Standard, International Organization for Standardization, March 2021. **Abstract:** The paper deals with the set-up and the application of an Artificial Intelligence technique based on Neural Networks (NNs) to gas turbine diagnostics, in order to evaluate its capabilities and its robustness. The data used for both training and testing the NNs were generated by means of a Cycle Program, calibrated on a Siemens V94.3A gas turbine. Such data are representative of operating points characterized by different boundary, load and health state conditions. The analyses carried out are aimed at the selection of the most appropriate NN structure for gas turbine diagnostics, by evaluating NN robustness with respect to: • interpolation capability and accuracy in the presence of data affected by measurement errors; • extrapolation capability in the presence of data lying outside the range of variation adopted for NN training; • accuracy in the presence of input data corrupted by bias errors; • accuracy when one input is not available. This situation is simulated by replacing the value of the unavailable input with its nominal value. (@ISO24029)

Steven A Janowsky Pruning versus clipping in neural networks *Physical Review A*, 39 (12): 6600, 1989. **Abstract:** The number of interconnections in a neutral network is reduced by eliminating the ”weakest” bonds. The performance is then improved by reapplying the learning algorithm.Received 30 November 1988DOI:https://doi.org/10.1103/PhysRevA.39.6600©1989 American Physical Society (@janowsky1989pruning)

Jinyuan Jia, Xiaoyu Cao, Binghui Wang, and Neil Zhenqiang Gong Certified robustness for top-k predictions against adversarial perturbations via randomized smoothing In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=BkeWw6VFwr>. **Abstract:** It is well-known that classifiers are vulnerable to adversarial perturbations. To defend against adversarial perturbations, various certified robustness results have been derived. However, existing certified robustnesses are limited to top-1 predictions. In many real-world applications, top-$k$ predictions are more relevant. In this work, we aim to derive certified robustness for top-$k$ predictions. In particular, our certified robustness is based on randomized smoothing, which turns any classifier to a new classifier via adding noise to an input example. We adopt randomized smoothing because it is scalable to large-scale neural networks and applicable to any classifier. We derive a tight robustness in $\\}ell_2$ norm for top-$k$ predictions when using randomized smoothing with Gaussian noise. We find that generalizing the certified robustness from top-1 to top-$k$ predictions faces significant technical challenges. We also empirically evaluate our method on CIFAR10 and ImageNet. For example, our method can obtain an ImageNet classifier with a certified top-5 accuracy of 62.8\\}% when the $\\}ell_2$-norms of the adversarial perturbations are less than 0.5 (=127/255). Our code is publicly available at: \\}url{https://github.com/jjy1994/Certify_Topk}. (@jia2020certified)

Kenneth Johnson, Radu Calinescu, and Shinji Kikuchi An incremental verification framework for component-based software systems In *Proceedings of the 16th International ACM Sigsoft Symposium on Component-Based Software Engineering*, CBSE ’13, page 33–42, New York, NY, USA, 2013. Association for Computing Machinery. ISBN 9781450321228. . URL <https://doi.org/10.1145/2465449.2465456>. **Abstract:** We present a tool-supported framework for the efficient reverification of component-based software systems after changes such as additions, removals or modifications of components. The incremental verification engine at the core of our INcremental VErification STrategy (INVEST) framework uses high-level algebraic representations of component-based systems to identify and execute the minimal set of component-wise reverification steps after a system change. The generality of the INVEST engine allows its integration with existing assume-guarantee verification paradigms. We illustrate this integration for an existing technique for the assume-guarantee verification of probabilistic systems. The resulting instance of the INVEST framework can reverify probabilistic safety properties of a cloud-deployed software system in a fraction of the time required by compositional assume-guarantee verification alone. (@10.1145/2465449.2465456)

Kyle D. Julian, Mykel J. Kochenderfer, and Michael P. Owen Deep neural network compression for aircraft collision avoidance systems *CoRR*, abs/1810.04240, 2018. **Abstract:** One approach to designing decision making logic for an aircraft collision avoidance system frames the problem as a Markov decision process and optimizes the system using dynamic programming. The resulting collision avoidance strategy can be represented as a numeric table. This methodology has been used in the development of the Airborne Collision Avoidance System X (ACAS X) family of collision avoidance systems for manned and unmanned aircraft, but the high dimensionality of the state space leads to very large tables. To improve storage efficiency, a deep neural network is used to approximate the table. With the use of an asymmetric loss function and a gradient descent algorithm, the parameters for this network can be trained to provide accurate estimates of table values while preserving the relative preferences of the possible advisories for each state. By training multiple networks to represent subtables, the network also decreases the required runtime for computing the collision avoidance advisory. Simulation studies show that the network improves the safety and efficiency of the collision avoidance system. Because only the network parameters need to be stored, the required storage space is reduced by a factor of 1000, enabling the collision avoidance system to operate using current avionics systems. (@acasxu:18)

Guy Katz, Clark W. Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer Reluplex: An efficient SMT solver for verifying deep neural networks In *Computer Aided Verification - 29th International Conference, CAV 2017, Heidelberg, Germany, July 24-28, 2017, Proceedings, Part I*, volume 10426 of *Lecture Notes in Computer Science*, 2017. . **Abstract:** Deep neural networks have emerged as a widely used and e ective means for tackling complex, real-world problems. However, a major obstacle in applying them to safety-critical systems is the great dif- culty in providing formal guarantees about their behavior. We present a novel, scalable, and ecient technique for verifying properties of deep neural networks (or providing counter-examples). The technique is based on the simplex method, extended to handle the non-convex Recti ed Lin- ear Unit (ReLU ) activation function, which is a crucial ingredient in many modern neural networks. The veri cation procedure tackles neu- ral networks as a whole, without making any simplifying assumptions. We evaluated our technique on a prototype deep neural network imple- mentation of the next-generation airborne collision avoidance system for unmanned aircraft (ACAS Xu). Results show that our technique can successfully prove properties of networks that are an order of magnitude larger than the largest networks veri ed using existing methods. 1 Introduction Arti cial neural networks \[7,31\] have emerged as a promising approach for cre- ating scalable and robust systems. Applications include speech recognition \[9\], image classi cation \[22\], game playing \[32\], and many others. It is now clear that software that may be extremely dicult for humans to implement can instead be created by training deep neural networks (DNN s), and that the performance of these DNNs is often comparable to, or even surpasses, the performance of manually crafted software. DNNs are becoming widespread, and this trend is likely to continue and intensify. Great e ort is now being put into using DNNs as controllers for safety-critical systems such as autonomous vehicles \[4\] and airborne collision avoidance systems for unmanned aircraft (ACAS Xu) \[13\]. DNNs are trained over a nite set of in- puts and outputs and are expected to generalize , i.e. to behave correctly for previously-unseen inputs. However, it has been observed that DNNs can react in unexpected and incorrect ways to even slight perturbations of their inputs \[33\]. This unexpected behavior of DNNs is likely to result in unsafe systems, or re- strict the usage of DNNs in safety-critical applications. Hence, there is an urgent ?This is the extended version of a paper with the same title that appeared at CAV 2017.arXiv:1702.01135v2 \[cs.AI\] 19 May 2017need for methods that can provide formal guarantees about DNN behavior. Un- fortunately, manua (@DBLP:conf/cav/KatzBDJK17)

Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton Cifar-10 (canadian institute for advanced research) URL <http://www.cs.toronto.edu/~kriz/cifar.html>. **Abstract:** Image classification requires the generation of features capable of detecting image patterns informative of group identity. The objective of this study was to classify images from the public CIFAR10 image dataset by leveraging combinations of disparate image feature sources from deep learning approaches. The majority of regular convolutional neural networks (CNN) are based on the same structure: modification of convolution and the process of max-pooling layers connected with a number of entirely linked layers. In this paper, the prime objective is to improve the effectiveness of simple convolutional neural network models. The Artificial Neural Network (ANN) algorithm is applied on a Canadian Institute For Advanced Research dataset (CIFAR-10) using two different CNN structures. The result of the improved model achieves 88% classification accuracy rate by running for 10 hours. The deep learning models are implemented with the use of Keras library available for Python programming language. (@cifar10)

Aounon Kumar, Alexander Levine, Soheil Feizi, and Tom Goldstein Certifying confidence via randomized smoothing In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 5165–5177. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/37aa5dfc44dddd0d19d4311e2c7a0240-Paper.pdf>. **Abstract:** Randomized smoothing has been shown to provide good certified-robustness guarantees for high-dimensional classification problems. It uses the probabilities of predicting the top two most-likely classes around an input point under a smoothing distribution to generate a certified radius for a classifier’s prediction. However, most smoothing methods do not give us any information about the confidence with which the underlying classifier (e.g., deep neural network) makes a prediction. In this work, we propose a method to generate certified radii for the prediction confidence of the smoothed classifier. We consider two notions for quantifying confidence: average prediction score of a class and the margin by which the average prediction score of one class exceeds that of another. We modify the Neyman-Pearson lemma (a key theorem in randomized smoothing) to design a procedure for computing the certified radius where the confidence is guaranteed to stay above a certain threshold. Our experimental results on CIFAR-10 and ImageNet datasets show that using information about the distribution of the confidence scores allows us to achieve a significantly better certified radius than ignoring it. Thus, we demonstrate that extra information about the base classifier at the input point can help improve certified guarantees for the smoothed classifier. Code for the experiments is available at https://github.com/aounon/cdf-smoothing. (@NEURIPS2020_37aa5dfc)

Jacob Laurel, Rem Yang, Shubham Ugare, Robert Nagel, Gagandeep Singh, and Sasa Misailovic A general construction for abstract interpretation of higher-order automatic differentiation *Proc. ACM Program. Lang.*, 6 (OOPSLA2), oct 2022. . URL <https://doi.org/10.1145/3563324>. **Abstract:** We present a novel, general construction to abstractly interpret higher-order automatic differentiation (AD). Our construction allows one to instantiate an abstract interpreter for computing derivatives up to a chosen order. Furthermore, since our construction reduces the problem of abstractly reasoning about derivatives to abstractly reasoning about real-valued straight-line programs, it can be instantiated with almost any numerical abstract domain, both relational and non-relational. We formally establish the soundness of this construction. We implement our technique by instantiating our construction with both the non-relational interval domain and the relational zonotope domain to compute both first and higher-order derivatives. In the latter case, we are the first to apply a relational domain to automatic differentiation for abstracting higher-order derivatives, and hence we are also the first abstract interpretation work to track correlations across not only different variables, but different orders of derivatives. We evaluate these instantiations on multiple case studies, namely robustly explaining a neural network and more precisely computing a neural network’s Lipschitz constant. For robust interpretation, first and second derivatives computed via zonotope AD are up to 4.76× and 6.98× more precise, respectively, compared to interval AD. For Lipschitz certification, we obtain bounds that are up to 11,850× more precise with zonotopes, compared to the state-of-the-art interval-based tool. (@10.1145/3563324)

Jacob Laurel, Siyuan Brant Qian, Gagandeep Singh, and Sasa Misailovic Synthesizing precise static analyzers for automatic differentiation *Proc. ACM Program. Lang.*, 7 (OOPSLA2), oct 2023. . URL <https://doi.org/10.1145/3622867>. **Abstract:** We present Pasado, a technique for synthesizing precise static analyzers for Automatic Differentiation. Our technique allows one to automatically construct a static analyzer specialized for the Chain Rule, Product Rule, and Quotient Rule computations for Automatic Differentiation in a way that abstracts all of the nonlinear operations of each respective rule simultaneously. By directly synthesizing an abstract transformer for the composite expressions of these 3 most common rules of AD, we are able to obtain significant precision improvement compared to prior works which compose standard abstract transformers together suboptimally. We prove our synthesized static analyzers sound and additionally demonstrate the generality of our approach by instantiating these AD static analyzers with different nonlinear functions, different abstract domains (both intervals and zonotopes) and both forward-mode and reverse-mode AD. We evaluate Pasado on multiple case studies, namely soundly computing bounds on a neural network’s local Lipschitz constant, soundly bounding the sensitivities of financial models, certifying monotonicity, and lastly, bounding sensitivities of the solutions of differential equations from climate science and chemistry for verified ranges of initial conditions and parameters. The local Lipschitz constants computed by Pasado on our largest CNN are up to 2750× more precise compared to the existing state-of-the-art zonotope analysis. The bounds obtained on the sensitivities of the climate, chemical, and financial differential equation solutions are between 1.31 − 2.81× more precise (on average) compared to a state-of-the-art zonotope analysis. (@10.1145/3622867)

Guang-He Lee, Yang Yuan, Shiyu Chang, and Tommi Jaakkola Tight certificates of adversarial robustness for randomly smoothed classifiers In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc., 2019. URL <https://proceedings.neurips.cc/paper_files/paper/2019/file/fa2e8c4385712f9a1d24c363a2cbe5b8-Paper.pdf>. **Abstract:** Strong theoretical guarantees of robustness can be given for ensembles of classifiers generated by input randomization. Specifically, an $\\}ell_2$ bounded adversary cannot alter the ensemble prediction generated by an additive isotropic Gaussian noise, where the radius for the adversary depends on both the variance of the distribution as well as the ensemble margin at the point of interest. We build on and considerably expand this work across broad classes of distributions. In particular, we offer adversarial robustness guarantees and associated algorithms for the discrete case where the adversary is $\\}ell_0$ bounded. Moreover, we exemplify how the guarantees can be tightened with specific assumptions about the function class of the classifier such as a decision tree. We empirically illustrate these results with and without functional restrictions across image and molecule datasets. (@NEURIPS2019_fa2e8c43)

Alexander Levine and Soheil Feizi (de)randomized smoothing for certifiable defense against patch attacks In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL <https://proceedings.neurips.cc/paper/2020/hash/47ce0875420b2dbacfc5535f94e68433-Abstract.html>. **Abstract:** Patch adversarial attacks on images, in which the attacker can distort pixels within a region of bounded size, are an important threat model since they provide a quantitative model for physical adversarial attacks. In this paper, we introduce a certifiable defense against patch attacks that guarantees for a given image and patch attack size, no patch adversarial examples exist. Our method is related to the broad class of randomized smoothing robustness schemes which provide high-confidence probabilistic robustness certificates. By exploiting the fact that patch attacks are more constrained than general sparse attacks, we derive meaningfully large robustness certificates against them. Additionally, in contrast to smoothing-based defenses against L_p and sparse attacks, our defense method against patch attacks is de-randomized, yielding improved, deterministic certificates. Compared to the existing patch certification method proposed by Chiang et al. (2020), which relies on interval bound propagation, our method can be trained significantly faster, achieves high clean and certified robust accuracy on CIFAR-10, and provides certificates at ImageNet scale. For example, for a 5-by-5 patch attack on CIFAR-10, our method achieves up to around 57.6% certified accuracy (with a classifier with around 83.8% clean accuracy), compared to at most 30.3% certified accuracy for the existing method (with a classifier with around 47.8% clean accuracy). Our results effectively establish a new state-of-the-art of certifiable defense against patch attacks on CIFAR-10 and ImageNet. Code is available at https://github.com/alevine0/patchSmoothing. (@DBLP:conf/nips/0001F20a)

Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, and Bo Li Tss: Transformation-specific smoothing for robustness certification In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, CCS ’21, page 535–557, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384544. . URL <https://doi.org/10.1145/3460120.3485258>. **Abstract:** As machine learning (ML) systems become pervasive, safeguarding their security is critical. However, recently it has been demonstrated that motivated adversaries are able to mislead ML systems by perturbing test data using semantic transformations. While there exists a rich body of research providing provable robustness guarantees for ML models against Lp bounded adversarial perturbations, guarantees against semantic perturbations remain largely underexplored. In this paper, we provide TSS-a unified framework for certifying ML robustness against general adversarial semantic transformations. First, depending on the properties of each transformation, we divide common transformations into two categories, namely resolvable (e.g., Gaussian blur) and differentially resolvable (e.g., rotation) transformations. For the former, we propose transformation-specific randomized smoothing strategies and obtain strong robustness certification. The latter category covers transformations that involve interpolation errors, and we propose a novel approach based on stratified sampling to certify the robustness. Our framework TSS leverages these certification strategies and combines with consistency-enhanced training to provide rigorous certification of robustness. We conduct extensive experiments on over ten types of challenging semantic transformations and show that TSS significantly outperforms the state of the art. Moreover, to the best of our knowledge, TSS is the first approach that achieves nontrivial certified robustness on the large-scale ImageNet dataset. For instance, our framework achieves 30.4% certified robust accuracy against rotation attack (within ±30°) on ImageNet. Moreover, to consider a broader range of transformations, we show TSS is also robust against adaptive attacks and unforeseen image corruptions such as CIFAR-10-C and ImageNet-C. (@10.1145/3460120.3485258)

Hongbin Liu, Jinyuan Jia, and Neil Zhenqiang Gong Pointguard: Provably robust 3d point cloud classification 2021. **Abstract:** 3D point cloud classification has many safety-critical applications such as autonomous driving and robotic grasping. However, several studies showed that it is vulnerable to adversarial attacks. In particular, an attacker can make a classifier predict an incorrect label for a 3D point cloud via carefully modifying, adding, and/or deleting a small number of its points. Randomized smoothing is state-of-the-art technique to build certifiably robust 2D image classifiers. However, when applied to 3D point cloud classification, randomized smoothing can only certify robustness against adversarially modified points.In this work, we propose PointGuard, the first defense that has provable robustness guarantees against adversarially modified, added, and/or deleted points. Specifically, given a 3D point cloud and an arbitrary point cloud classifier, our PointGuard first creates multiple subsampled point clouds, each of which contains a random subset of the points in the original point cloud; then our PointGuard predicts the label of the original point cloud as the majority vote among the labels of the subsampled point clouds predicted by the point cloud classifier. Our first major theoretical contribution is that we show PointGuard provably predicts the same label for a 3D point cloud when the number of adversarially modified, added, and/or deleted points is bounded. Our second major theoretical contribution is that we prove the tightness of our derived bound when no assumptions on the point cloud classifier are made. Moreover, we design an efficient algorithm to compute our certified robustness guarantees. We also empirically evaluate PointGuard on ModelNet40 and ScanNet benchmark datasets. (@liu2021pointguard)

Jeet Mohapatra, Ching-Yun Ko, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, and Luca Daniel Higher-order certification for randomized smoothing 2020. **Abstract:** Randomized smoothing is a recently proposed defense against adversarial attacks that has achieved SOTA provable robustness against $\\}ell_2$ perturbations. A number of publications have extended the guarantees to other metrics, such as $\\}ell_1$ or $\\}ell\_\\}infty$, by using different smoothing measures. Although the current framework has been shown to yield near-optimal $\\}ell_p$ radii, the total safety region certified by the current framework can be arbitrarily small compared to the optimal. In this work, we propose a framework to improve the certified safety region for these smoothed classifiers without changing the underlying smoothing scheme. The theoretical contributions are as follows: 1) We generalize the certification for randomized smoothing by reformulating certified radius calculation as a nested optimization problem over a class of functions. 2) We provide a method to calculate the certified safety region using $0^{th}$-order and $1^{st}$-order information for Gaussian-smoothed classifiers. We also provide a framework that generalizes the calculation for certification using higher-order information. 3) We design efficient, high-confidence estimators for the relevant statistics of the first-order information. Combining the theoretical contribution 2) and 3) allows us to certify safety region that are significantly larger than the ones provided by the current methods. On CIFAR10 and Imagenet datasets, the new regions certified by our approach achieve significant improvements on general $\\}ell_1$ certified radii and on the $\\}ell_2$ certified radii for color-space attacks ($\\}ell_2$ restricted to 1 channel) while also achieving smaller improvements on the general $\\}ell_2$ certified radii. Our framework can also provide a way to circumvent the current impossibility results on achieving higher magnitude of certified radii without requiring the use of data-dependent smoothing techniques. (@mohapatra2020higherorder)

Peter W. O’Hearn Continuous reasoning: Scaling the impact of formal methods In Anuj Dawar and Erich Grädel, editors, *Proceedings of the 33rd Annual ACM/IEEE Symposium on Logic in Computer Science, LICS 2018, Oxford, UK, July 09-12, 2018*, pages 13–25. ACM, 2018. . URL <https://doi.org/10.1145/3209108.3209109>. **Abstract:** This paper describes work in continuous reasoning, where formal reasoning about a (changing) codebase is done in a fashion which mirrors the iterative, continuous model of software development that is increasingly practiced in industry. We suggest that advances in continuous reasoning will allow formal reasoning to scale to more programs, and more programmers. The paper describes the rationale for continuous reasoning, outlines some success cases from within industry, and proposes directions for work by the scientific community. (@DBLP:conf/lics/OHearn18)

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala Pytorch: An imperative style, high-performance deep learning library In *Advances in Neural Information Processing Systems 32*, pages 8024–8035. Curran Associates, Inc., 2019. URL <http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf>. **Abstract:** Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs. In this paper, we detail the principles that drove the implementation of PyTorch and how they are reflected in its architecture. We emphasize that every aspect of PyTorch is a regular Python program under the full control of its user. We also explain how the careful and pragmatic implementation of the key components of its runtime enables them to work together to achieve compelling performance. We demonstrate the efficiency of individual subsystems, as well as the overall speed of PyTorch on several common benchmarks. (@NEURIPS2019_9015)

PyTorch Torch quantization support https://github.com/pytorch/pytorch/issues/87395. **Abstract:** Anomaly detection plays an important role in industry, especially in ensuring system safety and product quality. Due to the unavailability of anomalous data in many practical cases, anomaly detection is usually solved by one-class classification (OCC) methods using only normal data. As a classical OCC method, one-class support vector machine (OCSVM) is a popular discriminative approach for anomaly detection, which detects abnormal data points by establishing a decision boundary in the kernel space. However, the performance of OCSVM heavily relies on kernel parameters, whose selection is not trivial for anomaly detection problems. Moreover, for some uneven and complex data distributions, different data regions may have quite different densities and shapes, making it difficult for OCSVM to obtain good boundaries in all regions using a global kernel parameter. To address the above two issues, in this article, we propose a hybrid algorithm incorporating vector quantization and OCSVM (VQ-OCSVM). Specifically, vector quantization is used to extract distribution information of normal data, and the results are used to construct an explicit mapping function to map data into a high-dimensional feature space. Then, OCSVM is performed in the feature space to build a classifier. By introducing the explicit mapping into OCSVM, the proposed method can effectively bypass the kernel parameter selection problem of the classical OCSVM method. Furthermore, the constructed mapping carries the data distribution information, and the VQ-OCSVM model can be regarded as an integration of generative learning and discriminative learning. The complementary properties of these two paradigms make the proposed VQ-OCSVM algorithm have better generalization capacity for complex data distribution. Both qualitative and quantitative experimental results demonstrate the effectiveness and advantages of the proposed method. (@torch_issue)

Russell Reed Pruning algorithms-a survey *IEEE transactions on Neural Networks*, 4 (5): 740–747, 1993. **Abstract:** A rule of thumb for obtaining good generalization in systems trained by examples is that one should use the smallest system that will fit the data. Unfortunately, it usually is not obvious what size is best; a system that is too small will not be able to learn the data while one that is just big enough may learn very slowly and be very sensitive to initial conditions and learning parameters. This paper is a survey of neural network pruning algorithms. The approach taken by the methods described here is to train a network that is larger than necessary and then remove the parts that are not needed. (@reed1993pruning)

Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, and J. Zico Kolter Certified robustness to label-flipping attacks via randomized smoothing 2020. **Abstract:** Machine learning algorithms are known to be susceptible to data poisoning attacks, where an adversary manipulates the training data to degrade performance of the resulting classifier. In this work, we present a unifying view of randomized smoothing over arbitrary functions, and we leverage this novel characterization to propose a new strategy for building classifiers that are pointwise-certifiably robust to general data poisoning attacks. As a specific instantiation, we utilize our framework to build linear classifiers that are robust to a strong variant of label flipping, where each test example is targeted independently. In other words, for each test point, our classifier includes a certification that its prediction would be the same had some number of training labels been changed adversarially. Randomized smoothing has previously been used to guarantee—with high probability—test-time robustness to adversarial manipulation of the input to a classifier; we derive a variant which provides a deterministic, analytical bound, sidestepping the probabilistic certificates that traditionally result from the sampling subprocedure. Further, we obtain these certified bounds with minimal additional runtime complexity over standard classification and no assumptions on the train or test distributions. We generalize our results to the multi-class case, providing the first multi-class classification algorithm that is certifiably robust to label-flipping attacks. (@rosenfeld2020certified)

Jan Schuchardt, Aleksandar Bojchevski, Johannes Gasteiger, and Stephan Günnemann Collective robustness certificates: Exploiting interdependence in graph neural networks In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=ULQdiUTHe3y>. **Abstract:** In tasks like node classification, image segmentation, and named-entity recognition we have a classifier that simultaneously outputs multiple predictions (a vector of labels) based on a single input, i.e. a single graph, image, or document respectively. Existing adversarial robustness certificates consider each prediction independently and are thus overly pessimistic for such tasks. They implicitly assume that an adversary can use different perturbed inputs to attack different predictions, ignoring the fact that we have a single shared input. We propose the first collective robustness certificate which computes the number of predictions that are simultaneously guaranteed to remain stable under perturbation, i.e. cannot be attacked. We focus on Graph Neural Networks and leverage their locality property - perturbations only affect the predictions in a close neighborhood - to fuse multiple single-node certificates into a drastically stronger collective certificate. For example, on the Citeseer dataset our collective certificate for node classification increases the average number of certifiable feature perturbations from $7$ to $351$. (@schuchardt2021collective)

Hashim Sharif, Prakalp Srivastava, Muhammad Huzaifa, Maria Kotsifakou, Keyur Joshi, Yasmin Sarita, Nathan Zhao, Vikram S. Adve, Sasa Misailovic, and Sarita Adve Approxhpvm: A portable compiler ir for accuracy-aware optimizations *Proc. ACM Program. Lang.*, 3 (OOPSLA), oct 2019. . URL <https://doi.org/10.1145/3360612>. **Abstract:** We propose ApproxHPVM, a compiler IR and system designed to enable accuracy-aware performance and energy tuning on heterogeneous systems with multiple compute units and approximation methods. ApproxHPVM automatically translates end-to-end application-level quality metrics into accuracy requirements for individual operations. ApproxHPVM uses a hardware-agnostic accuracy-tuning phase to do this translation that provides greater portability across heterogeneous hardware platforms and enables future capabilities like accuracy-aware dynamic scheduling and design space exploration. ApproxHPVM incorporates three main components: (a) a compiler IR with hardware-agnostic approximation metrics, (b) a hardware-agnostic accuracy-tuning phase to identify error-tolerant computations, and (c) an accuracy-aware hardware scheduler that maps error-tolerant computations to approximate hardware components. As ApproxHPVM does not incorporate any hardware-specific knowledge as part of the IR, it can serve as a portable virtual ISA that can be shipped to all kinds of hardware platforms. We evaluate our framework on nine benchmarks from the deep learning domain and five image processing benchmarks. Our results show that our framework can offload chunks of approximable computations to special-purpose accelerators that provide significant gains in performance and energy, while staying within user-specified application-level quality metrics with high probability. Across the 14 benchmarks, we observe from 1-9x performance speedups and 1.1-11.3x energy reduction for very small reductions in accuracy. (@10.1145/3360612)

Benno Stein, Bor-Yuh Evan Chang, and Manu Sridharan Demanded abstract interpretation In Stephen N. Freund and Eran Yahav, editors, *PLDI ’21: 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation, Virtual Event, Canada, June 20-25, 2021*, pages 282–295. ACM, 2021. . URL <https://doi.org/10.1145/3453483.3454044>. **Abstract:** We consider the problem of making expressive static analyzers interactive. Formal static analysis is seeing increasingly widespread adoption as a tool for verification and bug-finding, but even with powerful cloud infrastructure it can take minutes or hours to get batch analysis results after a code change. While existing techniques offer some demand-driven or incremental aspects for certain classes of analysis, the fundamental challenge we tackle is doing both for arbitrary abstract interpreters. Our technique, demanded abstract interpretation, lifts program syntax and analysis state to a dynamically evolving graph structure, in which program edits, client-issued queries, and evaluation of abstract semantics are all treated uniformly. The key difficulty addressed by our approach is the application of general incremental computation techniques to the complex, cyclic dependency structure induced by abstract interpretation of loops with widening operators. We prove that desirable abstract interpretation meta-properties, including soundness and termination, are preserved in our approach, and that demanded analysis results are equal to those computed by a batch abstract interpretation. Experimental results suggest promise for a prototype demanded abstract interpretation framework: by combining incremental and demand-driven techniques, our framework consistently delivers analysis results at interactive speeds, answering 95% of queries within 1.2 seconds. (@DBLP:conf/pldi/0002CS21)

Måns Thulin The cost of using exact confidence intervals for a binomial proportion *Electronic Journal of Statistics*, 8, 03 2013. . **Abstract:** When computing a confidence interval for a binomial proportion $p$ one must choose between using an exact interval, which has a coverage probability of at least $1-\\}alpha$ for all values of $p$, and a shorter approximate interval, which may have lower coverage for some $p$ but that on average has coverage equal to $1-\\}alpha$. We investigate the cost of using the exact one and two-sided Clopper–Pearson confidence intervals rather than shorter approximate intervals, first in terms of increased expected length and then in terms of the increase in sample size required to obtain a desired expected length. Using asymptotic expansions, we also give a closed-form formula for determining the sample size for the exact Clopper–Pearson methods. For two-sided intervals, our investigation reveals an interesting connection between the frequentist Clopper–Pearson interval and Bayesian intervals based on noninformative priors. (@stat)

Vincent Tjeng, Kai Xiao, and Russ Tedrake Evaluating robustness of neural networks with mixed integer programming *arXiv preprint arXiv:1711.07356*, 2017. **Abstract:** Neural networks have demonstrated considerable success on a wide variety of real-world problems. However, networks trained only to optimize for training accuracy can often be fooled by adversarial examples - slightly perturbed inputs that are misclassified with high confidence. Verification of networks enables us to gauge their vulnerability to such adversarial examples. We formulate verification of piecewise-linear neural networks as a mixed integer program. On a representative task of finding minimum adversarial distortions, our verifier is two to three orders of magnitude quicker than the state-of-the-art. We achieve this computational speedup via tight formulations for non-linearities, as well as a novel presolve algorithm that makes full use of all information available. The computational speedup allows us to verify properties on convolutional networks with an order of magnitude more ReLUs than networks previously verified by any complete verifier. In particular, we determine for the first time the exact adversarial accuracy of an MNIST classifier to perturbations with bounded $l\_\\}infty$ norm $\\}epsilon=0.1$: for this classifier, we find an adversarial example for 4.38% of samples, and a certificate of robustness (to perturbations with bounded norm) for the remainder. Across all robust training procedures and network architectures considered, we are able to certify more samples than the state-of-the-art and find more adversarial examples than a strong first-order attack. (@tjeng2017evaluating)

Shubham Ugare, Debangshu Banerjee, Tarun Suresh, Sasa Misailovic, and Gagandeep Singh Toward continuous verification of dnns **Abstract:** Generative Adversarial Networks (GANs) are powerful generative models for numerous tasks and datasets. However, most of the existing models suffer from mode collapse. The most recent research indicates that the reason for it is that the optimal transportation map from random noise to the data distribution is discontinuous, but deep neural networks (DNNs) can only approximate continuous ones. Instead, the latent representation is a better raw material used to construct a transportation map point to the data distribution than random noise. Because it is a low-dimensional mapping related to the data distribution, the construction procedure seems more like expansion rather than starting all over. Besides, we can also search for more transportation maps in this way with smoother transformation. Thus, we have proposed a new training methodology for GANs in this paper to search for more transportation maps and speed the training up, named Express Construction. The key idea is to train GANs with two independent phases for successively yielding latent representation and data distribution. To this end, an Auto-Encoder is trained to map the real data into the latent space, and two couples of generators and discriminators are used to produce them. To the best of our knowledge, we are the first to decompose the training procedure of GAN models into two more uncomplicated phases, thus tackling the mode collapse problem without much more computational cost. We also provide theoretical steps toward understanding the training dynamics of this procedure and prove assumptions. No extra hyper-parameters have been used in the proposed method, which indicates that Express Construction can be used to train any GAN models. Extensive experiments are conducted to verify the performance of realistic image generation and the resistance to mode collapse. The results show that the proposed method is lightweight, effective, and less prone to mode collapse. (@ugaretoward)

Shubham Ugare, Gagandeep Singh, and Sasa Misailovic Proof transfer for fast certification of multiple approximate neural networks *Proc. ACM Program. Lang.*, 6 (OOPSLA): 1–29, 2022. . URL <https://doi.org/10.1145/3527319>. **Abstract:** Developers of machine learning applications often apply post-training neural network optimizations, such as quantization and pruning, that approximate a neural network to speed up inference and reduce energy consumption, while maintaining high accuracy and robustness. Despite a recent surge in techniques for the robustness verification of neural networks, a major limitation of almost all state-of-the-art approaches is that the verification needs to be run from scratch every time the network is even slightly modified. Running precise end-to-end verification from scratch for every new network is expensive and impractical in many scenarios that use or compare multiple approximate network versions, and the robustness of all the networks needs to be verified efficiently. We present FANC, the first general technique for transferring proofs between a given network and its multiple approximate versions without compromising verifier precision. To reuse the proofs obtained when verifying the original network, FANC generates a set of templates – connected symbolic shapes at intermediate layers of the original network – that capture the proof of the property to be verified. We present novel algorithms for generating and transforming templates that generalize to a broad range of approximate networks and reduce the verification cost. We present a comprehensive evaluation demonstrating the effectiveness of our approach. We consider a diverse set of networks obtained by applying popular approximation techniques such as quantization and pruning on fully-connected and convolutional architectures and verify their robustness against different adversarial attacks such as adversarial patches, L 0 , rotation and brightening. Our results indicate that FANC can significantly speed up verification with state-of-the-art verifier, DeepZ by up to 4.1x. (@DBLP:journals/pacmpl/UgareSM22)

Shubham Ugare, Debangshu Banerjee, Sasa Misailovic, and Gagandeep Singh Incremental verification of neural networks *Proc. ACM Program. Lang.*, 7 (PLDI), jun 2023. . URL <https://doi.org/10.1145/3591299>. **Abstract:** Complete verification of deep neural networks (DNNs) can exactly determine whether the DNN satisfies a desired trustworthy property (e.g., robustness, fairness) on an infinite set of inputs or not. Despite the tremendous progress to improve the scalability of complete verifiers over the years on individual DNNs, they are inherently inefficient when a deployed DNN is updated to improve its inference speed or accuracy. The inefficiency is because the expensive verifier needs to be run from scratch on the updated DNN. To improve efficiency, we propose a new, general framework for incremental and complete DNN verification based on the design of novel theory, data structure, and algorithms. Our contributions implemented in a tool named IVAN yield an overall geometric mean speedup of 2.4x for verifying challenging MNIST and CIFAR10 classifiers and a geometric mean speedup of 3.8x for the ACAS-XU classifiers over the state-of-the-art baselines. (@ugare2023incremental)

Willem Visser, Jaco Geldenhuys, and Matthew B. Dwyer Green: Reducing, reusing and recycling constraints in program analysis In *Proceedings of the ACM SIGSOFT 20th International Symposium on the Foundations of Software Engineering*, FSE ’12, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450316149. . URL <https://doi.org/10.1145/2393596.2393665>. **Abstract:** Despite the crucial role that businesses have played in the adoption of green supply chains, it appears that manufacturing enterprises have not effectively utilized their responsibility to ensure green manufacturing. This study uses Bakhresa Foods Product Limited as a case study to better understand the constraints facing manufacturing firms engaged in green manufacturing and offers suggestions for overcoming them. A qualitative design was adopted and data was gathered via in-depth interviews, focus group discussions (FGD), indirect observation, and secondary data analysis based on a purposive sample size of 60 respondents. Using the MAXQDA 10 program, data were analyzed using the thematic analysis technique. The study’s findings showed that constraints to reducing were associated with a lack of innovation, professionals, an understanding of economics and finance and presence of extensive inventories, and inadequate production planning. Remanufacturing was further constrained by poor technology, unfavorable consumer attitudes, and a weak market strategy. Other constraints with reuse came from poor customer attitudes and awareness, lack of uniform standards for materials and goods, and inadequate government support. Constraints to recycling include a lack of support from senior top management executives, significant upfront expenses, and low-quality recycled goods. Therefore, to improve the situation, we suggest that industries make use of modernized equipment, and clean technology, improve monitoring and evaluation systems, implement the lean production philosophy, and give education about the value of recycling programs. (@10.1145/2393596.2393665)

Binghui Wang, Jinyuan Jia, Xiaoyu Cao, and Neil Zhenqiang Gong Certified robustness of graph neural networks against adversarial structural perturbation In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, KDD ’21, page 1645–1653, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383325. . URL <https://doi.org/10.1145/3447548.3467295>. **Abstract:** Graph neural networks (GNNs) have recently gained much attention for node and graph classification tasks on graph-structured data. However, multiple recent works showed that an attacker can easily make GNNs predict incorrectly via perturbing the graph structure, i.e., adding or deleting edges in the graph. We aim to defend against such attacks via developing certifiably robust GNNs. Specifically, we prove the first certified robustness guarantee of any GNN for both node and graph classifications against structural perturbation. Moreover, we show that our certified robustness guarantee is tight. Our results are based on a recently proposed technique called randomized smoothing, which we extend to graph data. We also empirically evaluate our method for both node and graph classifications on multiple GNNs and multiple benchmark datasets. For instance, on the Cora dataset, Graph Convolutional Network with our randomized smoothing can achieve a certified accuracy of 0.49 when the attacker can arbitrarily add/delete at most 15 edges in the graph. (@10.1145/3447548.3467295)

Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, and J Zico Kolter Beta-crown: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification *arXiv preprint arXiv:2103.06624*, 2021. **Abstract:** Bound propagation based incomplete neural network verifiers such as CROWN are very efficient and can significantly accelerate branch-and-bound (BaB) based complete verification of neural networks. However, bound propagation cannot fully handle the neuron split constraints introduced by BaB commonly handled by expensive linear programming (LP) solvers, leading to loose bounds and hurting verification efficiency. In this work, we develop $\\}beta$-CROWN, a new bound propagation based method that can fully encode neuron splits via optimizable parameters $\\}beta$ constructed from either primal or dual space. When jointly optimized in intermediate layers, $\\}beta$-CROWN generally produces better bounds than typical LP verifiers with neuron split constraints, while being as efficient and parallelizable as CROWN on GPUs. Applied to complete robustness verification benchmarks, $\\}beta$-CROWN with BaB is up to three orders of magnitude faster than LP-based BaB methods, and is notably faster than all existing approaches while producing lower timeout rates. By terminating BaB early, our method can also be used for efficient incomplete verification. We consistently achieve higher verified accuracy in many settings compared to powerful incomplete verifiers, including those based on convex barrier breaking techniques. Compared to the typically tightest but very costly semidefinite programming (SDP) based incomplete verifiers, we obtain higher verified accuracy with three orders of magnitudes less verification time. Our algorithm empowered the $\\}alpha,\\}!\\}beta$-CROWN (alpha-beta-CROWN) verifier, the winning tool in VNN-COMP 2021. Our code is available at http://PaperCode.cc/BetaCROWN (@wang2021beta)

Tianhao Wei and Changliu Liu Online verification of deep neural networks under domain or weight shift *CoRR*, abs/2106.12732, 2021. URL <https://arxiv.org/abs/2106.12732>. **Abstract:** Although neural networks are widely used, it remains challenging to formally verify the safety and robustness of neural networks in real-world applications. Existing methods are designed to verify the network before deployment, which are limited to relatively simple specifications and fixed networks. These methods are not ready to be applied to real-world problems with complex and/or dynamically changing specifications and networks. To effectively handle such problems, verification needs to be performed online when these changes take place. However, it is still challenging to run existing verification algorithms online. Our key insight is that we can leverage the temporal dependencies of these changes to accelerate the verification process. This paper establishes a novel framework for scalable online verification to solve real-world verification problems with dynamically changing specifications and/or networks. We propose three types of acceleration algorithms: Branch Management to reduce repetitive computation, Perturbation Tolerance to tolerate changes, and Incremental Computation to reuse previous results. Experiment results show that our algorithms achieve up to $100\\}times$ acceleration, and thus show a promising way to extend neural network verification to real-world applications. (@DBLP:journals/corr/abs-2106-12732)

Edwin B. Wilson Probable inference, the law of succession, and statistical inference *Journal of the American Statistical Association*, 22 (158): 209–212, 1927. ISSN 01621459. URL <http://www.jstor.org/stable/2276774>. **Abstract:** (1927). Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association: Vol. 22, No. 158, pp. 209-212. (@10.2307/2276774)

Greg Yang, Tony Duan, J. Edward Hu, Hadi Salman, Ilya Razenshteyn, and Jerry Li Randomized smoothing of all shapes and sizes 2020. **Abstract:** Randomized smoothing is a recently proposed defense against adversarial attacks that has achieved state-of-the-art provable robustness against L2 perturbations. Soon after, a number of works devised new randomized smoothing schemes for other metrics, such as L1 or L-infinity; however, for each geometry, substantial effort was needed to derive new robustness guarantees. This begs the question: can we find a general theory for randomized smoothing? In this work we propose a novel framework for devising and analyzing randomized smoothing schemes, and validate its effectiveness in practice. Our theoretical contributions are as follows: (1) We show that for an appropriate notion of “optimal”, the optimal smoothing distributions for any “nice” norm have level sets given by the Wulff Crystal of that norm. (2) We propose two novel and complementary methods for deriving provably robust radii for any smoothing distribution. Finally, (3) we show fundamental limits to current randomized smoothing techniques via the theory of Banach space cotypes. By combining (1) and (2), we significantly improve the state-of-the-art certified accuracy in L1 on standard datasets. On the other hand, using (3), we show that, without more information than label statistics under random input perturbations, randomized smoothing cannot achieve nontrivial certified accuracy against perturbations of L-infinity-norm Omega(1/sqrt(d)), when the input dimension d is large. We provide code in github.com/tonyduan/rs4a. (@yang2020randomized)

Guowei Yang, Matthew B. Dwyer, and Gregg Rothermel Regression model checking In *2009 IEEE International Conference on Software Maintenance*, pages 115–124, 2009. . **Abstract:** Model checking is a promising technique for verifying program behavior and is increasingly finding usage in industry. To date, however, researchers have primarily considered model checking of single versions of programs. It is well understood that model checking can be very expensive for large, complex programs. Thus, simply reapplying model checking techniques on subsequent versions of programs as they evolve, in the limited time that is typically available for validating new releases, presents challenges. To address these challenges, we have developed a new technique for regression model checking (RMC), that applies model checking incrementally to new versions of systems. We report results of an empirical study examining the effectiveness of our technique; our results show that it is significantly faster than traditional model checking. (@5306334)

Ping yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, and Tom Goldstein Detection as regression: Certified object detection by median smoothing 2022. **Abstract:** Despite the vulnerability of object detectors to adversarial attacks, very few defenses are known to date. While adversarial training can improve the empirical robustness of image classifiers, a direct extension to object detection is very expensive. This work is motivated by recent progress on certified classification by randomized smoothing. We start by presenting a reduction from object detection to a regression problem. Then, to enable certified regression, where standard mean smoothing fails, we propose median smoothing, which is of independent interest. We obtain the first model-agnostic, training-free, and certified defense for object detection against $\\}ell_2$-bounded attacks. The code for all experiments in the paper is available at http://github.com/Ping-C/CertifiedObjectDetection . (@chiang2022detection)

Dinghuai Zhang, Mao Ye, Chengyue Gong, Zhanxing Zhu, and Qiang Liu Black-box certification with randomized smoothing: A functional optimization based framework In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 2316–2326. Curran Associates, Inc., 2020. URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/1896a3bf730516dd643ba67b4c447d36-Paper.pdf>. **Abstract:** Randomized classifiers have been shown to provide a promising approach for achieving certified robustness against adversarial attacks in deep learning. However, most existing methods only leverage Gaussian smoothing noise and only work for $\\}ell_2$ perturbation. We propose a general framework of adversarial certification with non-Gaussian noise and for more general types of attacks, from a unified functional optimization perspective. Our new framework allows us to identify a key trade-off between accuracy and robustness via designing smoothing distributions, helping to design new families of non-Gaussian smoothing distributions that work more efficiently for different $\\}ell_p$ settings, including $\\}ell_1$, $\\}ell_2$ and $\\}ell\_\\}infty$ attacks. Our proposed methods achieve better certification results than previous works and provide a new perspective on randomized smoothing certification. (@NEURIPS2020_1896a3bf)

Yifan Zhao, Hashim Sharif, Peter Pao-Huang, Vatsin Shah, Arun Narenthiran Sivakumar, Mateus Valverde Gasparino, Abdulrahman Mahmoud, Nathan Zhao, Sarita Adve, Girish Chowdhary, et al Approxcaliper: A programmable framework for application-aware neural network optimization *Proceedings of Machine Learning and Systems*, 5, 2023. **Abstract:** Cloud computing is a widely adopted platform for executing tasks of different application types that belong to the end users. In the cloud, application task is prone to failure for several reasons, such as software bug or exception, virtual or physical infrastructure failure. Cloud service providers are responsible for managing availability of scheduled computing tasks in order to provide high level QoS for their customers. Protecting task against failure is a challenging and not a trivial mission due to dynamic, heterogeneous and large distributed structure of the cloud environment. The existing works in the literature focus on task failure prediction and neglect the remedy (post) actions. In this work, we first study and analyze three publicly available large cluster datasets from Google, Alibaba, and Trinity, to characterize task failure in cloud computing platform. We then propose a failure-aware task scheduling framework that can predict the termination status for a set of given tasks during the runtime, and take the appropriate remedy actions. The framework uses deep learning methods named Artificial and Convolutional Neural Network, ANN and CNN, for different prediction purposes. In addition, we formalize the actions selection problem as Integer Linear Programming (ILP) model and propose a heuristic optimization solution that aims to minimize the failure probability of tasks and their resources usage. The results show ANN and CNN can achieve prediction accuracy of up to 94% and 92%, respectively using Google dataset. Moreover, the framework can protect up to 40% of tasks that are predicted as failed using Alibaba dataset by taking the appropriate remedy actions, and hence save many of cluster’s resources such as CPU and RAM. (@zhao2023approxcaliper)

Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, and Yurong Chen Incremental network quantization: Towards lossless CNNs with low-precision weights In *International Conference on Learning Representations*, 2017. URL <https://openreview.net/forum?id=HyQJ-mclg>. **Abstract:** This paper presents incremental network quantization (INQ), a novel method, targeting to efficiently convert any pre-trained full-precision convolutional neural network (CNN) model into a low-precision version whose weights are constrained to be either powers of two or zero. Unlike existing methods which are struggled in noticeable accuracy loss, our INQ has the potential to resolve this issue, as benefiting from two innovations. On one hand, we introduce three interdependent operations, namely weight partition, group-wise quantization and re-training. A well-proven measure is employed to divide the weights in each layer of a pre-trained CNN model into two disjoint groups. The weights in the first group are responsible to form a low-precision base, thus they are quantized by a variable-length encoding method. The weights in the other group are responsible to compensate for the accuracy loss from the quantization, thus they are the ones to be re-trained. On the other hand, these three operations are repeated on the latest re-trained group in an iterative manner until all the weights are converted into low-precision ones, acting as an incremental network quantization and accuracy enhancement procedure. Extensive experiments on the ImageNet classification task using almost all known deep CNN architectures including AlexNet, VGG-16, GoogleNet and ResNets well testify the efficacy of the proposed method. Specifically, at 5-bit quantization, our models have improved accuracy than the 32-bit floating-point references. Taking ResNet-18 as an example, we further show that our quantized models with 4-bit, 3-bit and 2-bit ternary weights have improved or very similar accuracy against its 32-bit floating-point baseline. Besides, impressive results with the combination of network pruning and INQ are also reported. The code is available at https://github.com/Zhouaojun/Incremental-Network-Quantization. (@zhou2017incremental)

</div>

# Appendix

## Observation for Binomial Confidence Interval Methods

In this section, we show the plots for the number of samples required to estimate an unknown binomial proportion parameter through two popular estimation techniques - the Wilson   and Agresti-Coull method  . For this experiment, we use three different values of the target error $`\chi`$ = 0.5 %, 0.75 %, and 1.0 % and a fixed confidence value $`(1 - \alpha) = 0.99`$ for both estimation methods. As shown in Fig <a href="#fig:additionSampleSizeapp" data-reference-type="ref" data-reference="fig:additionSampleSizeapp">14</a>, for a fixed target error $`\chi`$, confidence $`(1 - \alpha)`$, and estimation technique, the number of samples required for estimation peaks, when the actual parameter value is around $`0.5`$ and is the smallest around the boundaries. This is consistent with the observation described in Section <a href="#sec:motivation" data-reference-type="ref" data-reference="sec:motivation">3.1</a>.

<figure id="fig:additionSampleSizeapp">
<figure id="fig:additionSampleSizeappAgesti">
<img src="./figures/plot_binomial_proportion_agresti_coull.png"" />
<figcaption>Agresti-Coull method</figcaption>
</figure>
<figure id="fig:additionSampleSizeappWilson">
<img src="./figures/plot_binomial_proportion_wilson.png"" />
<figcaption>Wilson method</figcaption>
</figure>
<figcaption>The number of samples for the Agresti-Coull and Wilson method to achieve a target error <span class="math inline"><em>χ</em></span> with confidence <span class="math inline">(1 − <em>α</em>)</span> where <span class="math inline"><em>α</em> = 0.01</span>. The plots show that the number of required samples for different methods peaks at 0.5 and decreases towards the boundaries.</figcaption>
</figure>

## Theorems

<div class="proof">

*Proof.* If $`f(x+\epsilon) = c_A`$ and $`f^p(x+\epsilon) = f(x+\epsilon)`$ then $`f^p(x+\epsilon) = c_A`$.  
Thus, if $`f^p(x+\epsilon) \neq c_A`$ then $`f(x+\epsilon) \neq c_A`$ or $`f^p(x+\epsilon) \neq f(x+\epsilon)`$.  
Using union bound,
``` math
\mathbb{P}_\epsilon(f^p(x+\epsilon) \neq c_A) \leq \mathbb{P}_\epsilon(f(x+\epsilon) \neq c_A) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
(1 - \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A)) \leq (1-\mathbb{P}_\epsilon(f(x+\epsilon) = c_A)) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
\mathbb{P}_\epsilon(f(x+\epsilon) = c_A) \leq \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
\underline{p_A}- \zeta_{x}\leq \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A)
```
Similarly, if $`f(x+\epsilon) \neq c`$ then $`f^p(x+\epsilon) \neq c`$ or $`f^p(x+\epsilon) \neq f(x+\epsilon)`$.  
Hence, using union bound,
``` math
\mathbb{P}_\epsilon(f(x+\epsilon) \neq c) \leq \mathbb{P}_\epsilon(f^p(x+\epsilon) \neq c) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
(1 - \mathbb{P}_\epsilon(f(x+\epsilon) = c)) \leq (1-\mathbb{P}_\epsilon(f^p(x+\epsilon) = c)) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
\mathbb{P}_\epsilon(f^p(x+\epsilon) = c) \leq \mathbb{P}_\epsilon(f(x+\epsilon) = c) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
\max_{c \neq c_A} \mathbb{P}_\epsilon(f^p(x+\epsilon) = c) \leq \max_{c \neq c_A} \mathbb{P}_\epsilon(f(x+\epsilon) = c) + \zeta_{x}
```
``` math
\max_{c \neq c_A} \mathbb{P}_\epsilon(f^p(x+\epsilon) = c) \leq \overline{p_B}+ \zeta_{x}
```
Hence, using Theorem <a href="#thm:rs" data-reference-type="ref" data-reference="thm:rs">[thm:rs]</a>, $`g^p`$ satisfies $`g^p(x+\delta) = c_A`$ for all $`\delta`$ satisying $`\|\delta\|_2 \leq \frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}-\zeta_{x}) - \Phi^{-1}(\overline{p_B}+\zeta_{x}))`$ ◻

</div>

<div class="proof">

*Proof.* Since $`\underline{p_A}-\zeta_{x}\geq \frac{1}{2}`$, $`0 \leq \underline{p_A}\leq 1`$ and $`\zeta_{x}\geq 0`$, we get $`0 \leq \underline{p_A}-\zeta_{x}\leq 1`$

And since $`1-\underline{p_A}\geq \overline{p_B}`$, we get $`\overline{p_B}+\zeta_{x}\leq \frac{1}{2}`$, and thus, $`0 \leq \overline{p_B}+\zeta_{x}\leq 1`$

Since $`\Phi^{-1}(1-t) = -\Phi^{-1}(t)`$
``` math
\Phi^{-1}(\overline{p_B}+\zeta_{x}) = -\Phi^{-1}(1-(\overline{p_B}+\zeta_{x}))
```
``` math
= -\Phi^{-1}((1-\overline{p_B}) -\zeta_{x})
```
Since $`1-\underline{p_A}\geq \overline{p_B}`$
``` math
\leq -\Phi^{-1}(\underline{p_A}-\zeta_{x})
```
Hence,
``` math
\Phi^{-1}(\underline{p_A}-\zeta_{x}) \leq -\Phi^{-1}(\overline{p_B}+\zeta_{x})
```
``` math
\frac{\sigma}{2} \Phi^{-1}(\underline{p_A}-\zeta_{x}) \leq -\frac{\sigma}{2} \Phi^{-1}(\overline{p_B}+\zeta_{x})
```
Adding $`\frac{\sigma}{2} \Phi^{-1}(\underline{p_A}-\zeta_{x})`$ on both sides,
``` math
\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x}) \leq \frac{\sigma}{2} (\Phi^{-1}(\underline{p_A}-\zeta_{x}) -\Phi^{-1}(\overline{p_B}+\zeta_{x}))
```
 ◻

</div>

<div class="proof">

*Proof.* Suppose $`f`$ and $`f^p`$ are classifiers such that for a fixed $`x \in \mathbb{R}^m, \mathbb{P}_\epsilon (f(x+\epsilon) = c_A) \geq \underline{p_A}`$ and $`\mathbb{P}_\epsilon(f(x+\epsilon) = f^p(x+\epsilon)) > 1-\zeta_{x}`$. Note that this is true by the definition of $`\underline{p_A}`$, and is a separate $`\underline{p_A}`$ for each $`x`$. The statement is not true for all $`x`$ with single $`\underline{p_A}`$  
Let $`E_1`$ denote the event that $`\mathbb{P}_\epsilon (f(x+\epsilon) = c_A) \geq \underline{p_A}`$.  
Let $`E_2`$ denote the event that $`\mathbb{P}_\epsilon(f(x+\epsilon) = f^p(x+\epsilon)) > 1-\zeta_{x}`$.  
By Theorem <a href="#thm:irs" data-reference-type="ref" data-reference="thm:irs">[thm:irs]</a>,
``` math
\mathbb{P}_\epsilon(f(x+\epsilon) = c_A) \leq \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A) + \mathbb{P}_\epsilon(f(x+\epsilon) \neq f^p(x+\epsilon))
```
``` math
\underline{p_A}- \zeta_{x}\leq \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A)
```
  
Let $`E_3`$ denote the event that $`\underline{p_A}- \zeta_{x}\leq \mathbb{P}_\epsilon(f^p(x+\epsilon) = c_A)`$  
Since, $`E_1`$ and $`E_2`$ imply $`E_3`$ i.e. $`E_1 \cap E_2 \subseteq E_3`$,
``` math
\mathbb{P}(E_3) \geq \mathbb{P}(E_1 \cap E_2)
```
By the additive rule of probability,
``` math
\mathbb{P}(E_1 \cap E_2) = \mathbb{P}(E_1) + \mathbb{P}(E_2) - \mathbb{P}(E_1 \cup E_2)
```
``` math
\mathbb{P}(E_3) \geq (1 - \alpha) + (1 - \alpha_\zeta) - 1
```
``` math
\mathbb{P}(E_3) \geq 1 - (\alpha + \alpha_\zeta)
```
Hence, for classifier $`f^p`$, $`\mathbb{P}_\epsilon (f^p(x+\epsilon) = c_A) \geq \underline{p_A}-\zeta_{x}`$ has confidence at least $`1-(\alpha+\alpha_\zeta)`$ ◻

</div>

## Evaluation Networks

Table <a href="#tab:std_acc" data-reference-type="ref" data-reference="tab:std_acc">8</a> and Table <a href="#tab:smooth_acc" data-reference-type="ref" data-reference="tab:smooth_acc">9</a> respectively present the standard top-1 accuracy of the original and approximated base classifiers and smoothed classifiers respectively.

<div id="tab:std_acc">

<table>
<caption>Standard top-1 accuracy for (non-smoothed) networks for combinations of approximations and <span class="math inline"><em>σ</em></span>’s.</caption>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Architecture</td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td style="text-align: right;">original</td>
<td colspan="3" style="text-align: center;">Quantization</td>
<td colspan="3" style="text-align: center;">Prune</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
<td style="text-align: right;">5%</td>
<td style="text-align: right;">10%</td>
<td style="text-align: right;">20%</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">67.2</td>
<td style="text-align: right;">67.2</td>
<td style="text-align: right;">66.8</td>
<td style="text-align: right;">67.2</td>
<td style="text-align: right;">67.4</td>
<td style="text-align: right;">66.6</td>
<td style="text-align: right;">66.6</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-20</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">56.8</td>
<td style="text-align: right;">56.8</td>
<td style="text-align: right;">57.2</td>
<td style="text-align: right;">56.8</td>
<td style="text-align: right;">57</td>
<td style="text-align: right;">57.4</td>
<td style="text-align: right;">58</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">47.2</td>
<td style="text-align: right;">47.2</td>
<td style="text-align: right;">47.0</td>
<td style="text-align: right;">47.2</td>
<td style="text-align: right;">47</td>
<td style="text-align: right;">46.2</td>
<td style="text-align: right;">45.2</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">69.0</td>
<td style="text-align: right;">69.0</td>
<td style="text-align: right;">69.4</td>
<td style="text-align: right;">69.0</td>
<td style="text-align: right;">69.2</td>
<td style="text-align: right;">68.8</td>
<td style="text-align: right;">68.2</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-110</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">59.4</td>
<td style="text-align: right;">59.4</td>
<td style="text-align: right;">59.4</td>
<td style="text-align: right;">59.4</td>
<td style="text-align: right;">59.6</td>
<td style="text-align: right;">59</td>
<td style="text-align: right;">58.8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">47.0</td>
<td style="text-align: right;">47.0</td>
<td style="text-align: right;">46.8</td>
<td style="text-align: right;">46.8</td>
<td style="text-align: right;">46.8</td>
<td style="text-align: right;">47.2</td>
<td style="text-align: right;">47</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">24.2</td>
<td style="text-align: right;">24.2</td>
<td style="text-align: right;">24.4</td>
<td style="text-align: right;">24.2</td>
<td style="text-align: right;">24.2</td>
<td style="text-align: right;">24.4</td>
<td style="text-align: right;">24.2</td>
</tr>
<tr>
<td style="text-align: left;">ImageNet</td>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
<td style="text-align: right;">9.6</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
<td style="text-align: right;">6.4</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:std_acc" label="tab:std_acc"></span>

<div id="tab:smooth_acc">

<table>
<caption>standard top-1 accuracy for smoothed networks for combinations of approximations and <span class="math inline"><em>σ</em></span>’s.</caption>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Architecture</td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td style="text-align: right;">original</td>
<td colspan="3" style="text-align: center;">Quantization</td>
<td colspan="3" style="text-align: center;">Prune</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;"></td>
<td style="text-align: right;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
<td style="text-align: right;">5%</td>
<td style="text-align: right;">10%</td>
<td style="text-align: right;">20%</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">77.2</td>
<td style="text-align: right;">77</td>
<td style="text-align: right;">77.2</td>
<td style="text-align: right;">77.2</td>
<td style="text-align: right;">77.6</td>
<td style="text-align: right;">77.2</td>
<td style="text-align: right;">77.6</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-20</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">67.8</td>
<td style="text-align: right;">67.4</td>
<td style="text-align: right;">67.8</td>
<td style="text-align: right;">67.8</td>
<td style="text-align: right;">67.8</td>
<td style="text-align: right;">67.4</td>
<td style="text-align: right;">67.8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">55.6</td>
<td style="text-align: right;">55.6</td>
<td style="text-align: right;">55.6</td>
<td style="text-align: right;">55.8</td>
<td style="text-align: right;">54.8</td>
<td style="text-align: right;">55.2</td>
<td style="text-align: right;">55.6</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">76.6</td>
<td style="text-align: right;">76.4</td>
<td style="text-align: right;">76.2</td>
<td style="text-align: right;">76.4</td>
<td style="text-align: right;">76.2</td>
<td style="text-align: right;">76.2</td>
<td style="text-align: right;">76.4</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-110</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">66.2</td>
<td style="text-align: right;">67</td>
<td style="text-align: right;">68</td>
<td style="text-align: right;">66.4</td>
<td style="text-align: right;">67</td>
<td style="text-align: right;">66.8</td>
<td style="text-align: right;">66.6</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">55.6</td>
<td style="text-align: right;">55.4</td>
<td style="text-align: right;">56.2</td>
<td style="text-align: right;">56.2</td>
<td style="text-align: right;">55</td>
<td style="text-align: right;">55</td>
<td style="text-align: right;">54.8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">63.8</td>
<td style="text-align: right;">63.4</td>
<td style="text-align: right;">63.2</td>
<td style="text-align: right;">63.4</td>
<td style="text-align: right;">63.6</td>
<td style="text-align: right;">64</td>
<td style="text-align: right;">63</td>
</tr>
<tr>
<td style="text-align: left;">ImageNet</td>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">48.8</td>
<td style="text-align: right;">48.6</td>
<td style="text-align: right;">48.8</td>
<td style="text-align: right;">48.6</td>
<td style="text-align: right;">48.8</td>
<td style="text-align: right;">48.6</td>
<td style="text-align: right;">47.8</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;">34.4</td>
<td style="text-align: right;">34.2</td>
<td style="text-align: right;">33.8</td>
<td style="text-align: right;">34.2</td>
<td style="text-align: right;">34.2</td>
<td style="text-align: right;">34.4</td>
<td style="text-align: right;">33.4</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:smooth_acc" label="tab:smooth_acc"></span>

## $`\zeta_{x}`$ evaluation

We compute $`\zeta_{x}`$ value as the binomial confidence upper limit using method with $`n=1000`$ samples. For an experiment that adds Gaussian corruptions with $`\sigma`$ to the input, we use the network that is trained with Gaussian data augmentation with variance $`\sigma^2`$.

<div id="tab:zeta_all">

<table>
<caption><span class="math inline"><em>ζ</em><sub><em>x</em></sub></span> for approximate networks trained on different Gaussian augmentation <span class="math inline"><em>σ</em></span>’s.</caption>
<tbody>
<tr>
<td style="text-align: left;">Dataset</td>
<td style="text-align: left;">Architecture</td>
<td style="text-align: left;"><span class="math inline"><em>σ</em></span></td>
<td colspan="3" style="text-align: center;">Quantization</td>
<td colspan="3" style="text-align: center;">Prune</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: right;">fp16</td>
<td style="text-align: right;">bf16</td>
<td style="text-align: right;">int8</td>
<td style="text-align: right;">5%</td>
<td style="text-align: right;">10%</td>
<td style="text-align: right;">20%</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.04</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-20</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.008</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.03</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.007</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.007</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.02</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.25</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.009</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.04</td>
</tr>
<tr>
<td style="text-align: left;">CIFAR10</td>
<td style="text-align: left;">ResNet-110</td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.008</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.03</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;">0.006</td>
<td style="text-align: right;">0.008</td>
<td style="text-align: right;">0.009</td>
<td style="text-align: right;">0.007</td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.02</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">0.5</td>
<td style="text-align: right;"><span class="math inline">0.006</span></td>
<td style="text-align: right;">0.009</td>
<td style="text-align: right;"><span class="math inline">0.006</span></td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.09</td>
</tr>
<tr>
<td style="text-align: left;">ImageNet</td>
<td style="text-align: left;">ResNet-50</td>
<td style="text-align: left;">1.0</td>
<td style="text-align: right;"><span class="math inline">0.007</span></td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;"><span class="math inline">0.006</span></td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.08</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;">2.0</td>
<td style="text-align: right;"><span class="math inline">0.006</span></td>
<td style="text-align: right;">0.01</td>
<td style="text-align: right;"><span class="math inline">0.006</span></td>
<td style="text-align: right;">0.007</td>
<td style="text-align: right;">0.02</td>
<td style="text-align: right;">0.07</td>
</tr>
</tbody>
</table>

</div>

<span id="tab:zeta_all" label="tab:zeta_all"></span>

## Sensitivity to changing $`n`$

In Section <a href="#sec:exp_main" data-reference-type="ref" data-reference="sec:exp_main">5</a>, to save time due to a large number of approximations and DNNs tested, we used $`n=10^4`$ samples for $`g`$’s certification. Here, we present the effect of certifying with a larger $`n`$ by comparing the ACR vs certification time on the IRS and baseline approaches for ResNet-20 on CIFAR10. On average, for larger $`n`$, we demonstrate greater speedup for larger $`\sigma`$. For instance, for int8 quantization with $`\sigma = 1.0`$, the speedup for certifying with $`n=10^6`$ samples was $`5.85`$x as compared to certification with $`n=10^4`$ which yielded at $`2.65`$x speedup. However, for smaller $`\sigma`$, certification with a larger n results in less speedup. For $`\sigma = 0.25`$, we observe speedups from $`1.29`$x to $`1.37`$x for $`n = 10^4`$ whereas from $`0.93`$x to $`1.15`$x for $`n = 10^6`$.

<figure id="fig:abl_np">
<div class="center">
<img src="./figures/ablation_mean_radius_vs_time.png"" style="width:48.0%" />
</div>
<figcaption>CIFAR10 ResNet-20 with <span class="math inline"><em>σ</em> = 1</span>, for <span class="math inline"><em>n</em><sub><em>p</em></sub> ∈ {5%, 10%…80%}</span> of <span class="math inline"><em>n</em></span></figcaption>
</figure>

## Evaluation with larger $`n_p`$

The objective of IRS is to certify the approximated DNN with few samples. Thus, we consider $`n_p`$ ranging from $`1\%`$ to $`10\%`$. Nevertheless, we check IRS effectiveness for larger $`n_p`$ values in this ablation study.

Since, IRS certifies radius $`\sigma \Phi^{-1}(\underline{p_A}-\zeta_{x})`$ that is always smaller than original certified radius. When $`n_p = n`$, the baseline running from scratch should perform better than IRS, as it will reach a certification radius close to $`\sigma \Phi^{-1}(\underline{p_A})`$.

In this experiment, on CIFAR10 ResNet-20 with $`\sigma=1`$, we let $`n_p \in \{5\%, 10\% \dots 80\%\}`$ of $`n`$. Figure <a href="#fig:abl_np" data-reference-type="ref" data-reference="fig:abl_np">15</a> shows the ACR vs mean time plot for the baseline and IRS. We see that IRS gives speedup for $`n_p = 70\%`$. For $`n_p = 75\%`$ and $`n_p = 80\%`$, we see that baseline ACR is higher and IRS cannot achieve that ACR.

## Effect of standard deviation $`\sigma`$ on IRS speedup.

<figure id="fig:padist2">
<figure id="fig:padist_a">
<img src="./figures/pA_distribution.png"" />
<figcaption>ResNet-110 on CIFAR-10 (<span class="math inline"><em>σ</em> = 0.25</span>)</figcaption>
</figure>
<figure id="fig:padist_b">
<img src="./figures/pA_distribution.png"" />
<figcaption>ResNet-110 on CIFAR-10 (<span class="math inline"><em>σ</em> = 1.0</span>)</figcaption>
</figure>
<figure id="fig:padist_a">
<img src="./figures/pA_distribution.png"" />
<figcaption>ResNet-50 on ImageNet (<span class="math inline"><em>σ</em> = 1.0</span>)</figcaption>
</figure>
<figure id="fig:padist_b">
<img src="./figures/pA_distribution.png"" />
<figcaption>ResNet-50 on ImageNet (<span class="math inline"><em>σ</em> = 2.0</span>)</figcaption>
</figure>
<figcaption>Distribution of <span class="math inline">$\underline{p_A}$</span> values greater than 0.5 with different <span class="math inline"><em>σ</em></span> for ResNet-50 on ImageNet.</figcaption>
</figure>

Figure <a href="#fig:padist" data-reference-type="ref" data-reference="fig:padist">[fig:padist]</a> and Figure <a href="#fig:padist2" data-reference-type="ref" data-reference="fig:padist2">20</a>, present the $`\underline{p_A}`$ distribution between $`0.5`$ to $`1`$, for ResNet-110 on CIFAR-10 and ResNet-50 on ImageNet respectively. The x-axis represents the range of $`\underline{p_A}`$ values and the y-axis represents their respective proportion. The results show that while certifying larger $`\sigma`$, on average the $`\underline{p_A}`$ values are smaller. As shown in Figure <a href="#fig:padist_a" data-reference-type="ref" data-reference="fig:padist_a">18</a>, for $`\sigma = 0.25`$, less than $`35\%`$ of $`\underline{p_A}`$ values are smaller than $`0.95`$. On the other hand, in Figure <a href="#fig:padist_b" data-reference-type="ref" data-reference="fig:padist_b">19</a>, when $`\sigma = 1.0`$, the distribution is less left-skewed as nearly $`75\%`$ of $`\underline{p_A}`$ values are less than $`0.95`$. When the $`\sigma`$ is larger, the values of $`\underline{p_A}`$ tend to be farther away from 1. Therefore, the estimation of $`\underline{p_A}`$ is less precise in such cases, as observed in insight 2. As a result, non-incremental RS performs poorly compared to IRS in these situations, leading to a greater speedup with IRS.

## Threshold Parameter $`\gamma`$

Table <a href="#tab:gamma_pA" data-reference-type="ref" data-reference="tab:gamma_pA">11</a> presents the proportion of cases for which $`\underline{p_A}> \gamma`$ for the $`\gamma`$ chosen through hyperparameter search in Section <a href="#sec:ablation" data-reference-type="ref" data-reference="sec:ablation">5.4</a> for different $`\sigma`$ and networks.

<div id="tab:gamma_pA">

| Dataset | Architecture | $`\gamma`$ | $`\sigma`$ | $`\underline{p_A}> \gamma`$ |  |  |  |  |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
|  |  |  | 0.25 | 0.346 |  |  |  |  |
| CIFAR10 | ResNet-20 | 0.99 | 0.5 | 0.162 |  |  |  |  |
|  |  |  | 1.0 | 0.034 |  |  |  |  |
|  |  |  | 0.25 | 0.362 |  |  |  |  |
| CIFAR10 | ResNet-110 | 0.99 | 0.5 | 0.146 |  |  |  |  |
|  |  |  | 1.0 | 0.034 |  |  |  |  |
|  |  |  | 0.5 | 0.292 |  |  |  |  |
| ImageNet | ResNet-50 | 0.995 | 1.0 | 0.14 |  |  |  |  |
|  |  |  | 2.0 | 0.04 |  |  |  |  |

Proportion of $`\underline{p_A}`$ \> $`\gamma`$ for different $`\sigma`$ and networks.

</div>

<span id="tab:gamma_pA" label="tab:gamma_pA"></span>

For CIFAR10 ResNet-20, we observe that $`\underline{p_A}> \gamma = 0.346`$ when $`\sigma = 0.25`$ and $`\underline{p_A}> \gamma = 0.034`$ when $`\sigma = 1.0`$. Additionally, for ImageNet ResNet-50, the results show $`\underline{p_A}> \gamma = 0.292`$ when $`\sigma = 0.50`$ and $`\underline{p_A}> \gamma = 0.04`$ when $`\sigma = 2.0`$. As shown in Section <a href="#sec:exp_main" data-reference-type="ref" data-reference="sec:exp_main">5</a>, certifying larger $`\sigma`$ yields on average smaller $`\underline{p_A}`$. Expectedly, we see a smaller proportion of $`\underline{p_A}> \gamma`$ for larger $`\sigma`$ and vice versa.

## Quantization Plots

In this section, we present the ACR vs. time plots for all the quantization experiments. We use $`n=10^4`$ for samples for certification of $`g`$. For certifying $`g^p`$, we consider $`n_p`$ values from $`\{1\%, \dots 10\%\}`$ of $`n`$. Note that these smaller values of $`n, n_p`$ compared to Section <a href="#sec:exp_quant" data-reference-type="ref" data-reference="sec:exp_quant">5.1</a> allow us to perform a large number of experiments.

<span id="app:eval" label="app:eval"></span>

<figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figcaption>ResNet-20 on CIFAR10 with <span class="math inline"><em>σ</em> = 1.0</span>.</figcaption>
</figure>

<figure id="fig:cifar_int8">
<figure>
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure id="fig:cifar_bf16">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure id="fig:cifar_bf16">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figcaption>ResNet-110 on CIFAR10 with <span class="math inline"><em>σ</em> = 1.0</span>.</figcaption>
</figure>

<figure id="fig:cifar_int8">
<figure id="fig:cifar_bf16">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure id="fig:cifar_bf16">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figure id="fig:cifar_bf16">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>fp16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>bf16</figcaption>
</figure>
<figure id="fig:cifar_int8">
<img src="./figures/mean_radius_vs_total_time.png"" />
<figcaption>int8</figcaption>
</figure>
<figcaption>ResNet-50 on ImageNet with <span class="math inline"><em>σ</em> = 2.0</span>.</figcaption>
</figure>
