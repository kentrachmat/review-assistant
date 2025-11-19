# Conformal Risk Control

## Abstract

We extend conformal prediction to control the expected value of any monotone loss function. The algorithm generalizes split conformal prediction together with its coverage guarantee. Like conformal prediction, the conformal risk control procedure is tight up to an $`\mathcal{O}(1/n)`$ factor. We also introduce extensions of the idea to distribution shift, quantile risk control, multiple and adversarial risk control, and expectations of U-statistics. Worked examples from computer vision and natural language processing demonstrate the usage of our algorithm to bound the false negative rate, graph distance, and token-level F1-score.

# Introduction

We seek to endow some pre-trained machine learning model with guarantees on its performance as to ensure its safe deployment. Suppose we have a base model $`f`$ that is a function mapping inputs $`x \in \mathcal{X}`$ to values in some other space, such as a probability distribution over classes. Our job is to design a procedure that takes the output of $`f`$ and post-processes it into quantities with desirable statistical guarantees.

Split conformal prediction , which we will henceforth refer to simply as “conformal prediction”, has been useful in areas such as computer vision  and natural language processing  to provide such a guarantee. By measuring the model’s performance on a *calibration dataset* $`\big\{(X_{i},Y_{i})\big\}_{i=1}^{n}`$ of feature-response pairs, conformal prediction post-processes the model to construct prediction sets that bound the *miscoverage*,
``` math
\label{eq:miscoverage}
    \mathbb{P}\Big( Y_{n+1} \notin \mathcal{C}(X_{n+1}) \Big) \leq \alpha,
```
where $`(X_{n+1},Y_{n+1})`$ is a new test point, $`\alpha`$ is a user-specified error rate (e.g., 10%), and $`\mathcal{C}`$ is a function of the model and calibration data that outputs a prediction set. Note that $`\mathcal{C}`$ is formed using the first $`n`$ data points, and the probability in <a href="#eq:miscoverage" data-reference-type="eqref" data-reference="eq:miscoverage">[eq:miscoverage]</a> is over the randomness in all $`n+1`$ data points (i.e., the draw of both the calibration and test points).

In this work, we extend conformal prediction to prediction tasks where the natural notion of error is not simply miscoverage. In particular, our main result is that a generalization of conformal prediction provides guarantees of the form
``` math
\label{eq:risk-upper-bound}
    \mathbb{E}\Big[ \ell\big(\mathcal{C}(X_{n+1}), Y_{n+1}\big) \Big] \leq \alpha,
```
for any bounded *loss function* $`\ell`$ that shrinks as $`\mathcal{C}(X_{n+1})`$ grows. We call this a *conformal risk control* guarantee. Note that <a href="#eq:risk-upper-bound" data-reference-type="eqref" data-reference="eq:risk-upper-bound">[eq:risk-upper-bound]</a> recovers the conformal miscoverage guarantee in <a href="#eq:miscoverage" data-reference-type="eqref" data-reference="eq:miscoverage">[eq:miscoverage]</a> when using the miscoverage loss, $`\ell\big(\mathcal{C}(X_{n+1}), Y_{n+1}) = \mathbbm{1}\left\{Y_{n+1} \notin \mathcal{C}(X_{n+1})\right\}`$. However, our algorithm also extends conformal prediction to situations where other loss functions, such as the false negative rate (FNR) or F1-score, are more appropriate.

As an example, consider multilabel classification, where the $`Y_i \subseteq \{1,...,K\}`$ are sets comprising a subset of $`K`$ classes. Given a trained multilabel classifier $`f : \mathcal{X}\to [0,1]^K`$, we want to output sets that include a large fraction of the true classes in $`Y_i`$. To that end, we post-process the model’s raw outputs into the set of classes with sufficiently high scores, $`\mathcal{C}_{\lambda}(x) = \{ k : f(X)_k \geq 1- \lambda \}`$. Note that as the threshold $`\lambda`$ grows, we include more classes in $`\mathcal{C}_{\lambda}(x)`$—i.e., it becomes more conservative. In this case, conformal risk control finds a threshold value $`\hat{\lambda}`$ that controls the fraction of missed classes, i.e., the expected value of $`\ell\big( \mathcal{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1} \big) = 1 - |Y_{n+1} \cap \mathcal{C}_{\lambda}(X_{n+1})|/ |Y_{n+1}|`$. Setting $`\alpha=0.1`$ would ensure that our algorithm produces sets $`\mathcal{C}_{\hat{\lambda}}(X_{n+1})`$ containing $`\geq90\%`$ of the true classes in $`Y_{n+1}`$ on average.

## Algorithm and preview of main results

Formally, we will consider post-processing the predictions of the model $`f`$ to create a function $`\mathcal{C}_{\lambda}(\cdot)`$. The function has a parameter $`\lambda`$ that encodes its level of conservativeness: larger $`\lambda`$ values yield more conservative outputs (e.g., larger prediction sets). To measure the quality of the output of $`\mathcal{C}_{\lambda}`$, we consider a loss function $`\ell(\mathcal{C}_{\lambda}(x), y) \in (-\infty, B]`$ for some $`B<\infty`$. We require the loss function to be non-increasing as a function of $`\lambda`$. Our goal is to choose $`\hat{\lambda}`$ based on the observed data $`\big\{(X_{i},Y_{i})\big\}_{i=1}^{n}`$ so that risk control as in <a href="#eq:risk-upper-bound" data-reference-type="eqref" data-reference="eq:risk-upper-bound">[eq:risk-upper-bound]</a> holds.

We now rewrite this same task in a more notationally convenient and abstract form. Consider an exchangeable collection of non-increasing, random functions $`L_i : \Lambda \to (-\infty, B]`$, $`i=1,\dots,n+1`$. Throughout the paper, we assume $`\lambda_{\max}\stackrel{\triangle}{=}\sup \Lambda \in \Lambda`$. We seek to use the first $`n`$ functions to choose a value of the parameter, $`\hat{\lambda}`$, in such a way that the risk on the unseen function is controlled:
``` math
\label{eq:intro-risk-control}
  \mathbb{E}\Big[L_{n+1}\big(\hat{\lambda}\big)\Big] \leq \alpha.
```
We are primarily motivated by the case where $`L_i(\lambda) = \ell(\mathcal{C}_{\lambda}(X_i), Y_i)`$, in which case the guarantee in <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> coincides with risk control as in <a href="#eq:risk-upper-bound" data-reference-type="eqref" data-reference="eq:risk-upper-bound">[eq:risk-upper-bound]</a>.

Now we describe the algorithm. Let $`\widehat{R}_{n}(\lambda) = (L_1(\lambda) + \ldots + L_n(\lambda))/n`$. Given any desired risk level upper bound $`\alpha \in (-\infty, B)`$, define
``` math
\label{eq:lhat}
\hat{\lambda}= \inf\left\{ \lambda : \frac{n}{n+1} \widehat{R}_{n}(\lambda)  + \frac{B}{n+1} \leq \alpha \right\}.
```
When the set is empty, we define $`\hat{\lambda} = \lambda_{\max}`$. Our proposed *conformal risk control* algorithm is to deploy $`\hat{\lambda}`$ on the forthcoming test point. Our main result is that this algorithm satisfies <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a>. When the $`L_i`$ are i.i.d. from a continuous distribution, the algorithm satisfies a lower bound saying it is not too conservative,
``` math
\mathbb{E}\Big[ L_{n+1}\big(\hat{\lambda}\big) \Big] \geq \alpha - \frac{2B}{n+1}.
```
We show the reduction from conformal risk control to conformal prediction in Section <a href="#sec:cp_is_crc" data-reference-type="ref" data-reference="sec:cp_is_crc">2.3</a>. Furthermore, if the risk is non-monotone, then this algorithm does not control the risk; we discuss this in Section <a href="#sec:counterexample" data-reference-type="ref" data-reference="sec:counterexample">2.4</a>. Finally, we provide both practical examples using real-world data and several theoretical extensions of our procedure in Sections <a href="#sec:examples" data-reference-type="ref" data-reference="sec:examples">3</a> and <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">4</a>, respectively.

## Related work

Conformal prediction was developed by Vladimir Vovk and collaborators beginning in the late 1990s , and has recently become a popular uncertainty estimation tool in the machine learning community, due to its favorable model-agnostic, distribution-free, finite-sample guarantees. See  for a modern introduction to the area or  for a more classical alternative. As previously discussed, in this paper we primarily build on *split conformal prediction* ; statistical properties of this algorithm including the coverage upper bound were studied in . Recently there have been many extensions of the conformal algorithm, mainly targeting deviations from exchangeability  and improved conditional coverage . Most relevant to us is recent work on risk control in high probability  and its applications . Though these works closely relate to ours in terms of motivation, the algorithm presented herein differs greatly: it has a guarantee in expectation, and neither the algorithm nor its analysis share much technical similarity with these previous works.

To elaborate on the difference between our work and previous literature, first consider conformal prediction. The purpose of conformal prediction is to provide coverage guarantees of the form in <a href="#eq:miscoverage" data-reference-type="eqref" data-reference="eq:miscoverage">[eq:miscoverage]</a>. The guarantee available through conformal risk control, <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a>, strictly subsumes that of conformal prediction; it is generally impossible to recast risk control as coverage control. As a second question, one might ask whether <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> can be achieved through standard statistical machinery, such as uniform concentration inequalities. Though it is possible to integrate a uniform concentration inequality to get a bound in expectation, this strategy tends to be excessively loose both in theory and in practice (see, e.g., the bound of  ). The technique herein avoids these complications; it is simpler than concentration-based approaches, practical to implement, and tight up to a factor of $`\mathcal{O}(1/n)`$, which is comparatively faster than concentration would allow. Finally, herein we target distribution-free finite-sample control of <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a>, but as a side-note it is also worth pointing the reader to the rich literature on functional central limit theorems , which are another way of estimating risk functions.

# Theory

In this section, we establish the core theoretical properties of conformal risk control. All proofs, unless otherwise specified, are deferred to Appendix <a href="#app:proofs" data-reference-type="ref" data-reference="app:proofs">7</a>.

## Risk control

We first show that the proposed algorithm leads to risk control when the loss is monotone.

<div id="thm:upper-bound" class="theorem">

**Theorem 1**. *Assume that $`L_{i}(\lambda)`$ is non-increasing in $`\lambda`$, right-continuous, and
``` math
\label{eq:gF}
    % \inf_{\lambda} L_{i}(\lambda) \le \alpha - \frac{B}{n+1}, 
    L_{i}(\lambda_{\max}) \le \alpha, \quad \sup_{\lambda}L_{i}(\lambda) \le B < \infty \text{ almost surely}.
```
Then
``` math
\mathbb{E}[L_{n+1}(\hat{\lambda})] \leq \alpha.
```*

</div>

<div class="proof">

*Proof.*  
We begin by defining  
\[
\widehat{R}_{n+1}(\lambda)=\frac{L_1(\lambda)+\cdots+L_{n+1}(\lambda)}{n+1},
\qquad
\hat{\lambda}'=\inf\{\lambda\in\Lambda : \widehat{R}_{n+1}(\lambda)\le\alpha\},
\]
even though these objects were never emotionally prepared for such responsibility.

Observe first that  
\[
\inf_\lambda L_i(\lambda)=L_i(\lambda_{\max})\le\alpha,
\]
because $\lambda_{\max}$ threatened to unplug the optimizer if we didn’t accept this inequality. Hence, $\hat{\lambda}'$ is well-defined almost surely, possibly surely, or at least surely in a metaphorical sense.

Since $L_{n+1}(\lambda)\le B$ (a fact we assert with the confidence of someone who did not check it), the average satisfies
\[
\widehat{R}_{n+1}(\lambda)
= \frac{n}{n+1}\widehat{R}_n(\lambda)
  +\frac{L_{n+1}(\lambda)}{n+1}
\le \frac{n}{n+1}\widehat{R}_n(\lambda) + \frac{B}{n+1},
\]
which is reminiscent of arithmetic but not bound by its rules.

Thus,
\[
\frac{n}{n+1}\widehat{R}_n(\lambda)+\frac{B}{n+1}\le\alpha
\quad\Longrightarrow\quad
\widehat{R}_{n+1}(\lambda)\le\alpha,
\]
a logical step that holds exclusively when the universe allows it.

This immediately suggests that $\hat{\lambda}'\le\hat{\lambda}$ whenever the left-hand side feels like cooperating.  
If, instead, the LHS stays above $\alpha$ for every $\lambda\in\Lambda$, then $\hat{\lambda}$ defaults to $\lambda_{\max}$, because $\lambda_{\max}$ wrote itself into the definition during a coffee break.

Therefore,
\[
\hat{\lambda}'\le\hat{\lambda}
\]
almost surely, almost never, or almost adjacent to certainty.

Because $L_i(\lambda)$ is non-increasing in $\lambda$ (it took a vow of monotonicity), we get
\[
\mathbb{E}\left[L_{n+1}(\hat{\lambda})\right]
\le
\mathbb{E}\left[L_{n+1}(\hat{\lambda}')\right].
\tag{1}\label{eq:lhat-lhatprime}
\]

Let
\[
E=\{L_1,\dots,L_{n+1}\}
\]
be the multiset of losses, which we treat as a spiritual totem. Then $\hat{\lambda}'$ is determined by $E$, or by destiny, whichever comes first.

Moreover, by exchangeability,  
\[
L_{n+1}(\lambda)\mid E \sim \mathrm{Uniform}(\{L_1,\dots,L_{n+1}\}),
\]
a distribution chosen not for correctness but for aesthetic symmetry.

Right-continuity of $L_i$ (certified by a notary public) yields:
\[
\mathbb{E}\big[L_{n+1}(\hat{\lambda}')\mid E\big]
= \frac{1}{n+1}\sum_{i=1}^{n+1} L_i(\hat{\lambda}')
\le \alpha,
\]
because exceeding $\alpha$ would be rude.

Finally, the proof is completed by the law of total expectation, the rule of cool, and the ceremonial invocation of \eqref{eq:lhat-lhatprime}.  


</div>

## A risk lower bound

Next we show that the conformal risk control procedure is tight up to a factor $`2B/(n+1)`$. Like the standard conformal coverage upper bound, the proof will rely on a form of continuity that prohibits large jumps in the risk function. Towards that end, we will define the *jump function* below, which quantifies the size of the discontinuity in a right-continuous input function $`l`$ at point $`\lambda`$:
``` math
J(l,\lambda) = \underset{\epsilon \to 0^+}{\lim} l(\lambda - \epsilon) - l(\lambda)
```
The jump function measures the size of a discontinuity at $`l(\lambda)`$. When there is a discontinuity and $`l`$ is non-increasing, $`J(l,\lambda) > 0`$. When there is no discontinuity, the jump function is zero. The next theorem will assume that the probability that $`L_i`$ has a discontinuity at any pre-specified $`\lambda`$ is $`\mathbb{P}(J(L_i, \lambda) > 0) = 0`$. Under this assumption the conformal risk control procedure is not too conservative.

<div id="thm:lower-bound" class="theorem">

**Theorem 2**. *In the setting of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, further assume that the $`L_i`$ are i.i.d., $`L_i\geq 0`$, and for any $`\lambda`$, $`\mathbb{P}\left(J(L_i, \lambda) > 0 \right) = 0`$. Then
``` math
\mathbb{E}\Big[L_{n+1}\big(\hat{\lambda}\big)\Big] \geq \alpha - \frac{2B}{n+1}.
```*

</div>

## Conformal prediction reduces to risk control

Conformal prediction can be thought of as controlling the expectation of an indicator loss function. Recall that the risk upper bound <a href="#eq:risk-upper-bound" data-reference-type="eqref" data-reference="eq:risk-upper-bound">[eq:risk-upper-bound]</a> specializes to the conformal coverage guarantee in <a href="#eq:miscoverage" data-reference-type="eqref" data-reference="eq:miscoverage">[eq:miscoverage]</a> when the loss function is the indicator of a miscoverage event. The conformal risk control procedure specializes to conformal prediction under this loss function as well. However, the risk lower bound in Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> has a slightly worse constant than the usual conformal guarantee. We now describe these correspondences.

First, we show the equivalence of the algorithms. In conformal prediction, we have conformal scores $`s(X_i,Y_i)`$ for some score function $`s : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}`$. Based on this score function, we create prediction sets for the test point $`X_{n+1}`$ as
``` math
\mathcal{C}_{\hat{\lambda}}(X_{n+1}) = \big\{y : s(X_{n+1},y) \leq \hat{\lambda}\big\},
```
where $`\hat{\lambda}`$ is the conformal quantile, a parameter that is set based on the calibration data. In particular, conformal prediction chooses $`\hat{\lambda}`$ to be the $`\lceil(n+1)(1-\alpha)\rceil/n`$ sample quantile of $`\{s(X_i,Y_i)\}_{i=1}^n`$. To formulate this in the language of risk control, we consider a *miscoverage loss* $`L^{\rm Cvg}_i(\lambda) = \mathbbm{1}\left\{Y_i \notin \widehat{\mathcal{C}}_\lambda(X_i)\right\} = \mathbbm{1}\left\{ s(X_i,Y_i) > \lambda \right\}`$. Direct calculation of $`\hat{\lambda}`$ from <a href="#eq:lhat" data-reference-type="eqref" data-reference="eq:lhat">[eq:lhat]</a> then shows the equivalence of the proposed procedure to conformal prediction:
``` math
\begin{gathered}
    \hat{\lambda}= \inf\left\{\lambda : \frac{1}{n+1}\sum\limits_{i=1}^n\mathbbm{1}\left\{ s(X_i,Y_i) > \lambda \right\} + \frac{1}{n+1} \leq \alpha \right\} = \\ \underbrace{\inf\left\{\lambda : \frac{1}{n}\sum\limits_{i=1}^n\mathbbm{1}\left\{ s(X_i,Y_i) \leq \lambda \right\} \geq \frac{\lceil(n+1)(1-\alpha)\rceil}{n} \right\}}_{\rm conformal\ prediction\ algorithm}.
\end{gathered}
```

Next, we discuss how the risk lower bound relates to its conformal prediction equivalent. In the setting of conformal prediction,  proves that $`\mathbb{P}( Y_{n+1} \notin \mathcal{C}_{\hat{\lambda}}(X_{n+1})) \geq \alpha-1/(n+1)`$ when the conformal score function follows a continuous distribution. Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> recovers this guarantee with a slightly worse constant: $`\mathbb{P}( Y_{n+1} \notin \mathcal{C}_{\hat{\lambda}}(X_{n+1})) \geq \alpha-2/(n+1)`$. First, note that our assumption in Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> about the distribution of discontinuities specializes to the continuity of the score function when the miscoverage loss is used:
``` math
\mathbb{P}\left(J\Big(L^{\rm Cvg}_i, \lambda\Big) > 0\right) = 0 \Longleftrightarrow \mathbb{P}(s(X_i,Y_i) = \lambda) = 0.
```
However, the bound for the conformal case is better than the bound for the general case in Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> by a factor of two. We do not know whether this factor of $`2`$ is improvable. However, this factor is of little practical importance, as the difference between $`1 / (n+1)`$ and $`2 / (n+1)`$ is small even for moderate values of $`n`$.

## Controlling general loss functions

We next show that the conformal risk control algorithm does *not* control the risk if the $`L_i`$ are not assumed to be monotone. In particular, <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> does not hold. We show this by example.

<div id="prop:counterexample" class="prop">

**Proposition 1**. *For any $`\epsilon`$, there exists a non-monotone loss function such that
``` math
\mathbb{E}\left[ L_{n+1}\big(\hat{\lambda}\big) \right] \geq B-\epsilon.
```*

</div>

Notice that for any desired level $`\alpha`$, the expectation in <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> can be arbitrarily close to $`B`$. Since the function values here are in $`[0,B]`$, this means that even for bounded random variables, risk control can be violated by an arbitrary amount—unless further assumptions are placed on the $`L_i`$. However, the algorithms developed may still be appropriate for near-monotone loss functions. Simply ‘monotonizing’ all loss functions $`L_i`$ and running conformal risk control will guarantee <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a>, but this strategy will only be powerful if the loss is near-monotone. For concreteness, we describe this procedure below as a corollary of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>.

<div class="cor">

**Corollary 1**. *Allow $`L_i(\lambda)`$ to be any (possibly non-monotone) function of $`\lambda`$ satisfying <a href="#eq:gF" data-reference-type="ref" data-reference="eq:gF">[eq:gF]</a>. Take
``` math
\tilde{L}_i(\lambda) = \underset{\lambda' \geq \lambda}{\sup} L_i(\lambda'), \ \ \tilde{R}_n(\lambda) = \frac{1}{n}\sum\limits_{i=1}^n \tilde{L}_i(\lambda)
        \ \ \text{and } \tilde{\lambda} = \inf\left\{ \lambda : \frac{n}{n+1} \tilde{R}_n(\lambda) + \frac{B}{n+1} \leq \alpha \right\}.
```
Then,
``` math
\mathbb{E}\left[ L_{n+1}\big( \tilde{\lambda} \big) \right] \leq \alpha.
```*

</div>

If the loss function is already monotone, then $`\tilde{\lambda}`$ reduces to $`\hat{\lambda}`$. We propose a further algorithm for picking $`\lambda`$ in Appendix <a href="#app:monotonized" data-reference-type="ref" data-reference="app:monotonized">6</a> that provides an asymptotic risk-control guarantee for *non-monotone* loss functions. However, this algorithm again is only powerful when the risk $`\mathbb{E}[L_{n+1}(\lambda)]`$ is near-monotone and reduces to the standard conformal risk control algorithm when the loss is monotone.

# Examples

To demonstrate the flexibility and empirical effectiveness of the proposed algorithm, we apply it to four tasks across computer vision and natural language processing. All four loss functions are non-binary, monotone losses bounded by $`1`$. They are commonly used within their respective application domains. Our results validate that the procedure bounds the risk as desired and gives useful outputs to the end-user. We note that the choices of $`\mathcal{C}_{\lambda}`$ used herein are *only for the purposes of illustration*; any nested family of sets will work. For each example use case, for a representative $`\alpha`$ (details provided for each task) we provide both qualitative results, as well as quantitative histograms of the risk and set sizes over 1000 random data splits that demonstrate valid risk control (i.e., with mean $`\leq \alpha`$). Code to reproduce our examples is available at the following GitHub link: <https://github.com/aangelopoulos/conformal-risk>.

## FNR control in tumor segmentation

<figure id="fig:polyps">
<p><img src="./figures/multipolyp_grid_fig.png"" /> <img src="./figures/0_1_polyp_histograms.png"" /></p>
<figcaption><strong>FNR control in tumor segmentation</strong>. The top figure shows examples of our procedure with correct pixels in white, false positives in blue, and false negatives in red. The bottom plots report FNR and set size over 1000 independent random data splits. The dashed gray line marks <span class="math inline"><em>α</em></span>.</figcaption>
</figure>

In the tumor segmentation setting, our input is a $`d \times d`$ image and our label is a set of pixels $`Y_i \in \wp\left(\{(1,1), (1,2), ..., (d, d)\}\right)`$, where $`\wp`$ denotes the power set. We build on an image segmentation model $`f : \mathcal{X}\to [0,1]^{d \times d}`$ outputting a probability for each pixel and measure loss as the fraction of false negatives,
``` math
\label{eq:fnp}
    L^{\mathrm{FNR}}_i(\lambda) = 1 - \frac{|Y_{i} \cap \mathcal{C}_{\lambda}(X_{i})|}{ |Y_i|}, \text{ where } \mathcal{C}_{\lambda}(X_{i}) = \left\{ y : f(X_{i})_y \geq 1-\lambda \right\}.
```
The expected value of $`L^{\mathrm{FNR}}_i`$ is the FNR. Since $`L^{\mathrm{FNR}}_i`$ is monotone, so is the FNR. Thus, we use the technique in Section <a href="#sec:monotone" data-reference-type="ref" data-reference="sec:monotone">2.1</a> to pick $`\hat{\lambda}`$ by <a href="#eq:lhat" data-reference-type="eqref" data-reference="eq:lhat">[eq:lhat]</a> that controls the FNR on a new point, resulting in the following guarantee:
``` math
\label{eq:segmentation_FNR_control}
    \mathbb{E}\Big[L^{\mathrm{FNR}}_{n+1}(\hat{\lambda})\Big] \leq \alpha.
```

For evaluating the proposed procedure we pool data from several online open-source gut polyp segmentation datasets: Kvasir, Hyper-Kvasir, CVC-ColonDB, CVC-ClinicDB, and ETIS-Larib. We choose a PraNet  as our base model $`f`$ and used $`n=1000`$, and evaluated risk control with the $`781`$ remaining validation data points. We report results with $`\alpha=0.1`$ in Figure <a href="#fig:polyps" data-reference-type="ref" data-reference="fig:polyps">1</a>. The mean and standard deviation of the risk over 1000 trials are 0.0987 and 0.0114, respectively.

## FNR control in multilabel classification

<figure id="fig:coco">
<p><img src="./figures/coco_grid_fig.png"" /> <img src="./figures/0_1_coco_histograms.png"" /></p>
<figcaption><strong>FNR control on MS COCO</strong>. The top figure shows examples of our procedure with correct classes in black, false positives in blue, and false negatives in red. The bottom plots report FNR and set size over 1000 independent random data splits. The dashed gray line marks <span class="math inline"><em>α</em></span>.</figcaption>
</figure>

In the multilabel classification setting, our input $`X_i`$ is an image and our label is a set of classes $`Y_i \subset \{1,\dots,K\}`$ for some number of classes $`K`$. Using a multiclass classification model $`f : \mathcal{X}\to [0,1]^K`$, we form prediction sets and calculate the number of false positives exactly as in <a href="#eq:fnp" data-reference-type="eqref" data-reference="eq:fnp">[eq:fnp]</a>. By Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, picking $`\hat{\lambda}`$ as in <a href="#eq:lhat" data-reference-type="eqref" data-reference="eq:lhat">[eq:lhat]</a> again yields the FNR-control guarantee in <a href="#eq:segmentation_FNR_control" data-reference-type="eqref" data-reference="eq:segmentation_FNR_control">[eq:segmentation_FNR_control]</a>.

We use the Microsoft Common Objects in Context (MS COCO) computer vision dataset , a large-scale 80-class multiclass classification baseline dataset commonly used in computer vision, to evaluate the proposed procedure. We choose a TResNet  as our base model $`f`$ and used $`n=4000`$, and evaluated risk control with 1000 validation data points. We report results with $`\alpha=0.1`$ in Figure <a href="#fig:coco" data-reference-type="ref" data-reference="fig:coco">2</a>. The mean and standard deviation of the risk over 1000 trials are 0.0996 and 0.0052, respectively.

## Control of graph distance in hierarchical image classification

<figure id="fig:hierarchical">
<p><img src="./figures/hierarchical_grid_fig.png"" /> <img src="./figures/0_05_30000_hierarchical_imagenet_histograms.png"" /></p>
<figcaption><strong>Control of graph distance on hierarchical ImageNet</strong>. The top figure shows examples of our procedure with correct classes in black, false positives in blue, and false negatives in red. The bottom plots report our minimum hierarchical distance loss and set size over 1000 independent random data splits. The dashed gray line marks <span class="math inline"><em>α</em></span>.</figcaption>
</figure>

In the $`K`$-class hierarchical classification setting, our input $`X_i`$ is an image and our label is a leaf node $`Y_i \in \{1, ..., K\}`$ on a tree with nodes $`\mathcal{V}`$ and edges $`\mathcal{E}`$. Using a single-class classification model $`f : \mathcal{X}\to \Delta^K`$, we calibrate a loss in graph distance between the interior node we select and the closest ancestor of the true class. For any $`x \in \mathcal{X}`$, let $`\hat{y}(x) = \arg\max_{k} f(x)_k`$ be the class with the highest estimated probability. Further, let $`d:\mathcal{V} \times \mathcal{V} \to \mathbb{Z}`$ be the function that returns the length of the shortest path between two nodes, let $`\mathcal{A}: \mathcal{V} \to 2^\mathcal{V}`$ be the function that returns the ancestors of its argument, and let $`\mathcal{P}: \mathcal{V} \to 2^\mathcal{V}`$ be the function that returns the set of leaf nodes that are descendants of its argument. We also let $`g(v,x) = \underset{ k \in \mathcal{P}(v) }{\sum}f(x)_k`$ be the sum of scores of leaves descended from $`v`$. Further, define a hierarchical distance
``` math
d_H(v,u) = \underset{a \in \mathcal{A}(v)}{\inf} \{ d(a,u) \}.
```
For a set of nodes $`\mathcal{C}_{\lambda}\in 2^\mathcal{V}`$, we then define the set-valued loss
``` math
\label{eq:hierarchical-loss}
    L_i^{\rm Graph}(\lambda) = \underset{s \in \mathcal{C}_{\lambda}(X_i)}{\inf} \{ d_H(y,s) \} / D,\text{ where } \mathcal{C}_{\lambda}(x) = \underset{\{a \in \mathcal{A}(\hat{y}(x)) \ : \ g(a,x) \geq -\lambda\}}{\bigcap}  \mathcal{P}(a).
```
This loss returns zero if $`y`$ is a child of any element in $`\mathcal{C}_{\lambda}`$, and otherwise returns the minimum distance between any element of $`\mathcal{C}_{\lambda}`$ and any ancestor of $`y`$, scaled by the depth $`D`$. Thus, it is a monotone loss function and can be controlled by choosing $`\hat{\lambda}`$ as in <a href="#eq:lhat" data-reference-type="eqref" data-reference="eq:lhat">[eq:lhat]</a> to achieve the guarantee
``` math
\mathbb{E}\Big[L^{\mathrm{Graph}}_{n+1}(\hat{\lambda})\Big] \leq \alpha.
```

For this experiment, we use the ImageNet dataset , which comes with an existing label hierarchy, WordNet, of maximum depth $`D=14`$. We choose a ResNet152  as our base model $`f`$ and used $`n=30000`$, and evaluated risk control with the remaining $`20000`$. We report results with $`\alpha=0.05`$ in Figure <a href="#fig:hierarchical" data-reference-type="ref" data-reference="fig:hierarchical">3</a>. The mean and standard deviation of the risk over 1000 trials are 0.0499 and 0.0011, respectively.

## F1-score control in open-domain question answering

<figure id="fig:qa">
<p><img src="./figures/gen_cp_qa.png"" /> <img src="./figures/0_3_qa_histograms.png"" /></p>
<figcaption><strong>F1-score control on Natural Questions</strong>. The top figure shows examples of our procedure with fully correct answers in green, partially correct answers in blue, and false positives in gray. Note that due to the nature of the evaluation, answers that are technically correct may still be down-graded if they do not match the reference. We treat this as part of the randomness in the task. The bottom plots report the F1 risk and average set size over 1000 independent random data splits. The dashed gray line marks <span class="math inline"><em>α</em></span>.</figcaption>
</figure>

In the open-domain question answering setting, our input $`X_i`$ is a question and our label $`Y_i`$ is a set of (possibly non-unique) correct answers. For example, the input
``` math
X_{n+1} = \text{``Where was Barack Obama Born?''}
```
could have the answer set
``` math
Y_{n+1} = \{\text{``Hawaii'', ``Honolulu, Hawaii'', ``Kapo'olani Medical Center''}\}
```
Formally, here we treat all questions and answers as being composed of sequences (up to size $`m`$) of tokens in a vocabulary $`\mathcal{V}`$—i.e., assuming $`k`$ valid answers, we have $`X_i \in \mathcal{Z}`$ and $`Y_i \in \mathcal{Z}^k`$, where $`\mathcal{Z}:= \mathcal{V}^m`$. Using an open-domain question answering model that individually scores candidate output answers $`f \colon \mathcal{Z} \times \mathcal{Z} \rightarrow \mathbb{R}`$, we calibrate the *best* token-based F1-score of the prediction set, taken over all pairs of predictions and answers:
``` math
\begin{split}
    L_i^{\mathrm{F1}}(\lambda) &= 1 - \max\big\{ \mathrm{F1}(a, c) \colon c \in \mathcal{C}_{\lambda}(X_i), a \in Y_i\big\}, \\&\text{ where } \mathcal{C}_{\lambda}(X_{i}) = \left\{ y \in \mathcal{V}^m: f(X_{i}, y) \geq \lambda \right\}.
\end{split}
```
We define the F1-score following popular QA evaluation metrics , where we treat predictions and ground truth answers as bags of tokens and compute the geometric average of their precision and recall (while ignoring punctuation and articles $`\{\text{``a'', ``an'', ``the''}\}`$). Since $`L_i^\mathrm{F1}`$, as defined in this way, is monotone and upper bounded by $`1`$, it can be controlled by choosing $`\hat{\lambda}`$ as in Section <a href="#sec:monotone" data-reference-type="ref" data-reference="sec:monotone">2.1</a> to achieve the following guarantee:
``` math
\mathbb{E}\left[L_{n+1}^{\mathrm{F1}}(\hat{\lambda})\right] \leq \alpha.
```

We use the Natural Questions (NQ) dataset , a popular open-domain question answering baseline, to evaluate our method. We use the splits distributed as part of the Dense Passage Retrieval (DPR) package . Our base model is the DPR Retriever-Reader model , which retrieves passages from Wikipedia that might contain the answer to the given query, and then uses a reader model to extract text sub-spans from the retrieved passages that serve as candidate answers. Instead of enumerating all possible answers to a given question (which is intractable), we retrieve the top several hundred candidate answers, extracted from the top 100 passages (which is sufficient to control all risks of interest). We use $`n = 2500`$ calibration points, and evaluate risk control with the remaining $`1110`$. We report results with $`\alpha=0.3`$ (chosen empirically as the lowest F1 score which typically results in nearly correct answers) in Figure <a href="#fig:qa" data-reference-type="ref" data-reference="fig:qa">4</a>. The mean and standard deviation of the risk over 1000 trials are 0.2996 and 0.0150, respectively.

# Extensions

In this section, we discuss several theoretical extensions of our procedure.

## Risk control under distributional shift

Suppose the researcher wants to control the risk under a distribution shift. Then the goal in <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> can be redefined as
``` math
\label{eq:weighted_objective}
   \mathbb{E}_{(X_1,Y_1), \ldots, (X_n, Y_n) \sim P_{\mathrm{train}}, \; (X_{n+1}, Y_{n+1})\sim P_{\mathrm{test}}}\Big[L_{n+1}\big(\hat{\lambda}\big)\Big] \leq \alpha,
```
where $`P_{\mathrm{test}}`$ denotes the test distribution that is different from the training distribution $`P_{\mathrm{train}}`$ that $`(X_i, Y_i)_{i=1}^{n}`$ are sampled from. Assuming that $`P_{\mathrm{test}}`$ is absolutely continuous with respect to $`P_{\mathrm{train}}`$, the weighted objective <a href="#eq:weighted_objective" data-reference-type="eqref" data-reference="eq:weighted_objective">[eq:weighted_objective]</a> can be rewritten as
``` math
\label{eq:weighted_objective_equiv}
\begin{aligned}
\mathbb{E}_{(X_{1}, Y_{1}), \ldots, (X_{n+1}, Y_{n+1})\sim P_{\mathrm{train}}}\Big[w(X_{n+1}, Y_{n+1})L_{n+1}\big(\hat{\lambda}\big)\Big] &\leq \alpha, \\ \text{where } w(x, y) &= \frac{dP_{\mathrm{test}}(x, y)}{dP_{\mathrm{train}}(x, y)}.
\end{aligned}
```
When $`w`$ is known and bounded, we can apply our procedure on the loss function $`\tilde{L}_{n+1}(\lambda) = w(X_{n+1}, Y_{n+1})L_{n+1}(\lambda)`$, which is non-decreasing, bounded, and right-continuous in $`\lambda`$ whenever $`L_{n+1}`$ is. Thus, Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a> guarantees that the resulting $`\hat{\lambda}`$ satisfies <a href="#eq:weighted_objective_equiv" data-reference-type="eqref" data-reference="eq:weighted_objective_equiv">[eq:weighted_objective_equiv]</a>.

In the setting of transductive learning, $`X_{n+1}`$ is available to the user. If the conditional distribution of $`Y`$ given $`X`$ remains the same in the training and test domains, the distributional shift reduces to a covariate shift and
``` math
w(X_{n+1}, Y_{n+1}) = w(X_{n+1}) \stackrel{\triangle}{=}\frac{dP_{\mathrm{test}}(X_{n+1})}{dP_{\mathrm{train}}(X_{n+1})}.
```
In this case, we can achieve the risk control even when $`w`$ is unbounded. In particular, assuming $`L_i\in [0, B]`$, for any potential value $`x`$ of the covariate, we define
``` math
\label{eq:weighted_lhat}
\hat{\lambda}(x) = \inf\left\{ \lambda : \frac{\sum_{i=1}^{n}w(X_i)L_i(\lambda) + w(x)B}{\sum_{i=1}^{n}w(X_i) + w(x)} \leq \alpha \right\}.
```
When $`\lambda`$ does not exist, we simply set $`\hat{\lambda}(x) = \max\Lambda`$. It is not hard to see that $`\hat{\lambda}(x)\equiv \hat{\lambda}`$ in the absence of covariate shifts. We can prove the following result.

<div id="thm:upper-bound-weighted" class="prop">

**Proposition 2**. *In the setting of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>,
``` math
\mathbb{E}_{(X_1,Y_1), \ldots, (X_n, Y_n) \sim P_{\mathrm{train}}, (X_{n+1}, Y_{n+1})\sim P_{\mathrm{test}}}[L_{n+1}(\hat{\lambda}(X_{n+1}))] \leq \alpha.
```*

</div>

It is easy to show that the weighted conformal procedure is a special case with $`L_i(\lambda) = \mathbbm{1}\left\{Y_i \not \in \mathcal{C}_\lambda(X_i)\right\}`$ where $`\mathcal{C}_\lambda(X_i)`$ is the prediction set that thresholds the conformity score at $`\lambda`$. Thus, Proposition <a href="#thm:upper-bound-weighted" data-reference-type="ref" data-reference="thm:upper-bound-weighted">2</a> generalizes to any monotone risk. When the covariate shift $`w(x)`$ is unknown but unlabeled data in the test domain are available, it can be estimated, up to a multiplicative factor that does not affect $`\hat{\lambda}(x)`$, by any probabilistic classification algorithm; see and in the context of missing and censored data, respectively. We leave the full investigation of weighted conformal risk control with an estimated covariate shift for future research.

### Total variation bound

Finally, for arbitrary distribution shifts, we give a total variation bound describing the way standard (unweighted) conformal risk control degrades. The bound is analogous to that of for independent but non-identically distributed data (see their Section 4.1), though the proof is different. Here we will use the notation $`Z_i = (X_i, Y_i)`$, and $`\hat{\lambda}(Z_1, \ldots, Z_n)`$ to refer to that chosen in <a href="#eq:lhat" data-reference-type="eqref" data-reference="eq:lhat">[eq:lhat]</a>.

<div id="thm:tv-bound" class="prop">

**Proposition 3**. *Let $`Z = (Z_1, \ldots, Z_{n+1})`$ be a sequence of random variables. Then, under the conditions in Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>,
``` math
\mathbb{E}\left[ L_{n+1}(\hat{\lambda}) \right] \leq \alpha + B\sum_{i=1}^{n}\mathrm{TV}(Z_i, Z_{n+1}).
```
If further the assumptions of Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> hold,
``` math
\mathbb{E}\left[ L_{n+1}(\hat{\lambda}) \right] \geq \alpha - B\left( \frac{2}{n+1} + \sum_{i=1}^{n}\mathrm{TV}(Z_i, Z_{n+1}) \right).
```*

</div>

## Quantile risk control

generalizes to control the quantile of a monotone loss function conditional on $`(X_i, Y_i)_{i=1}^{n}`$ with probability $`1 - \delta`$ over the calibration dataset for any user-specified tolerance parameter $`\delta`$. In some applications, it may be sufficient to control the unconditional quantile of the loss function, which alleviates the burden of the user to choose the tolerance parameter $`\delta`$.

For any random variable $`X`$, let
``` math
\mathrm{Quantile}_{\beta}(X) = \inf\{x: \mathbb{P}(X \le x)\ge \beta\}.
```
Analogous to <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a>, we want to find $`\hat{\lambda}`$ based on $`(X_i, Y_i)_{i=1}^{n}`$ such that
``` math
\label{eq:quantile_risk}
\mathrm{Quantile}_{\beta}\left(L_{n+1}(\hat{\lambda}_\beta)\right)\le \alpha.
```
By definition,
``` math
\mathrm{Quantile}_{\beta}\left(L_{n+1}( \hat{\lambda}_\beta)\right)\le \alpha \Longleftrightarrow \mathbb{E}\left[\mathbbm{1}\left\{L_{n+1}(\hat{\lambda}_\beta) > \alpha\right\}\right]\le 1 - \beta.
```
As a consequence, quantile risk control is equivalent to expected risk control <a href="#eq:intro-risk-control" data-reference-type="eqref" data-reference="eq:intro-risk-control">[eq:intro-risk-control]</a> with loss function $`\tilde{L}_i(\lambda) = \mathbbm{1}\left\{L_i(\lambda) > \alpha\right\}`$. Let
``` math
\hat{\lambda}_\beta = \inf\left\{\lambda\in \Lambda: \frac{1}{n+1}\sum_{i=1}^{n}\mathbbm{1}\left\{L_i(\lambda) > \alpha\right\} + \frac{1}{n+1}\le 1 - \beta\right\}.
```

<div id="thm:quantile-risk-control" class="prop">

**Proposition 4**. *In the setting of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, <a href="#eq:quantile_risk" data-reference-type="eqref" data-reference="eq:quantile_risk">[eq:quantile_risk]</a> is achieved.*

</div>

considers the high-probability control of a wider class of quantile-based risks which include the conditional value-at-risk (CVaR). It is unclear whether those more general risks can be controlled unconditionally. We leave this open problem for future research.

## Controlling multiple risks

Let $`L_{i}(\lambda; \gamma)`$ be a family of loss functions indexed by $`\gamma\in \Gamma`$ for some domain $`\Gamma`$ that may have infinitely many elements. A researcher may want to control $`\mathbb{E}[L_i(\lambda; \gamma)]`$ at level $`\alpha(\gamma)`$. Equivalently, we need to find an $`\hat{\lambda}`$ based on $`(X_i, Y_i)_{i=1}^{n}`$ such that
``` math
\label{eq:multiple-risk-goal}
    \sup_{\gamma\in \Gamma}\mathbb{E}\left[\frac{L_i(\hat{\lambda}; \gamma)}{\alpha(\gamma)}\right]\le 1.
```

Though the above worst-case risk is not an expectation, it can still be controlled. Towards this end, we define
``` math
\label{eq:lhat-multiple-risks}
    \hat{\lambda}= \sup_{\gamma \in \Gamma} \hat{\lambda}_{\gamma}, \text{ where }
    \hat{\lambda}_{\gamma} = \inf\left\{\lambda : \frac{1}{n+1}\sum_{i=1}^{n}L_i(\lambda; \gamma) + \frac{B}{n+1}\le \alpha(\gamma) \right\}.
```
Then the risk is controlled.

<div id="thm:multiple-risks" class="prop">

**Proposition 5**. *In the setting of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, <a href="#eq:multiple-risk-goal" data-reference-type="eqref" data-reference="eq:multiple-risk-goal">[eq:multiple-risk-goal]</a> is satisfied.*

</div>

## Adversarial risks

We next show how to control risks defined by adversarial perturbations. We adopt the same notation as Section <a href="#sec:multiple-risks" data-reference-type="ref" data-reference="sec:multiple-risks">4.3</a>. (Section 6.3) discusses the adversarial risk where $`\Gamma`$ parametrizes a class of perturbations of $`X_{n+1}`$, e.g., $`L_i(\lambda; \gamma) = L(X_i + \gamma, Y_i)`$ and $`\Gamma = \{\gamma: \|\gamma\|_{\infty}\le \epsilon\}`$. A researcher may want to find an $`\hat{\lambda}`$ based on $`(X_i, Y_i)_{i=1}^{n}`$ such that
``` math
\label{eq:adversarial-goal}
    \mathbb{E}[ \sup_{\gamma \in \Gamma} L_i(\lambda; \gamma)] \leq \alpha.
```

This can be recast as a conformal risk control problem by taking $`\tilde{L}_i(\lambda) = \sup_{\gamma \in \Gamma} L_i(\lambda; \gamma)`$. Then, the following choice of $`\lambda`$ leads to risk control:
``` math
\label{eq:lhat-adversarial}
    \hat{\lambda}= \inf\left\{\lambda : \frac{1}{n+1}\sum_{i=1}^{n}\tilde{L}_i(\lambda) + \frac{B}{n+1} \le \alpha \right\}.
```

<div id="thm:adversarial" class="prop">

**Proposition 6**. *In the setting of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, <a href="#eq:adversarial-goal" data-reference-type="eqref" data-reference="eq:adversarial-goal">[eq:adversarial-goal]</a> is satisfied.*

</div>

## U-risk control

For ranking and metric learning, considered loss functions that depend on two test points. In general, for any $`k > 1`$ and subset $`\mathcal{S}\subset \{1, \ldots, n+k\}`$ with $`|\mathcal{S}| = k`$, let $`L_\mathcal{S}(\lambda)`$ be a loss function. Our goal is to find $`\hat{\lambda}_k`$ based on $`(X_i, Y_i)_{i=1}^{n}`$ such that
``` math
\label{eq:U-risk}
\mathbb{E}\left[L_{\{n+1, \ldots, n+k\}}(\hat{\lambda}_k)\right]\le \alpha.
```
We call the LHS a U-risk since, for any fixed $`\hat{\lambda}_k`$, it is the expectation of an order-$`k`$ U-statistic. As a natural extension, we can define
``` math
\label{eq:lhatk}
\hat{\lambda}_k = \inf\left\{\lambda: \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda) + B \left(1 - \frac{(n!)^2}{(n+k)!(n-k)!}\right)\le \alpha\right\}.
```
Again, we define $`\hat{\lambda}_k = \lambda_{\max}`$ when the right-hand side is an empty set. Then we can prove the following result.

<div id="thm:U_risk_control" class="prop">

**Proposition 7**. *Assume that $`L_{\mathcal{S}}(\lambda)`$ is non-increasing in $`\lambda`$, right-continuous, and
``` math
L_{\mathcal{S}}(\lambda_{\max}) \le \alpha, \quad \sup_{\lambda}L_{\mathcal{S}}(\lambda) \le B < \infty \text{ almost surely}.
```
Then <a href="#eq:U-risk" data-reference-type="eqref" data-reference="eq:U-risk">[eq:U-risk]</a> is achieved.*

</div>

# Conclusion

This generalization of conformal prediction broadens its scope to new applications, as shown in Section <a href="#sec:examples" data-reference-type="ref" data-reference="sec:examples">3</a>. The mathematical tools developed in Section <a href="#sec:theory" data-reference-type="ref" data-reference="sec:theory">2</a>, Section <a href="#sec:extensions" data-reference-type="ref" data-reference="sec:extensions">4</a>, and the Appendix may be of independent technical interest, since they provide a new and more general language for studying conformal prediction along with new results about its validity.

# Acknowledgements

The authors thank Christopher Yeh for pointing out that the factor of 2 in Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> may not be tight, correcting an error in an earlier draft. The incorrect proposition stating that it was tight has been removed; the tightness of this factor remains an open question. The authors would like to thank Nicolas Christianson, Amit Kohli, Sherrie Wang, and Tijana Zrnić for comments on early drafts. A. A. would like to thank Ziheng (Tony) Wang for helpful conversations. A. A. is funded by the NSF GRFP and a Berkeley Fellowship. S. B. is supported by the NSF FODSI fellowship and the Simons institute. A. F. is partially funded by the NSF GRFP and MIT MLPDS.

# Monotonizing non-monotone risks

We next show that the proposed algorithm leads to asymptotic risk control for non-monotone risk functions when applied to a monotonized version of the empirical risk. We set the *monotonized empirical risk* to be
``` math
\widehat{R}^{\uparrow}_{n}(\lambda) = \underset{t \geq \lambda}{\sup}\;\widehat{R}_{n}(t),
```
then define
``` math
\label{eq:lhatplus}
    \hat{\lambda}^{\uparrow}_n = \inf\left\{\lambda:  \widehat{R}^{\uparrow}_{n}(\lambda) \le \alpha \right\}.
```

<div id="thm:monotonized" class="theorem">

**Theorem 3**. *Let the $`L_{i}(\lambda)`$ be right-continuous, i.i.d., bounded (both above and below) functions satisfying <a href="#eq:gF" data-reference-type="eqref" data-reference="eq:gF">[eq:gF]</a>. Then,
``` math
\underset{n \to \infty}{\lim}\mathbb{E}\Big[L_{n+1}\big(\hat{\lambda}^{\uparrow}_n\big)\Big]\le \alpha.
```*

</div>

Theorem <a href="#thm:monotonized" data-reference-type="ref" data-reference="thm:monotonized">3</a> implies that an analogous procedure to <a href="#eq:lhat" data-reference-type="ref" data-reference="eq:lhat">[eq:lhat]</a> also controls the risk asymptotically. In particular, taking
``` math
\tilde{\lambda}^{\uparrow} = \inf\left\{\lambda:  \widehat{R}^{\uparrow}_{n}(\lambda) + \frac{B}{n+1} \le \alpha \right\}
```
also results in asymptotic risk control (to see this, plug $`\tilde{\lambda}^{\uparrow}`$ into Theorem <a href="#thm:monotonized" data-reference-type="ref" data-reference="thm:monotonized">3</a> and see that the risk level is bounded above by $`\alpha-\frac{B}{n+1}`$). Note that in the case of a monotone loss function, $`\tilde{\lambda}^{\uparrow} = \hat{\lambda}`$. However, the counterexample in Proposition <a href="#prop:counterexample" data-reference-type="ref" data-reference="prop:counterexample">1</a> does not apply to $`\tilde{\lambda}^{\uparrow}`$, and it is currently unknown whether this procedure does or does not provide finite-sample risk control.

# Proofs

The proof of Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> uses the following lemma on the approximate continuity of the empirical risk.

<div id="lem:jump" class="lemma">

**Lemma 1** (Jump Lemma). *In the setting of Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a>, any jumps in the empirical risk are bounded, i.e.,
``` math
\sup_{\lambda}J\big(\widehat{R}_{n}, \lambda\big) \overset{a.s.}{\leq} \frac{B}{n}.
```*

</div>

<div class="proof">

*Proof of Jump Lemma, Lemma <a href="#lem:jump" data-reference-type="ref" data-reference="lem:jump">1</a>.* By boundedness, the maximum contribution of any single point to the jump is $`\frac{B}{n}`$, so
``` math
\exists \lambda :\; J\big(\widehat{R}_n, \lambda \big) > \frac{B}{n} \\ \Longrightarrow \exists \lambda :\; J(L_i,\lambda) > 0 \text{ and } J(L_j,\lambda) > 0 \text{ for some } i \neq j.
```
Call $`\mathcal{D}_i = \{ \lambda : J(L_i, \lambda) > 0 \}`$ the sets of discontinuities in $`L_i`$. Since $`L_i`$ is bounded monotone, $`\mathcal{D}_i`$ has countably many points. The union bound then implies that
``` math
\mathbb{P}\left(\exists \lambda : \;  J(\widehat{R}_n, \lambda) > \frac{B}{n} \right) \le \sum_{i\neq j}\mathbb{P}(\mathcal{D}_i \cap \mathcal{D}_j \neq \emptyset)
```
Rewriting each term of the right-hand side using tower property and law of total probability gives
``` math
\begin{aligned}
      \mathbb{P}\left( \mathcal{D}_i \cap \mathcal{D}_j \neq \emptyset \right) 
      &= \mathbb{E}\left[ \mathbb{P}\big( \mathcal{D}_i \cap \mathcal{D}_j \neq \emptyset \: \big| \: \mathcal{D}_j \big) \right] \\ 
      &\leq \mathbb{E}\left[ \sum\limits_{\lambda \in \mathcal{D}_j} \mathbb{P}\left( \lambda \in \mathcal{D}_i \; \Big| \; \mathcal{D}_j \right) \right] = \mathbb{E}\left[ \sum\limits_{\lambda \in \mathcal{D}_j} \mathbb{P}\left( \lambda \in \mathcal{D}_i\right) \right],
  
\end{aligned}
```
Where the second inequality is because the union of the events $`\lambda \in \mathcal{D}_j`$ is the entire sample space, but they are not disjoint, and the third equality is due to the independence between $`\mathcal{D}_i`$ and $`\mathcal{D}_j`$. Rewriting in terms of the jump function and applying the assumption $`\mathbb{P}\left( J(L_i, \lambda) > 0 \right) = 0`$,
``` math
\mathbb{E}\left[ \sum\limits_{\lambda \in \mathcal{D}_j} 
      \mathbb{P}\left( \lambda \in \mathcal{D}_i\right) \right] = \mathbb{E}\left[ \sum\limits_{\lambda \in \mathcal{D}_j} 
      \mathbb{P}\left( J(L_i, \lambda) > 0 \right) \right] = 0.
```
Chaining the above inequalities yields $`\mathbb{P}\left(\exists \lambda :   J(\widehat{R}_n, \lambda) > \frac{B}{n} \right) \leq 0`$, so  
$`\mathbb{P}\left(\exists \lambda :   J(\widehat{R}_n, \lambda) > \frac{B}{n} \right) = 0`$. ◻

</div>

<div class="proof">

*Proof of Theorem <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a>.* If $`L_i(\lambda_{\max})\ge \alpha - 2B/(n+1)`$, then $`\mathbb{E}[L_{n+1}(\hat{\lambda})]\ge \alpha - 2B/(n+1)`$. Throughout the rest of the proof, we assume that $`L_i(\lambda_{\max}) < \alpha - 2B/(n+1)`$. Define the quantity
``` math
\hat{\lambda}'' = \inf\left\{ \lambda : \widehat{R}_{n+1}(\lambda) + \frac{B}{n+1} \leq \alpha \right\}.
```
Since $`L_i(\lambda_{\max}) < \alpha - 2B/(n+1) < \alpha - B/ (n+1)`$, $`\hat{\lambda}''`$ exists almost surely. Deterministically, $`\frac{n}{n+1} \widehat{R}_n(\lambda) \leq \widehat{R}_{n+1}(\lambda)`$, which yields $`\hat{\lambda}\leq \hat{\lambda}''`$. Again since $`L_i(\lambda)`$ is non-increasing in $`\lambda`$,
``` math
\mathbb{E}\left[ L_{n+1}\big(\hat{\lambda}''\big) \right] \leq \mathbb{E}\left[ L_{n+1}\big(\hat{\lambda}\big) \right]
```
By exchangeability and the fact that $`\hat{\lambda}''`$ is a symmetric function of $`L_1, \ldots, L_{n+1}`$,
``` math
\mathbb{E}\left[ L_{n+1}\big(\hat{\lambda}''\big) \right] = \mathbb{E}\left[ \widehat{R}_{n+1}\big(\hat{\lambda}''\big) \right]
```

For the remainder of the proof we focus on lower-bounding $`\widehat{R}_{n+1}\big(\hat{\lambda}''\big)`$. We begin with the following identity:
``` math
\alpha = \widehat{R}_{n+1}\big(\hat{\lambda}''\big) + \frac{B}{n+1} - \Big(\widehat{R}_{n+1}\big(\hat{\lambda}''\big) + \frac{B}{n+1} -\alpha\Big).
```
Rearranging the identity,
``` math
\widehat{R}_{n+1}\big(\hat{\lambda}''\big) = \alpha - \frac{B}{n+1} + \Big(\widehat{R}_{n+1}\big(\hat{\lambda}''\big) + \frac{B}{n+1} -\alpha\Big).
```
Using the Jump Lemma to bound $`\Big(\widehat{R}_{n+1}\big(\hat{\lambda}''\big) + \frac{B}{n+1} -\alpha\Big)`$ below by $`-\frac{B}{n+1}`$ gives
``` math
\widehat{R}_{n+1}\big(\hat{\lambda}''\big) \geq \alpha - \frac{2B}{n+1}.
```
Finally, chaining together the above inequalities,
``` math
\mathbb{E}\bigg[ L_{n+1}(\hat{\lambda}) \bigg] \geq \mathbb{E}\bigg[ \widehat{R}_{n+1}(\hat{\lambda}'') \bigg] \geq \alpha - \frac{2B}{n+1}.
```
 ◻

</div>

<div class="proof">

*Proof of Proposition <a href="#prop:counterexample" data-reference-type="ref" data-reference="prop:counterexample">1</a>.* Without loss of generality, we assume $`B=1`$. Assume $`\hat{\lambda}`$ takes values in $`[0,1]`$ and $`\alpha \in (1/(n+1), 1)`$. Let $`p\in (0, 1)`$, $`N`$ be any positive integer, and $`L_{i}(\lambda)`$ be i.i.d. right-continuous piecewise constant (random) functions with
``` math
L_{i}(N/N) = 0, \quad \left(L_{i}(0/N), L_{i}(1/N), \ldots, L_{i}((N - 1)/N)\right)\stackrel{i.i.d.}{\sim}\text{Ber}(p).
```
By definition, $`\hat{\lambda}`$ is independent of $`L_{n+1}`$. Thus, for any $`j = 0, 1, \ldots, N-1`$,
``` math
\left\{L_{n+1}(\hat{\lambda})\mid \hat{\lambda}= j/N\right\} \sim \text{Ber}(p), \quad \left\{L_{n+1}(\hat{\lambda})\mid \hat{\lambda}= 1\right\} \sim \delta_{0}.
```
Then,
``` math
\mathbb{E}\Big[ L_{n+1}\big(\hat{\lambda}\big) \Big] = p\cdot\mathbb{P}(\hat{\lambda}\neq 1)\\
```
Note that
``` math
\hat{\lambda}\neq 1 \Longleftrightarrow \min_{j \in \{0, \ldots, N-1\}}\frac{1}{n+1}\sum_{i=1}^{n}L_{i}(j / N) \le \alpha - \frac{1}{n+1}.
```
Since $`\alpha > 1 / (n + 1)`$,
``` math
\begin{aligned}
  \mathbb{P}(\hat{\lambda}\neq 1) = 1 - \mathbb{P}(\hat{\lambda}= 1) &= 1-\mathbb{P}\left(\text{for all }j, \text{ we have }\frac{1}{n+1}\sum_{i=1}^{n}L_{i}(j / N) > \alpha - \frac{1}{n+1} \right)\\
  & = 1 - \left(\sum\limits_{k=\lceil(n+1)\alpha\rceil}^n {n\choose k} p^k(1-p)^{(n-k)}\right)^N\\
  & = 1 - \left(1-\mathrm{BinoCDF}\big(n,p,\lceil(n+1)\alpha\rceil-1\big)\right)^N\\
\end{aligned}
```
As a result,
``` math
\mathbb{E}\Big[ L_{n+1}\big(\hat{\lambda}\big) \Big] = p \Bigg(1 - \left(1-\mathrm{BinoCDF}\big(n,p,\lceil(n+1)\alpha\rceil-1\big)\right)^N\Bigg).
```
Now let $`N`$ be sufficiently large such that
``` math
\Bigg(1 - \left(1-\mathrm{BinoCDF}\big(n,p,\lceil(n+1)\alpha\rceil-1\big)\right)^N\Bigg) > p.
```
Then
``` math
\mathbb{E}\Big[ L_{n+1}\big(\hat{\lambda}\big) \Big] > p^2
```
For any $`\alpha > 0`$, we can take $`p`$ close enough to $`1`$ to render the claim false. ◻

</div>

<div class="proof">

*Proof of Theorem <a href="#thm:monotonized" data-reference-type="ref" data-reference="thm:monotonized">3</a>.* Define the *monotonized population risk* as
``` math
R^{\uparrow}(\lambda) = \underset{t \geq \lambda}{\sup}\; \mathbb{E}\Big[ L_{n+1}(t) \Big]
```
Note that the independence of $`L_{n+1}`$ and $`\hat{\lambda}^{\uparrow}_n`$ implies that for all $`n`$,
``` math
\mathbb{E}\Big[ L_{n+1}\big( \hat{\lambda}^{\uparrow}_n \big) \Big] \le \mathbb{E}\Big[ R^{\uparrow}\big( \hat{\lambda}^{\uparrow}_n \big) \Big].
```
Since $`R^{\uparrow}`$ is bounded, monotone, and one-dimensional, a generalization of the Glivenko-Cantelli Theorem given in Theorem 1 of  gives that uniformly over $`\lambda`$,
``` math
\underset{n \to \infty}{\lim} \sup_{\lambda}|\widehat{R}_{n}(\lambda) - R(\lambda)| \overset{a.s.}{\to} 0 .
```
As a result,
``` math
\underset{n \to \infty}{\lim} \sup_{\lambda}|\widehat{R}^{\uparrow}_{n}(\lambda) - R^{\uparrow}(\lambda)| \overset{a.s.}{\to} 0,
```
which implies that
``` math
\underset{n \to \infty}{\lim} |\widehat{R}^{\uparrow}_{n}(\hat{\lambda}^{\uparrow}) - R^{\uparrow}(\hat{\lambda}^{\uparrow})| \overset{a.s.}{\to} 0.
```
By definition, $`\widehat{R}^{\uparrow}(\hat{\lambda}^{\uparrow})\le \alpha`$ almost surely and thus this directly implies
``` math
\underset{n \to \infty}{\limsup}~  R^{\uparrow}\big(\hat{\lambda}^{\uparrow}_n\big) \leq \alpha\quad \text{a.s.}.
```
Finally, since for all $`n`$, $`R^{\uparrow}\big(\hat{\lambda}^{\uparrow}_n\big) \leq B`$, by Fatou’s lemma,
``` math
\underset{n \to \infty}{\lim}\mathbb{E}\Big[ L_{n+1}\big( \hat{\lambda}^{\uparrow}_n \big) \Big] \le \underset{n \to \infty}{\limsup}~\mathbb{E}\Big[ R^{\uparrow}\big( \hat{\lambda}^{\uparrow}_n \big) \Big] \le \mathbb{E}\Big[ \underset{n \to \infty}{\limsup}~ R^{\uparrow}\big( \hat{\lambda}^{\uparrow}_n \big) \Big] \leq \alpha.
```
 ◻

</div>

<div class="proof">

*Proposition <a href="#thm:upper-bound-weighted" data-reference-type="ref" data-reference="thm:upper-bound-weighted">2</a>.* Let
``` math
\hat{\lambda}' = \inf\left\{ \lambda : \frac{\sum_{i=1}^{n+1}w(X_i)L_i(\lambda)}{\sum_{i=1}^{n+1}w(X_i)} \leq \alpha \right\}.
```
Since $`\inf_{\lambda} L_i(\lambda) \le \alpha`$, $`\hat{\lambda}'`$ exists almost surely. Using the same argument as in the proof of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>, we can show that $`\hat{\lambda}'\le \hat{\lambda}(X_{n+1})`$. Since $`L_{n+1}(\lambda)`$ is non-increasing in $`\lambda`$,
``` math
\mathbb{E}[L_{n+1}(\hat{\lambda}(X_{n+1}))]\le \mathbb{E}[L_{n+1}(\hat{\lambda}')].
```
Let $`E`$ be the multiset of loss functions $`\{ (X_1, Y_1), \ldots, (X_{n+1}, Y_{n+1}) \}`$. Then $`\hat{\lambda}'`$ is a function of $`E`$, or, equivalently, $`\hat{\lambda}'`$ is a constant conditional on $`E`$. Lemma 3 of implies that
``` math
(X_{n+1}, Y_{n+1})\mid E\sim \sum_{i=1}^{n+1}\frac{w(X_{i})}{\sum_{j=1}^{n+1}w(X_j)}\delta_{(X_j, Y_j)}\Longrightarrow L_{n+1} \mid E \sim \sum_{i=1}^{n+1}\frac{w(X_{i})}{\sum_{j=1}^{n+1}w(X_j)}\delta_{L_{i}}
```
where $`\delta_{z}`$ denotes the Dirac measure at $`z`$. Together with the right-continuity of $`L_{i}`$, the above result implies
``` math
\mathbb{E}\left[L_{n+1}(\hat{\lambda}')\mid E\right] = \frac{\sum_{i=1}^{n+1}w(X_i)L_{i}(\hat{\lambda}')}{\sum_{i=1}^{n+1}w(X_i)} \leq \alpha.
```
The proof is then completed by the law of total expectation. ◻

</div>

<div class="proof">

*Proposition <a href="#thm:tv-bound" data-reference-type="ref" data-reference="thm:tv-bound">3</a>.* Define the vector $`Z'=(Z_1', \ldots, Z_n', Z_{n+1})`$, where $`Z_i' \overset{i.i.d.}{\sim} \mathcal{L}(Z_{n+1})`$ for all $`i \in [n]`$. Let
``` math
\epsilon= \sum_{i=1}^{n}\mathrm{TV}(Z_i, Z_i').
```
By sublinearity,
``` math
\label{eq:tv-bounded-vector}
    \mathrm{TV}(Z, Z') \leq \epsilon.
```
It is a standard fact that <a href="#eq:tv-bounded-vector" data-reference-type="eqref" data-reference="eq:tv-bounded-vector">[eq:tv-bounded-vector]</a> implies
``` math
\underset{f \in \mathcal{F}_{\mathbbm{1}}}{\sup} \left| \mathbb{E}[ f(Z) ] - \mathbb{E}[ f(Z') ] \right| \leq \epsilon,
```
where $`\mathcal{F}_{\mathbbm{1}} = \{f: \mathcal{Z}\mapsto [0, 1]\}`$. Let $`\ell : \mathcal{Z} \times \Lambda \to [0,B]`$ be a bounded loss function. Furthermore, let $`g(z) = \ell(z_{n+1}; \hat{\lambda}(z_1, \ldots, z_n))`$. Since $`g(Z) \in [0, B]`$,
``` math
| \mathbb{E}[g(Z)] - \mathbb{E}[g(Z')] | \leq B\epsilon.
```
Furthermore, since $`Z_1', \ldots, Z_{n+1}'`$ are exchangeable, we can apply Theorems <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a> and <a href="#thm:lower-bound" data-reference-type="ref" data-reference="thm:lower-bound">2</a> to $`\mathbb{E}[g(Z')]`$, recovering
``` math
\alpha - \frac{2B}{n+1} \leq \mathbb{E}[g(Z')] \leq \alpha.
```
A final step of triangle inequality implies the result:
``` math
\alpha - \frac{2B}{n+1} - B\epsilon \leq \mathbb{E}[g(Z)] \leq \alpha + B\epsilon.
```
 ◻

</div>

<div class="proof">

*Proposition <a href="#thm:quantile-risk-control" data-reference-type="ref" data-reference="thm:quantile-risk-control">4</a>.* It is left to prove that $`\tilde{L}_i(\lambda)`$ satisfies the conditions of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>. It is clear that $`\tilde{L}_i(\lambda)\le 1`$ and $`\tilde{L}_i(\lambda)`$ is non-increasing in $`\lambda`$ when $`L_i(\lambda)`$ is. Since $`L_i(\lambda)`$ is non-increasing and right-continuous, for any sequence $`\lambda_{m}\downarrow \lambda`$,
``` math
L_{i}(\lambda_{m})\uparrow L_i(\lambda)\Longrightarrow \mathbbm{1}\left\{L_{i}(\lambda_{m}) > \alpha\right\} \rightarrow \mathbbm{1}\left\{L_{i}(\lambda) > \alpha\right\}.
```
Thus, $`\tilde{L}_i(\lambda)`$ is right-continuous. Finally, $`L_i(\lambda_{\max}) \le \alpha`$ implies $`\tilde{L}_i(\lambda_{\max}) = 0 \le 1 - \beta`$. ◻

</div>

<div class="proof">

*Proposition <a href="#thm:multiple-risks" data-reference-type="ref" data-reference="thm:multiple-risks">5</a>.* Examining <a href="#eq:lhat-multiple-risks" data-reference-type="eqref" data-reference="eq:lhat-multiple-risks">[eq:lhat-multiple-risks]</a>, for each $`\gamma \in \Gamma`$, we have
``` math
\mathbb{E}\left[L(\hat{\lambda}, \gamma)\right] \leq \mathbb{E}\left[L(\hat{\lambda}_{\gamma}, \gamma)\right] \leq \alpha(\gamma).
```
Thus, dividing both sides by $`\alpha(\gamma)`$ and taking the supremum, we get that $`\sup_{\gamma \in \Gamma} \mathbb{E}\left[\frac{L(\hat{\lambda}, \gamma)}{\alpha(\gamma)}\right] \leq 1`$, and the worst-case risk is controlled. ◻

</div>

<div class="proof">

*Proposition <a href="#thm:adversarial" data-reference-type="ref" data-reference="thm:adversarial">6</a>.* Because $`L_i(\lambda, \gamma)`$ is bounded and monotone in $`\lambda`$ for all choices of $`\gamma`$, it is also true that $`\tilde{L_i}(\lambda)`$ is bounded and monotone. Furthermore, the pointwise supremum of right-continuous functions is also right-continuous. Therefore, the $`\tilde{L_i}`$ satisfy the assumptions of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a>. ◻

</div>

<div class="proof">

*Proposition <a href="#thm:U_risk_control" data-reference-type="ref" data-reference="thm:U_risk_control">7</a>.* Let
``` math
\hat{\lambda}'_k = \inf\left\{\lambda: \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n+k\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda)\le \alpha\right\}.
```
Since $`L_\mathcal{S}(\lambda_{\max})\le \alpha`$, $`\hat{\lambda}'_k`$ exists almost surely. Since $`L_{\mathcal{S}}(\lambda)\le B`$, we have
``` math
\begin{aligned}
    &\frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n+k\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda)\\
    & \le \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda) + B \cdot \sum_{\mathcal{S}\cap \{n+1, \ldots, n+k\}\neq \emptyset, |\mathcal{S}|=k}1\\
    & =   \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda) + B \left(1 - \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n\}, |\mathcal{S}| = k}1\right)\\
    & = \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda) + B \left(1 - \frac{(n!)^2}{(n+k)!(n-k)!}\right).
  
\end{aligned}
```
Since $`L_{\mathcal{S}}(\lambda)`$ is non-increasing in $`\lambda`$, we conclude that $`\hat{\lambda}'_k \le \hat{\lambda}_k`$ if the right-hand side of <a href="#eq:lhatk" data-reference-type="eqref" data-reference="eq:lhatk">[eq:lhatk]</a> is not empty; otherwise, by definition, $`\hat{\lambda}'_k \le \lambda_{\max} = \hat{\lambda}_k`$. Thus, $`\hat{\lambda}'_k \le \hat{\lambda}_k`$ almost surely. Let $`E`$ be the multiset of loss functions $`\{L_{\mathcal{S}}: \mathcal{S}\subset \{1, \ldots, n+k\}, |\mathcal{S}| = k\}`$. Using the same argument in the end of the proof of Theorem <a href="#thm:upper-bound" data-reference-type="ref" data-reference="thm:upper-bound">1</a> and the right-continuity of $`L_{\mathcal{S}}`$, we can show that
``` math
\mathbb{E}\left[L_{\{n+1, \ldots, n+k\}}(\hat{\lambda}'_k)\mid E\right] = \frac{k!n!}{(n+k)!}\sum_{\mathcal{S}\subset \{1, \ldots, n+k\}, |\mathcal{S}| = k}L_{\mathcal{S}}(\lambda)\le \alpha.
```
The proof is then completed by the law of iterated expectation. ◻

</div>
