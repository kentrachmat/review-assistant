# Conformal Risk Control

## Abstract

Residual neural networks are state-of-the-art deep learning models. Their continuous-depth analog, neural ordinary differential equations (ODEs), are also widely used. Despite their success, the link between the discrete and continuous models still lacks a solid mathematical foundation. In this article, we take a step in this direction by establishing an implicit regularization of deep residual networks towards neural ODEs, for nonlinear networks trained with gradient flow. We prove that if the network is initialized as a discretization of a neural ODE, then such a discretization holds throughout training. Our results are valid for a finite training time, and also as the training time tends to infinity provided that the network satisfies a Polyak-Łojasiewicz condition. Importantly, this condition holds for a family of residual networks where the residuals are two-layer perceptrons with an overparameterization in width that is only linear, and implies the convergence of gradient flow to a global minimum. Numerical experiments illustrate our results.

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
