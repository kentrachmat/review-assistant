# Incremental Randomized Smoothing Certification

## Abstract

As Large Language Models (LLMs) are deployed more widely, customization with respect to vocabulary, style, and character becomes more important. In this work, we introduce model arithmetic, a novel inference framework for composing and biasing LLMs without the need for model (re)training or highly specific datasets. In addition, the framework allows for more precise control of generated text than direct prompting and prior controlled text generation (CTG) techniques. Using model arithmetic, we can express prior CTG techniques as simple formulas and naturally extend them to new and more effective formulations. Further, we show that speculative sampling, a technique for efficient LLM sampling, extends to our setting. This enables highly efficient text generation with multiple composed models with only marginal overhead over a single model. Our empirical evaluation demonstrates that model arithmetic allows fine-grained control of generated text while outperforming state-of-the-art on the task of toxicity reduction. We release an open source easy-to-use implementation of our framework at <https://github.com/eth-sri/language-model-arithmetic>.

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