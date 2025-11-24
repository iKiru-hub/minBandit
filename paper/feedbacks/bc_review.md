# Reviewer 1
The manuscript presents a bio-inspired 2-layer neural network optimized using an evolutionary algorithm to effectively perform multi-arm bandit tasks in environments with variability in reward distributions. The proposed model provides comparable performance to some typical algorithms for solving multi-arm bandit tasks, such as Thompson Sampling and UCB, while replicating some features of biological plasticity rules. The authors provide a comprehensive overview of previous attempts in the same direction and discuss their findings within that context. They also provide a fair description of the limitations of their current model. The overall approach is interesting and valuable to researchers interested in learning at the intersection of biological and normative approaches. However, I have some comments, for example, regarding the model design and its parameters that need to be clarified further.
 
---
- [x] (1)
I believe the introduction of the manuscript can benefit from richer connections to existing literature and, in particular, further emphasis on the importance of the key principle of reward maximization as a fundamental aspect of cognition.
I would suggest, authors expand the very first sentence, “The ability … is a fundamental aspect of cognition,” a little bit, and mention finer-grained aspects of cognition. I’d suggest, at least, mentioning and citing core decision-making (Dayan and Daw, 2008), perception (Safavi and Dayan, 2022), and emotion processing (Bach and Dayan, 2017).
 
 ---
Sensitivity to model parameters:

- [x] I could not find any information about some model parameters, for instance, how the size and the time constants of each layer are chosen, and how sensitive the results are to these choices.
    |> time constants are evolved, no much sensitivity given the population distribution

- [x] The activation functions (phi_v and phi_u) are set in a very specific form (as discussed in Appendix 5.1). It would be helpful to discuss the intuition behind this choice and also how sensitive the results are to this specific form. For example, if one chooses a different activation function, such as ReLU, would the results change?
    |> maintaining flexibility in the definition of a useful curve shape, thus acting as two basis functions. an approximation of ReLU can be obtained by certain parameters choices: it has not evolved so.

- [x] The timings of the option selection process are set manually (2s and 5s), but the logic behind these choices and the sensitivity of the results to them are not discussed.
    |> duration times are evolved as well
    |> sensitivity to explore by an additional plot in the appendix probing the effects of synchronicity of duration values (pre vs post)
 
 ---
- [x] I am a bit confused about the role of I_ext. One would naturally expect that in a biologically plausible model, the external input reflects input from the environment, for instance, the received reward, but it does not seem to be the case here. Instead, the reward only appears in the learning rule without discussing how it is given to the model.
    |> I_ext is simply meant to initiate the decision process, more like an attention mechanism than a sensory input (the task is one, no ambiguity and no further information needed)
    |> reward is delivered as a boolean/binary value, much like absence/presence of positive feedback after an action
 
 ---
- [x] (4)
The details of optimization in different environments were not clear. Is the model optimized on all 4 environments simultaneously, or would the authors have one model for each environment? In the case of the former, please provide more details about how it is done. In other words, are NSA models presented in Table 1 the same, or do they have different parameters?
    |> each model is evolved over all environment
 
 ---
- [x] It would be helpful to provide a more intuitive understanding of why layer U acts as a memory trace, for example, is it related to feedback from V?
    |> true, it is not really memory but mostly a representation of the option that is mapped to outside. it is however unnecessary

- [x] Also, please clarify whether the weights from V to U are fixed to 1 during the optimization or if they are initialized as 1 but can change.
    |> they are initialized and fixed to 1, what's relevant is their activation value so the weight would be an additional but unnecessary degree of freedom
 
 ---
- [x] (6)
The consensus mechanism between U and V seems a bit artificial; is there any biological support for this?
    |> alignment of the OFC - ACC
 
 ---
Minor comments:
1. It would be helpful to cite different sections of the appendix more thoroughly in the main text. For instance, instead of referring to the whole Appendix 5, its sections get cited in relevant places.
2. Some typos:
- [x] Page 5: extra comma in “that is, the bandits”
- [x] Page 6 and 11: “In more detail”

## references
Dayan, Peter, and Nathaniel D. Daw. "Decision theory, reinforcement learning, and the brain." Cognitive, Affective, & Behavioral Neuroscience 8.4 (2008): 429-453.
 
Safavi, Shervin, and Peter Dayan. "Multistability, perceptual value, and internal foraging." Neuron 110.19 (2022): 3076-3090.
 
Bach, Dominik R., and Peter Dayan. "Algorithms for survival: a comparative perspective on emotions." Nature Reviews Neuroscience 18.5 (2017): 311-319.

---
---

# Reviewer 2
In this paper, the authors implement a simple recurrent firing rate neural circuit model for solving multi-armed banded (MAB) tasks. The authors show that after fitting the parameters with evolutionary search, the model is able to perform competitively with standard regret minimization algorithms, such as Thompson sampling and upper confidence bound (UCB).
 
Overall the paper is well-written and motivated, and the math is rigorous, although I do make some minor suggestions about correcting language and notation. I believe the work is an interesting demonstration of how a simple biologically plausible circuit can solve MAB tasks and can serve as a starting point for investigating the underlying neural mechanisms. I believe the paper will be suitable for publication after my comments are addressed.
 
--
**Major comments**
 
- [x] (0)
My main concern is that the baselines (Thompson sampling, UCB, epsilon greedy) may not be given a fair chance here.
 
For one, I couldn't find any details about how exactly they were implemented. For Thompson sampling, did the authors use Bayesian filtering to compute the posterior for each arm, e.g. using a Beta-Binomial, and then sample from those posteriors? For UCB, did the authors use the same posteriors to compute the confidence intervals? How were the confidence intervals defined? What was the scaling parameter between the value and the confidence interval? For epsilon greedy, what was epsilon? These details should be clarified somewhere.
 
Relatedly, did those models have any free parameters and how are they fit? This is crucial – the author’s model has many free parameters, which raises the possibility that that improved performance may be at least in part due to overfitting to this domain. In contrast, the baselines have very few parameters and are very generic, applicable to domains as diverse as board games and video games. I believe the authors should at least fit the free parameters of the baselines in the same way as they fit the parameters of their model.
 
---
**Minor comments**
 
---
(1) Introduction
 
- [x] The introduction has many paragraphs that consist of only one or two sentences. Would recommend grouping them to streamline the exposition.
 
- [x] {**} “In fact, these methods can achieve state-of-the-art performance” – which methods do you mean by “these methods”?
    |> bio-inspired ones
 
- [x] “In addition, bioinspired models enhance algorithmic interpretability by clarifying the functional relationships between internal components” – I would disagree, biologically plausible models are often less interpretable algorithmically, as it is difficult to tell what each neuron / set of neurons / set of synapses is doing/learning.
    |> partially. what I meant was that the architecture has more specialized features to which is easier (and costructed as such) to label and map to neural elements. as for the learning, it is true that the possibly more degrees of freedom and complex relationship makes more opaque the overall state-space landscape.

- [x] “Although other approaches such as Bayesian learning can demonstrate optimal performance
and match human data well [16], they are more difficult to relate to neuronal dynamics.“ – again I would disagree. Computations from purely normative models can still be mapped to brain regions and circuits. Classic example: TD reward prediction errors and dopaminergic neurons in the midbrain (Schultz et al 1996 science). Example that’s very relevant to this study: mapping of Thompson sampling and UCB and their hybrid to different brain regions (Tomov et al 2020 nature communications).
    |> true, i should restricted the semantic breadth of my statement --> deleted
 
 ---
2.1 Binomial MAB problem
 
- [x] “with an associated reward distribution” – more appropriately, “associated reward probabilities”. The distribution is the Bernoulli distribution with the probabilities as parameters
 
“and the policy as a function that returns a selected arm π(ht ) = at“ – this seems to imply that only deterministic policies are considered, which is inconsistent with Thompson sampling and epsilon greedy, and the author’s own formulation in the previous sentence: “the policy is often defined as a *distribution* over actions“. More generally: at ~ π(ht ).
 
“Formally, given defined a function r(π) that returns the expected reward while following policy π,” – why introduce confusing new notation when you can simply reuse the p’s for the arms and index according to the action chosen by π
 
---
2.2 Neural Selection Agreement model (NSA)
 
- [x] “The first, U, represents the memory traces of the K available options ( that is,, the bandits)“ – unnecessary italics and double ,
 
- [x] “while the second, V, encodes their values ac- cording to current policy“ – “according to *the* current policy”
 
- [x] Figure 1: missing tilde ~ over W_UV
    |> actually no, in the figure it is represented the *connectivity*, while in the system of equations it is used the *surrogate* connectivity by means of the gating function
 
- [x] “More in detail, the weight matrix WV U is simply made of 1s,” – the diagonal of the weight matrix
 
- [x] “The function Φv is defined as the weighted sum of a generalized sigmoid and a Gaussian” – I know the authors visualized these in the results but it would be super useful to have a schematic somewhere here
 
---
2.2.1 Option selection
 
- [x] “After a fixed time ∼ 2s, the second phase begins” – how do you solve the ODE? What is the time delta, if you’re doing a discrete approximation?
   |> forward Euler algorithm. just step the dynamics.
 
- [x] argmaxk {v} – how are ties broken? This seems critical since all the weights are initialized uniformly.
    |> first result
 
- [x] “Lastly, the structure of the option selection process resembles the prefrontal circuitry, as the choices emerge from the state sampling of the network following a period of autonomous neural activity.” – this analogy seems a bit tenuous, I would recommend either elaborating more and actually visualizing the state sampling and stabilization, or removing it.
    |> *todo*
 
 ---
2.3 Learning
 
- [x] “a reward R ∈ {0, 1} with probability pk” – this is a little imprecise, it returns reward R = 1 with probability pk
 
- [x] "In particular, these characteristics can be combined to define mechanisms of synapse-type specific plasticity as a function of current synaptic strength" - can you elaborate a little bit more and also on the next sentence? In general, throughout the paper the authors draw analogies with known circuit motifs in the brain without elaborating (including an abstract). I would recommend either removing them or elaborating a little more.

- [x] Algorithm 1: "Let system evolve through population coupling according to 2.2;" - how is that different mechanistically from "Update populations u, v according to 2.2;" a few lines above? If it's the same thing I would recommend using the exact same phrasing, to avoid confusion.

---
3 Experiments

- [x] "The NSA model has been tested" - unusual past perfect tense here and everywhere else. Usually this is used to denote things that have been done in the literature prior to this study. I recommend switching to past simple "mode was tested"

- [x] "At the end of each trial i it is drawn a new distribution" - grammar

- [x] "the target distribution keep changing" - keeps

- [x] Figure 2: inconsistent notation with the text - KAB vs MAB

---
3.2 Evolution search

- [x] "Evolution search" - this terminology is somewhat unusual, I believe "evolutionary search" is more standard, and in this case the author appeared to be using a genetic algorithm, which is a particular kind of evolutionary search.

- [x] "The most relevant parameters resulted to be those concerning directly neuronal activity and learning." - I would rephrase this sentence

- [x] "The neural response functions are shown in 3d" - isn't it 3c? Also I recommend describing the results in the same order in the text and in the figure panels (a,b,c,d).

- [x] "Overall, our NSA model displays a solid performance over all environments" - solid sounds a little informal, maybe competitive? Also how do we know if these differences are significant? Some statistical tests would be helpful
    |> *todo*

- [ ] Figure 4b, top panel: how is the reward on the first trial so close to 1? Even when there's clearly several options? Shouldn't all models be basically at chance?

- [ ] "The results reported how all models are capable of robust performance in the first trial even in the presence of high uncertainty" - same thing, how is that possible if the models don't know anything about any of the arms at this point? This raises questions about the methodology and the rest of the results.

---

mention:
- [ ] testing distribution for different environments
- [ ] redo entropy plot
