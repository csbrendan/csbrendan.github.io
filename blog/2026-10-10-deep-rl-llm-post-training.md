# From Deep RL to LLM Post-Training: Lessons from My First Foray

I've been fine-tuning large language models for years, using techniques such as supervised fine-tuning (SFT), preference optimization, and Direct Preference Optimization (DPO). But recently, I've been making my first real foray into reinforcement learning (RL) for LLM post-training, particularly the newer techniques used to improve reasoning and agentic behavior. What has made this journey especially interesting is that many of the ideas now driving LLM post-training are closely related to concepts I first encountered studying deep reinforcement learning: exploration, reward design, policy optimization, credit assignment, and the fundamental difference between imitating successful behavior and learning through experience.

The algorithms and infrastructure have evolved considerably, but the underlying questions are surprisingly familiar.

## Supervised Learning Is Still Imitation

One of the most useful ways to think about supervised fine-tuning is as behavior cloning. In traditional reinforcement learning, behavior cloning involves training a policy to imitate demonstrations from an expert. Rather than discovering a strategy through interaction with an environment, the model learns to reproduce actions that appear in a dataset. SFT works in much the same way for language models. We provide examples of desired inputs and outputs, and the model learns to assign higher probability to the demonstrated responses.

This approach can be extraordinarily effective, particularly when the task is well-defined and the demonstrations are high quality. In fact, one of my first research projects involving clinical prescribing decisions illustrated just how far supervised learning alone can take us. In that project, I was investigating whether reinforcement learning could improve a language model's ability to make safe, guideline-concordant prescription decisions. The initial plan was to establish a supervised baseline and then use reinforcement learning with verifiable rewards to improve clinical decision-making.

But after fine-tuning and evaluating the supervised model, an important result emerged: SFT had already captured much of the performance available under our evaluation framework. We could not demonstrate sufficient remaining headroom to justify the additional complexity of RL. That was a useful lesson. Before investing in reinforcement learning, we need to establish that there is something meaningful left for the policy to discover.

If supervised learning already produces strong performance on the task, adding RL does not automatically make the system better. It may simply introduce optimization complexity, additional compute requirements, and opportunities for reward exploitation. Sometimes the correct research conclusion is that RL is not yet warranted. That experience helped shape how I approached my subsequent experiments.

## Where Reinforcement Learning Becomes Interesting

The limitation of behavior cloning becomes more apparent when the correct behavior cannot be completely specified through static examples. This is especially true for multi-step reasoning and agentic workflows. Consider an agent that must gather information, decide whether it has enough evidence, ask clarifying questions, use tools, and eventually commit to a decision. The appropriate next action depends on what happened earlier in the interaction.

A successful demonstration can show the agent what to do in a particular situation, but it cannot necessarily teach the agent how to recover from unfamiliar states or determine which actions will produce better outcomes. This is where reinforcement learning becomes particularly interesting. Instead of merely imitating successful trajectories, the model can interact with an environment, receive feedback, and optimize its behavior based on the outcomes of those interactions.

The distinction is fundamental: Supervised learning asks, "What would the expert do here?" Reinforcement learning asks, "What actions actually lead to better outcomes?"

For an LLM agent, those actions might involve asking another question, retrieving a document, invoking a tool, declining to proceed, or producing a final answer. The objective is no longer simply to reproduce a reference response. It is to learn a policy that performs well across a distribution of interactions.

## RLHF, DPO, and the Return of Online RL

Reinforcement learning has been part of language model post-training for some time. Reinforcement Learning from Human Feedback (RLHF) introduced a framework for improving language model behavior using preference information, often by training a reward model and optimizing a policy against it. Later, techniques such as Direct Preference Optimization offered a simpler way to learn from preference comparisons without explicitly performing an online RL optimization loop.

I've worked with approaches like DPO for years, but the current wave of reasoning-model post-training has renewed my interest in reinforcement learning itself. In particular, methods such as Group Relative Policy Optimization (GRPO) have made it practical to optimize language models using rewards that can be evaluated programmatically. This is often described as reinforcement learning with verifiable rewards, or RLVR. For tasks with objectively checkable outcomes, the reward signal can come from a verifier rather than a learned human-preference model.

Mathematical correctness, code execution, structured-output validity, and certain constrained decision-making tasks all offer opportunities for this approach. But there is an important distinction between having a reward function and having a useful learning signal. A reward may be perfectly computable yet still fail to encourage the behavior we actually want. That became one of the central lessons of my own experiments.

## My Experiments with Multi-Turn Clinical Agents

Building on the earlier prescribing research, I began exploring a more interactive clinical decision-making environment. Rather than asking the model to make a single prescribing decision from a complete patient description, the environment required the agent to decide whether additional information was needed before acting. The agent could ask clarifying questions, gather information, and eventually choose an appropriate action, such as approving a request, modifying it, or escalating for clinical review.

This introduced a genuine sequential decision problem. Asking a question could improve the final decision by resolving uncertainty, but it also carried a cost. Asking too few questions could lead to unsupported or unsafe decisions. Asking too many could create unnecessary friction without improving the outcome. The agent therefore needed to learn not simply which final decision was correct, but when it had gathered enough evidence to make that decision.

This distinction matters because a supervised model can learn to imitate the pattern of asking questions without necessarily learning the underlying value of information. It might ask questions because similar questions appeared in training demonstrations. Or it might prematurely commit to a decision because its supervised examples disproportionately favored immediate answers. Neither behavior necessarily reflects an optimal policy.

The RL environment offered a way to evaluate the consequences of those choices.

## Reward Design Is Harder Than It Looks

One of the biggest lessons from this work has been how sensitive reinforcement learning is to reward design. At first glance, the objective seems straightforward. Reward correct clinical decisions. Penalize unsafe decisions. Add a small cost for unnecessary questions. But the interaction between these terms can fundamentally change what the agent learns. If questions are too expensive, the model may learn to avoid gathering information, even when clarification is clinically important.

If questions are effectively free, the agent may learn to ask repeatedly because information-gathering is safer than committing to a decision. If escalation receives an overly favorable reward, the model may discover that escalating everything is a reliable way to avoid penalties. In each case, the policy can improve its numerical reward while moving away from the behavior we intended. This is the familiar reward-hacking problem from classical reinforcement learning, appearing in a new setting.

In my experiments, even relatively small adjustments to the cost of asking questions could materially change the incentive structure. The challenge was not simply to create a reward function that produced reasonable scores. It was to create one that preserved a meaningful distinction between good and bad policies. That required repeatedly testing the environment, evaluating baselines, and examining whether the reward actually supported the behaviors we wanted the model to discover.

## Is There Enough Headroom for RL?

Another lesson carried over directly from the earlier supervised prescribing project: before training an RL policy, establish whether RL has a realistic opportunity to improve performance. I found it useful to evaluate several distinct questions. First, can the environment distinguish strong behavior from weak behavior? Second, does the supervised policy already perform near the practical ceiling? Third, are there better trajectories that the current policy can plausibly discover through exploration?

And fourth, does the reward function assign those better trajectories meaningfully higher rewards? This last question is particularly important. A task may have substantial room for improvement from a human perspective, but if the reward function cannot distinguish better behavior from worse behavior, the RL algorithm has little useful signal to optimize. Similarly, a well-designed reward function is not enough if the model's exploration never encounters successful trajectories.

In my multi-turn experiments, I examined whether alternative action sequences could produce better outcomes than the supervised policy and whether the model could generate those alternatives with sufficient frequency. These diagnostics were often more informative than simply launching another training run.

They helped separate three very different problems: The model does not know how to produce better behavior. The model can produce better behavior, but the current policy rarely selects it. The reward function fails to recognize better behavior.

Reinforcement learning may help with the second problem, and potentially the first when exploration can discover useful strategies. It cannot reliably solve the third without improving the feedback signal.

## Sparse Rewards and Credit Assignment

Another connection to classical deep RL is the problem of credit assignment. Suppose a multi-turn agent asks three questions and then makes a final decision. If the final outcome is correct, which of those questions contributed to success? Was the first question essential? Did the second question resolve the relevant uncertainty? Was the third unnecessary? A single final reward does not directly answer these questions. The optimization algorithm must infer which actions contributed to the eventual outcome, often through a noisy training signal.

This is difficult in conventional reinforcement learning, and it becomes increasingly challenging as language model agents execute longer trajectories involving tool calls, intermediate reasoning, and external state. The problem is especially interesting because language models operate in an enormous action space. A question is not merely one discrete action. There are countless ways to formulate it, each potentially eliciting different information from the environment.

Even if the model eventually reaches a correct answer, the trajectory may have been inefficient, unnecessarily risky, or dependent on accidental information. Rewarding only the final answer can obscure these differences. This is one reason I believe evaluation of agentic systems needs to account for trajectories, not just final outputs. A successful result is important, but how the agent arrived there matters too.

## Offline Learning Versus Online Experience

One of the broader ideas I've been thinking about is the distinction between learning from existing data and learning through interaction. Much of modern LLM development relies on offline data: documents, demonstrations, preference comparisons, and previously generated trajectories. These datasets are powerful, but they reflect behavior that has already occurred. Online reinforcement learning offers something different.

The policy can generate new trajectories, interact with an environment, observe consequences, and update itself based on those experiences. This creates the possibility of discovering strategies that were not explicitly represented in the original training data. Of course, online interaction introduces significant challenges. Environments must be reliable and sufficiently realistic. Rewards must be trustworthy. Exploration must be constrained appropriately, particularly in safety-critical domains.

For my clinical experiments, the agent interacts with a simulated, controlled environment rather than autonomously making real-world medical decisions. This distinction is essential. The research objective is to understand whether learning through feedback can improve sequential decision-making, not to replace physician judgment. More broadly, I think the opportunity for LLM post-training is increasingly tied to the quality of the environments in which models learn.

A powerful optimization algorithm is useful, but the environment determines which experiences the model encounters and what it can learn from them.

## The Environment Is Part of the Training System

This leads to one of my biggest conceptual takeaways. When building an RL system, the environment is not merely a benchmark used to evaluate the trained model. It is part of the learning process itself. In supervised learning, we spend considerable effort curating datasets, improving labels, and selecting representative examples. In reinforcement learning, comparable effort must go into designing environments, actions, observations, transitions, and rewards.

An environment that is too easy may provide little useful learning signal. One that is too difficult may produce almost no successful trajectories. One with unrealistic shortcuts may encourage strategies that perform well in simulation but fail elsewhere. And one with poorly aligned rewards may actively train undesirable behavior. For language model agents, this means environment engineering may become just as important as dataset engineering has been for supervised learning.

It also changes how I think about the value of proprietary data. Data is useful not only as examples to imitate, but potentially as the foundation for interactive environments in which agents can practice consequential decisions and receive meaningful feedback. That feels like an important direction for applied machine learning, particularly in enterprise settings with specialized workflows and domain-specific knowledge.

## What I've Taken Away So Far

My first serious exploration of modern RL-based LLM post-training has reinforced several lessons. First, supervised fine-tuning remains an exceptionally strong baseline. Before using RL, establish that there is meaningful headroom for improvement. Second, reinforcement learning becomes most interesting when the problem involves sequential decisions, uncertainty, exploration, and consequences that are difficult to capture through static demonstrations.

Third, reward design is not an implementation detail. It defines the incentives the policy learns to optimize, and small changes can produce significant behavioral differences. Fourth, evaluation must examine more than final-answer accuracy. For agents, the trajectory matters: what information was gathered, which actions were selected, how uncertainty was handled, and whether the process was efficient and safe. Finally, the quality of the learning environment may be as important as the choice of optimization algorithm.

These observations have also given me a new appreciation for the continuity between classical deep reinforcement learning and today's language model post-training techniques. The architectures, action spaces, and computational scale are different, but many of the foundational challenges remain the same.

## Closing Thoughts

I'm still early in this particular journey. Although I've worked with supervised fine-tuning and preference-based optimization for years, directly building environments, designing rewards, measuring exploration headroom, and training language models with online RL has been a different experience. It has also been one of the more intellectually rewarding directions I've explored recently. What excites me most is not simply the prospect of making models score higher on benchmarks.

It's the possibility of training agents that learn more effective strategies through experience, especially in domains where decisions unfold over time and success depends on interacting intelligently with an environment. At the same time, my experiments have made me more cautious about assuming RL is always the answer. Sometimes supervised learning is sufficient. Sometimes the model needs better demonstrations. Sometimes it needs a more informative reward.

And sometimes the most important work is designing an environment where meaningful learning is possible in the first place.

For me, that's been the central lesson of this first foray into modern LLM reinforcement learning:

**The most interesting question isn't just how to optimize the policy. It's how to create the conditions under which the policy can actually learn something new.**
