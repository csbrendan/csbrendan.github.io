# Deep Reinforcement Learning: A Terms List

A glossary inspired by my deep reinforcement learning class, spanning imitation learning, control, offline RL, meta-RL, and LLM post-training.

## 1. Actor-Critic

Actor-critic methods split learning into two parts: the actor chooses actions, and the critic estimates how good actions are. This is useful in continuous control because the actor can improve using value guidance instead of raw Monte Carlo trial and error. The tradeoff is complexity and instability, because now both the policy and the value estimates can be wrong and affect each other.

## 2. Advantage Function

The advantage function measures how much better or worse an action is than the state’s baseline value. It helps distinguish “good action in this state” from “good state overall.” Methods like AWAC-style weighting use it because it focuses policy learning on actions that are above-average for the states they appear in.

## 3. Advantage-Based RL

Advantage-based RL uses an estimate of how much better an action was than the usual expected outcome in that state. Instead of only asking whether an action got high reward, it asks whether that action was better than the baseline for that situation. This often leads to more efficient and stable policy updates, but it depends on having a good estimate of the advantage.

## 4. Advantage-Weighted Actor Critic (AWAC) Style Update

The policy update used in IQL is similar to AWAC: actions in the dataset with higher estimated advantage get imitated more strongly. Intuitively, it is like weighted behavior cloning where “good” actions count more. This is safer than optimizing over arbitrary actions, but it can only promote actions that were actually seen in the data.

## 5. Advantage-Weighted Regression (AWR)

Advantage-weighted regression is a method that updates a policy by imitating actions in the dataset, but weighting each action by how advantageous it appears to be. Intuitively, it is behavior cloning with more emphasis on actions that seem better than average and less emphasis on weaker ones. It is attractive because it stays close to data support while still using reward information, though its quality depends on how well the advantages are estimated.

## 6. Asynchronous RL

Asynchronous RL is a training setup where data collection and learning happen in parallel instead of waiting for each other step by step. Multiple workers or model copies generate rollouts at the same time while another process updates the policy, which makes training much faster and keeps expensive hardware better utilized. In LLM RL pipelines, this is especially relevant for on-policy rollout generation, since sampling long responses is slow, so systems often generate batches of completions on many workers while the trainer updates on recently collected data. The tradeoff is that the rollouts can become slightly stale because they may have been produced by an older version of the policy, so it is faster and more scalable than fully synchronous training, but usually less cleanly on-policy.

## 7. Behavior Cloning (BC) / Supervised Imitation

BC is imitation learning done as supervised learning: train a policy to copy the expert’s action for each observed state, instead of learning from rewards. It is good when you have strong demonstrations and want something simple and stable. The downside, is that small mistakes compound once the agent drifts into states not covered by the demonstrations, so BC can be brittle on unstable tasks.

## 8. Behavior Cloning Pretraining

In A2, behavior cloning was used to initialize the actor from successful demonstrations before RL fine-tuning. This is good for sparse-reward tasks because it gives the policy a strong starting point instead of asking it to discover everything from scratch. The tradeoff is that the policy can remain biased toward demonstration behavior unless RL fine-tuning improves it.

## 9. Behavior Policy

The behavior policy is the unknown policy or mixture of policies that generated the offline dataset. Offline RL methods must work with whatever data this policy happened to produce. If the data is narrow or poor quality, the learned policy is constrained by that coverage, even if the algorithm itself is strong.

## 10. Bellman Target / Bellman Backup

A Bellman target uses the current reward plus an estimate of future value to update the critic. It is the basic recursive idea behind TD learning and Q-learning. It is sample efficient, but if the future estimate is biased, that error can keep propagating through learning.

## 11. Black-Box Meta-RL / RL²

RL² is an end-to-end meta-RL method that tries to learn adaptation implicitly through recurrent experience across tasks. It can in principle learn both how to explore and how to exploit, but that coupling makes the problem hard. For example, a recurrent policy may learn some useful behavior without discovering a structured information-gathering strategy.

## 12. Clipped Double Q / Min over Critics

This means using two target critics and taking the minimum of their Q-values when forming the backup. It is meant to reduce overestimation, because if one critic is unrealistically optimistic, the other may keep the target grounded. The tradeoff is possible underestimation, but in practice it often stabilizes training.

## 13. Conservative Q-Learning (CQL)

CQL is an offline RL method that explicitly pushes down Q-values for actions not supported by the data, while keeping Q-values on dataset actions relatively higher. The point is to prevent the policy from chasing fake high values caused by extrapolation error. Compared with IQL, it is more directly conservative and often easier to think about, but it can become overly pessimistic if regularized too strongly.

## 14. CQL Alpha (α)

α controls how strong the conservatism penalty is in CQL. At α = 0, you effectively remove the conservative part and fall back toward ordinary Q-learning style updates, which is riskier offline. Larger α gives more protection against overestimation, but too much can make the method overly pessimistic and hurt performance.

## 15. CQL Regularizer / Conservatism Penalty

This is the extra term in CQL that penalizes large Q-values broadly, especially away from the data. It is what makes CQL “conservative.” The benefit is protection against overestimation; the cost is that if the penalty is too strong, even genuinely good actions may get undervalued.

## 16. Credit Assignment

Credit assignment is the problem of figuring out which parts of a trajectory deserve praise or blame for the final result. In reasoning LLMs, this means deciding which intermediate steps helped produce a correct answer and which were spurious or harmful. Good credit assignment is crucial for learning efficient reasoning, but it is difficult when only the final outcome is observed.

## 17. Critic Ensemble

A critic ensemble uses multiple critics instead of just one. Multiple critics can make targets more robust and training more stable. Ensembles can reduce variance and overoptimism, but they cost more compute and memory.

## 18. DAgger (Dataset Aggregation)

DAgger improves on BC by repeatedly letting the learned policy act, collecting the states it actually visits, and relabeling those states with expert actions. It is especially good for recovery behavior because it teaches the agent what to do in its own mistake states, not just expert states. Compared with BC, it is more robust but needs more interaction and repeated expert labeling.

## 19. Deep Q-Network (DQN)

DQN is a value-based method that learns Q-values and acts by choosing the action with the highest estimated Q. In A4 it was adapted into a goal-conditioned setting by feeding both state and goal into the Q-network. DQN is conceptually simple, but in sparse-reward multi-goal problems it often needs help from techniques like HER.

## 20. Dense Rewards

Dense rewards provide a smoother learning signal at many steps, for example negative distance to a goal. They are easier to optimize because the agent gets constant feedback on progress. In a task such as Sawyer with distance-based reward information, HER may provide less benefit because the reward is already dense.

## 21. Distribution Shift

Distribution shift here means the states seen at training time differ from the states seen when the learned policy runs on its own. For example, Hopper can fail under BC because once the cloned policy makes a slight mistake, it enters out-of-distribution states and errors snowball. DAgger is designed mainly to fix this problem.

## 22. DPO (Direct Preference Optimization)

DPO is a preference optimization method that trains a language model directly from preferred-versus-rejected outputs, without needing a separate online RL loop. It can be understood as using preference data to directly increase the probability of preferred responses and decrease the probability of dispreferred ones, while staying anchored to a reference model. It is attractive because it is simpler and cheaper than RLHF, though it may give up some of the flexibility of an explicit reward-model-plus-RL pipeline.

## 23. DREAM

DREAM is a decoupled meta-RL method that separately learns exploration and exploitation. Instead of hoping one recurrent policy learns everything end-to-end, it explicitly learns a task representation and trains exploration to reveal the information needed for exploitation. Its intended strength is learning structured explore-then-execute behavior.

## 24. Expectile Parameter (ζ)

ζ controls how optimistic or selective the expectile regression is. A higher expectile pays more attention to better outcomes in the data; a lower one is more conservative. This makes ζ a tradeoff between exploiting the best dataset behavior and staying robust when good outcomes are rare or noisy.

## 25. Expectile Regression

Expectile regression fits an asymmetric notion of “average” where errors on one side count more than errors on the other. In IQL, this is used to make the value function focus on higher-quality actions in the dataset instead of the ordinary mean. It is useful because it nudges learning toward better-supported good behavior, but the expectile choice is a sensitive hyperparameter.

## 26. Expert Demonstrations

Expert demonstrations are trajectories or state-action pairs produced by a strong policy or human. They are useful because they give direct examples of what actions should look like in important states. Their weakness is coverage: if the expert never visits a class of bad or recovery states, the learned policy may fail badly when it reaches them.

## 27. Exploration Bonus

An exploration bonus is extra reward added to make novel or informative states more attractive during data collection. In RND, the bonus comes from prediction error. It is useful because better data collection improves later offline learning, but bonuses can distort behavior toward novelty for its own sake.

## 28. Exploration Episode vs Exploitation Episode

In the A4 meta-RL setup, the agent first gets an exploration episode to gather information, then an exploitation episode where returns matter. This separation makes the logic of adaptation easier to analyze: explore to infer the task, then exploit using that information. It also makes clear why methods that muddle these together may learn weaker strategies.

## 29. Explore-Then-Execute Strategy

This is the structured meta-RL pattern where the agent first gathers task information and then uses it optimally. A map-then-bus task illustrates the idea: consult a map to learn the route, then take the right bus. It is better than aimless exploration because the information collected is chosen specifically to improve later return.

## 30. Goal-Conditioned Reinforcement Learning

Goal-conditioned RL trains a policy or value function to solve many goals by conditioning on a goal input in addition to the current state. This is useful because one model can generalize across targets instead of learning a separate policy per goal. It works especially well when goals share structure, but exploration becomes hard if rewards are sparse.

## 31. GRPO (Group Relative Policy Optimization)

GRPO is a PPO-style RL method that uses relative comparisons within a group of sampled outputs, rather than relying on a separately learned value function in the usual way. The basic idea is to sample several completions for the same prompt, compare their rewards, and use those relative differences to decide which completions should be upweighted or downweighted. It is attractive for LLM reasoning because it can simplify training and reduce some of the overhead of standard PPO, but it still depends heavily on the quality of sampled groups and reward signals.

## 32. HER Final Relabeling

Final relabeling replaces the goal with the final state reached in the episode. It is simple and often strong because the final state is guaranteed to have been achieved. The tradeoff is that it may relabel every step toward a goal that is not the most informative one for earlier parts of the trajectory.

## 33. HER Future Relabeling

Future HER relabels each transition with a state that is achieved later in the same episode. This is often the most natural relabeling because it preserves the causal idea that the current action sequence actually led to that future state. It creates hindsight goals that are directly connected to what happened after the transition.

## 34. HER Random Relabeling

Random HER picks some achieved state from the episode as the relabeled goal. This gives more diversity than final relabeling and can spread learning over different achieved outcomes. The downside is that some randomly chosen relabels may be less relevant or coherent than future-based ones.

## 35. Hindsight Experience Replay (HER)

HER relabels failed experience as if the goal had been whatever state the agent actually reached. That turns “failure” into useful supervision and is especially powerful in sparse-reward tasks where true success is rare. Bit-flipping illustrates the sparse-reward setting where HER can help substantially; tasks that already provide dense rewards may benefit less.

## 36. Hyperparameter Tuning

This is the process of varying settings like training steps, network size, amount of data, or regularization strength to change performance. BC illustrates the classic tradeoff: too little training can underfit, too much can start to overfit, and there is often a middle range that works best. In RL, hyperparameters often matter more than people expect because learning is indirect and unstable.

## 37. Imitation Learning

Imitation learning means learning behavior from expert demonstrations rather than from trial-and-error reward optimization. It is appealing because it can avoid hard exploration and reward design. The tradeoff is that it usually inherits the quality and coverage of the demonstrator, and plain imitation usually cannot outperform the expert data it was trained on.

## 38. Implicit Q-Learning (IQL)

IQL is an offline RL method designed to avoid the dangerous step of evaluating unseen OOD actions. Instead of asking, “what if I took some new action here?”, it learns values only from actions actually present in the dataset and then extracts a policy from those learned values. It is good when you want a relatively stable offline RL method that stays close to data support, but its performance still depends heavily on dataset quality.

## 39. Instruction Fine-Tuning

Instruction fine-tuning is the LLM version of supervised imitation. A pretrained language model is trained on prompt-response pairs so it learns to follow instructions and behave more like an assistant rather than just continue text. It is very effective for formatting, tone, and general task-following, but it does not directly optimize for human preference or correctness beyond the examples it imitates.

## 40. Intrinsic Reward / Information Gain

The DREAM exploration policy is rewarded not for task completion directly, but for increasing information about the task. This intrinsic reward is basically “how much did this new transition tell me about what task I am in?” It is useful when the external reward does not directly supervise exploration, but it only works if information really is what the exploitation policy needs.

## 41. KL Penalty

A KL penalty is a regularization term that discourages a new policy from drifting too far from a reference policy. In LLM RL, it is often used to keep the updated model close to the pretrained or instruction-tuned model while still improving reward. This makes training more stable and reduces reward hacking, but if the penalty is too strong it can also prevent meaningful improvement.

## 42. Meta-Reinforcement Learning

Meta-RL means learning how to adapt quickly across a distribution of tasks, rather than just solving one fixed task. The goal is that experience on one episode can help performance on the next episode in the same task. It is powerful for structured task families, but can be hard to optimize because the agent must learn both task inference and control.

## 43. Model Predictive Control (MPC)

MPC chooses actions by planning ahead with a model of the dynamics, rather than only relying on a learned policy. MuJoCo MPC can show the effect of reward functions quickly without waiting for full RL training. It is great for fast reward iteration and control, but it depends on having a usable simulator or model and can be computationally expensive online.

## 44. Mutual Information Objective

DREAM trains exploration partly by maximizing mutual information between the latent task encoding and the exploration trajectory. Intuitively, exploration should gather observations that reveal the task as clearly as possible. This is a principled way to reward informative exploration, though it requires extra modeling machinery like encoders and decoders.

## 45. Off-Policy Learning

Off-policy learning means the algorithm can learn from data generated by older policies or different policies altogether. This is more sample efficient because past data can be reused, but it creates a mismatch between the policy being learned and the policy that produced the data. In both classic RL and LLM training, that mismatch can make value estimates unreliable if not handled carefully.

## 46. Offline Reinforcement Learning

Offline RL means learning a policy from a fixed dataset without collecting new interactions during training. It is useful when online interaction is expensive, unsafe, or impossible. The main challenge, highlighted in the lecture and A3, is that policies can exploit Q-function errors on actions not well supported by the dataset.

## 47. On-Policy Learning

On-policy learning means the algorithm trains mostly on data generated by the current version of the policy. This is appealing because the training data closely matches the behavior of the model being optimized, which makes policy improvement cleaner and theoretically better grounded. The downside is cost: for LLMs, generating fresh rollouts every update can be very expensive.

## 48. Online RL

Online RL means the policy keeps collecting fresh experience from the environment while it is being trained. This allows the model to adapt its behavior as it improves and gather data that matches its current policy. It is powerful and flexible, but often expensive and unstable because collecting new data is part of the training loop.

## 49. Out-of-Distribution (OOD) States or Actions

OOD means the policy or value function is asked to handle states or actions not well represented in the data. In imitation learning, OOD states appear when the policy drifts off the expert trajectory. In offline RL, OOD actions are especially dangerous because the Q-function may assign them unrealistically high values, which is a big reason IQL and CQL are needed.

## 50. Overfitting

Overfitting means the policy matches the training data too specifically and generalizes worse when states vary slightly at test time. Performance peaking and then dropping with too many BC steps is an intuitive example. In practice, overfitting matters more when expert data is limited or narrow.

## 51. Policy Gradient

A policy gradient method updates the policy directly by increasing the probability of actions or outputs that led to higher reward. In LLM RL, this means increasing the likelihood of responses that scored well and decreasing the likelihood of poor ones. It is conceptually simple and very general, but often noisy and unstable unless paired with baselines, clipping, or trust-region style constraints.

## 52. PPO (Proximal Policy Optimization)

PPO is an on-policy RL algorithm that improves a policy while limiting how large each update can be. It is popular because it is much more stable than naive policy gradient updates: instead of letting the model swing too hard toward whatever scored well in one batch, it clips the update so learning stays gradual. In LLM RLHF, PPO became a standard choice because it works reasonably well for optimizing against reward models, though it is compute-heavy and requires fresh rollouts.

## 53. Q-Function / Critic

A Q-function estimates how good it is to take action (a) in state (s), usually in terms of expected future return. It is central to many methods here because it guides policy improvement. The big problem is that inaccurate Q-values, especially overestimated ones, can push learning in the wrong direction.

## 54. Random Epsilon-Greedy Exploration Data

This is offline data collected with a simple exploration policy that is partly random and partly follows current value estimates. It is easy to generate and can cover some space, but it is often low quality or unfocused. In offline RL, better exploration data usually means better final policies.

## 55. Random Network Distillation (RND)

RND encourages exploration by rewarding states whose features are hard for a predictor network to match against a fixed random target network. Intuitively, unfamiliar states have high prediction error, so they look novel. This often produces richer datasets than random exploration, but it adds extra machinery and can still chase novelty that is not task-relevant unless used carefully.

## 56. Rejection Fine-Tuning (RFT)

Rejection fine-tuning means sampling multiple outputs from a model, keeping only the ones that satisfy some success criterion, and then fine-tuning on those successful outputs. For reasoning models, this often means generating many candidate solutions and training only on the ones that reach the correct final answer. It is a simple way to improve a model using its own successful trajectories, but it can accidentally reinforce flawed intermediate reasoning if success alone is used as the filter.

## 57. Replay Buffer

A replay buffer is a storage pool of previously collected training samples that the algorithm can learn from later. In standard RL it stores transitions or trajectories; in LLM RL it can store prompt-response-reward examples or full generated traces. Replay buffers improve data reuse and efficiency, but the tradeoff is that older samples may become stale if the policy changes a lot.

## 58. Reward Function / Reward Design

A reward function tells the agent what behavior is desirable by assigning numbers to outcomes or transitions. A2 emphasized that reward design is often the hardest part of RL, because a reward that sounds reasonable can still drive weird behavior. Reward functions are powerful, but they can be brittle proxies for what you really want.

## 59. Rollout

A rollout is one sampled run of the current policy through a task. In LLM RL, that usually means giving the model a prompt, letting it generate a response, scoring that response, and then using that prompt-response sample for training. Rollouts are useful because they are the raw experience the algorithm learns from, but they can be expensive to generate, especially for long reasoning traces.

## 60. Sample Efficiency

Sample efficiency is how much learning the agent gets from each trajectory or transition. HER improves sample efficiency by extracting extra supervised signal from the same episodes. This matters a lot when interaction is expensive or rewards are too sparse for plain learning to make progress.

## 61. Sparse Rewards

Sparse rewards only give useful feedback rarely, such as only when the full task succeeds. They are hard because the agent gets very little signal about which actions were helpful along the way. Examples include the hammer task and especially the bit-flipping setting, where HER helps a lot because ordinary learning gets almost no useful positive examples.

## 62. Synchronous Training

Synchronous training means rollout generation and policy updates happen in lockstep, usually waiting for a full batch before proceeding. This is cleaner and more strictly on-policy, since the data is tightly matched to the current model. The tradeoff is lower hardware utilization and slower throughput, especially for long LLM generations.

## 63. Target Networks

Target networks are slower-moving copies of the critic used to compute more stable Bellman targets. They help because learning becomes less like chasing a target that changes every step. The tradeoff is that they slow the adaptation of the target estimates, but that is usually worth it for stability.

## 64. Task Encoding (z)

In DREAM, z is a compact representation of the task that contains the information the exploitation policy needs. This is useful because the exploitation policy does not need to reconstruct the full task from scratch every time, only condition on the right summary. The quality of z matters a lot: too weak and exploitation fails, too rich and exploration becomes harder to infer reliably.

## 65. Temporal Difference (TD) Learning

TD learning updates values using one-step bootstrapped targets instead of waiting for full returns. It is usually more sample efficient than Monte Carlo methods, especially in long tasks. The tradeoff is bias and error propagation: if the bootstrap target is wrong, TD can spread that mistake.

## 66. Trajectory

A trajectory is the full sequence of states, actions, and outcomes from one episode of interaction. In language-model RL, a trajectory can be thought of as the prompt, the generated tokens or reasoning steps, the final completion, and the reward or verifier score attached to it. In many LLM papers, “trajectory” and “rollout” are used almost interchangeably, though “trajectory” emphasizes the full sequential record while “rollout” emphasizes the act of sampling it.

## 67. Trajectory Stitching

Trajectory stitching means combining good pieces of behavior from different trajectories into a better overall policy. This is one of the reasons offline RL can outperform plain imitation: it can use reward information to connect locally good segments that were never demonstrated end-to-end together. It works only if the dataset contains those useful pieces in the first place.

## 68. Underfitting

Underfitting means the model has not learned enough from the data to capture useful behavior. With too few training steps, the policy has not had enough updates to match the expert well. It tends to produce weak performance but is usually more stable than severe overfitting.

## 69. Update-to-Data Ratio (UTD)

UTD is how many gradient updates you do per environment step collected. Higher UTD can extract more learning from each sample, potentially improving sample efficiency. But if it is too high, the agent can overfit to replay data or amplify critic errors, so it is a speed-stability tradeoff.

## 70. Value Function

A value function estimates how good a state is, usually in terms of expected future reward. In LLM reasoning, a value-like estimate can represent how promising a partial solution or prefix is before the final answer is complete. Value functions help reduce variance and improve credit assignment, but bad value estimates can mislead the policy.

## 71. Value Function in IQL

IQL introduces a separate value function V(s) in addition to Q(s, a). The point is to estimate a high expectile of in-dataset action values without having to maximize over potentially OOD actions. That makes IQL safer offline than standard actor-critic methods, though also a bit more indirect.

## 72. Value-Based RL

Value-based RL learns to estimate how good states or state-action pairs are and then uses those estimates to choose actions. The most common form is learning a Q-function, which predicts expected future reward for taking a given action in a given state. It can be very sample efficient, but performance depends heavily on the accuracy of the learned value estimates.

## 73. Verifier

A verifier is a scoring mechanism that checks whether a model’s output satisfies some criterion, often correctness. In reasoning LLMs, a verifier may simply check whether the final math answer matches the ground truth, or more generally score whether a solution is valid. Verifiers are powerful because they give more objective feedback than vague human preference, but they work best on tasks where correctness is easy to test.
