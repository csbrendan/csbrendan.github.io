# Deep RL for Post Training

The first post training step is usually instruction fine tuning, which is essentially behavior cloning for natural language. The model is shown prompts and target responses and trained to imitate the desired style. This moves it from text completion to a helpful assistant that can answer questions and follow a format.

From there, post training can use reinforcement learning from human feedback, or RLHF. The model generates multiple answers and humans pick the ones they prefer. It is easier for humans to rank answers than to generate all those examples for training.

The resulting comparisons are used to train a reward model. Once we have a reward model, the language model can be optimized using its feedback during RL.

When the true objective is hard to write down directly, learn a proxy reward from feedback and optimize that.

For LLMs, the proxy is learned from preference data rather than being coded by hand like a robotics reward. The policy is the language model itself. The optimization loop encourages generations that the reward model scores highly.

## Reward hacking and KL penalties

Reward hacking shows up here too. If you optimize the model too aggressively against the reward model, it can produce strange yet high scoring outputs. This is a current limitation in post training.

To keep the model from drifting too far, RLHF can use a KL penalty. You reward the model for outputs that people prefer, but penalize it for moving too far from the original model, called the reference model. This is in the spirit of constrained policy optimization in RL.

An alternative is direct preference optimization, or DPO. It simplifies the process because there is no need to train a separate reward model and run an RL loop. DPO uses a classification objective on preferred versus rejected outputs. This makes it cheaper to run.

All of this so far is the assistant tuning side of things: instruction tuning, preference data, reward models, RLHF, DPO, and KL regularization. These are mostly associated with alignment, helpfulness, harmlessness, and dialogue quality.

## Reasoning and sparse rewards

As we look at post training for reasoning, the connection to deep RL becomes more explicit. When an LLM solves a math or coding problem, it can be viewed as a sequential decision process. The prompt is the initial state, each generated step is an action, and the final answer determines the reward.

In many reasoning benchmarks, the reward is sparse. The model only gets a reward for the final answer being correct. That is a classic RL setting with sparse rewards, but expressed in tokens instead of motor torques.

This raises a few questions:

- How does the model assign credit to intermediate steps?
- How does it avoid learning from lucky but flawed solutions?
- How does it improve efficiency when good reasoning traces are limited?

One early answer is rejection fine tuning. Sample many solutions from the current model, keep only those that end in the correct answer, and then fine tune on those. This is similar to policy improvement through filtered imitation: generate behavior from the current policy, retain successful trajectories, and imitate them.

It is simple and surprisingly effective, but flawed at times. If a sampled solution luckily reaches the correct conclusion with weak or spurious intermediate steps, training on those traces can reinforce bad reasoning habits.

In RL, this is poor credit assignment. A final success signal is not sufficient to tell which parts of the trajectory were useful.

## Value and advantage at each step

More advanced reasoning methods start to look like RL methods based on advantage or value. These methods try to estimate how valuable a partial solution prefix is.

One way to do this is through rollouts from intermediate states. Keep the prefix fixed, complete the rest of the solution multiple times, and estimate how often that prefix leads to eventual success. This gives a notion of value at each step. You can then define something like an advantage, measuring whether a particular step improved or worsened the chances of success.

Once you have these signals, they can be used to filter out bad reasoning steps, or they can be turned into dense rewards for online RL. This is similar to advantage weighted regression and actor critic learning.

## Offline RL and the limits of the data

Much of LLM post training does not happen in a pure online interaction loop. It often starts with static datasets of prompts, responses, rankings, or reasoning traces. That makes the setting similar to offline RL, where the learned policy must improve using a fixed batch of data.

In both offline RL and preference tuning, a policy can move too far beyond the distribution where its value estimates and preference signals are reliable. When that happens, optimization becomes brittle.

DPO, KL regularization, and staying close to a reference model can all be viewed as responses to that problem.

## Better base policies make RL more useful

The reasoning lecture also emphasized the importance of online RL, especially in models trained to think longer. Once the model is good enough to generate meaningful reasoning traces with multiple steps, RL can amplify that capability.

The model can sample long solutions, receive verifiable rewards, and gradually learn better search behavior, better self correction, and better verification habits.

The key change was that the models became good enough for RL to improve them in more visible ways.

Reasoning models can learn more structured internal procedures. They can generate an answer, check it, backtrack, revise, and continue. In classical control problems, this looks like a richer action space or a more expressive policy class. In an LLM, it appears as longer chains of reasoning and more deliberate search.

Better base policies make RL more useful.

## The feedback still matters

Post training is still constrained by the quality of the feedback signal. Human preferences are noisy. Pairwise ranking in RLHF throws away information about the strength of preference. Reward models can be exploited. Binary correctness is easy to verify for math but much harder for open ended reasoning.

The bottleneck is often supervision. Performance depends on the structure of the feedback as well as the algorithm. Dense rewards are generally easier to learn from than sparse rewards.

Which deep RL methods seem most relevant to post training LLMs?

At the assistant layer, the main transfers are behavior cloning, reward modeling, policy optimization with a KL constraint, and DPO.

At the reasoning layer, they are RL with sparse rewards, credit assignment using rollouts, value and advantage estimation, filtered or rejection based policy improvement, and online RL with verifiable rewards. Around all of this is a concern familiar from offline RL: staying close to the support of good data.

The domain may have changed from physical control to language generation, but the backbone stayed similar. The models are bigger. The actions are tokens. The rewards are often preferences or answer checkers rather than environment signals.

The core challenge remains the same: learn a policy that does more of what we want, from feedback that is always noisy, incomplete, and easier to exploit than it first appears.
