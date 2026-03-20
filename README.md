# NPB136_Final

## *Modeling Dopamine and Serotonin in Reinforcement Learning: A Computational Psychiatry Approach to Depression and ADHD*

### Central Question: 
How well can a simple reinforcement-learning model, with dopamine formalized as reward prediction error signaling and serotonin formalized as either punishment sensitivity or behavioral inhibition, reproduce stylized behavioral patterns relevant to depression and ADHD? Dopamine’s link to reward prediction error is a classic result in neuroscience and RL, while recent work suggests serotonin affects reinforcement learning in more than one possible way, including punishment sensitivity, aversive processing, and behavioral inhibition. 

### Our Project:
We examine how neuromodulatory systems discussed in class can be formalized in reinforcement-learning models. Prior work has linked dopamine to reward prediction error signals in temporal-difference learning, while computational and experimental studies suggest serotonin contributes to punishment learning, aversive processing, and behavioral inhibition. 

We propose to implement a simple reinforcement-learning model, such as Q-learning or temporal-difference learning, and extend it with parameters that represent dopaminergic and serotonergic modulation. Dopamine will be modeled as influencing updating from positive reward prediction errors, while serotonin will be modeled as affecting either punishment sensitivity, aversive weighting, or inhibitory choice bias. 

We will then simulate pharmacology-inspired manipulations, such as increased dopaminergic gain or increased serotonergic modulation, and test how these changes alter exploration, exploitation, sensitivity to reward and punishment, and adaptation after negative feedback. Finally, we will compare the resulting behavioral patterns to findings associated with depression and ADHD, including reduced outcome sensitivity in depression and altered reinforcement sensitivity or choice switching in ADHD. Our goal is not to make clinical treatment recommendations, but to explore how simplified computational models can generate hypotheses linking neural systems, behavior, and psychiatric pharmacology. 
