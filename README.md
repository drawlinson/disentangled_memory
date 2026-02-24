# Active perception and disentangled representations allow continual, episodic, zero-shot and few-shot learning
This repository includes all the methods for the results and graphics contained in the paper [Active perception and disentangled representations allow continual, episodic zero and few-shot learning](https://arxiv.org/abs/2602.19355) (published Feb 2026). The paper describes an episodic Reinforcement Learning (RL) Agent which rapidly learns to handle encounters with various objects and animals, a bit like a simplified "Dungeons & Dragons" game. This is just our first demonstration of the architecture and we hope to release new results on some benchmark datasets soon.

## Abstract
Generalization is often regarded as an essential property of machine learning systems. However, perhaps not every component of a system needs to generalize. Training models for generalization typically produces entangled representations at the boundaries of entities or classes, which can lead to destructive interference when rapid, high-magnitude updates are required for continual or few-shot learning. Techniques for fast learning with non-interfering representations exist, but they generally fail to generalize. Here, we describe a Complementary Learning System (CLS) in which the fast learner entirely foregoes generalization in exchange for continual zero-shot and few-shot learning. Unlike most CLS approaches, which use episodic memory primarily for replay and consolidation, our fast, disentangled learner operates as a parallel reasoning system. The fast learner can overcome observation variability and uncertainty by leveraging a conventional slow, statistical learner within an active perception system: A contextual bias provided by the fast learner induces the slow learner to encode novel stimuli in familiar, generalized terms, enabling zero-shot and few-shot learning. This architecture demonstrates that fast, context-driven reasoning can coexist with slow, structured generalization, providing a pathway for robust continual learning.

![alt text](/readme.png?raw=true)

## Notebooks
* [stimulus_response_test.ipynb](stimulus_response_test.ipynb) - manually play with the test environment and navigate the "encounters" from the paper
* [stimulus_response_dataset.ipynb](stimulus_response_dataset.ipynb) - creates a dataset like the one used in the paper
* [stimulus_response_experiment.ipynb](stimulus_response_experiment.ipynb) - can be used to train and test models per the streaming, few-shot and zero-shot conditions described in the paper
* [stimulus_response_analysis.ipynb](stimulus_response_analysis.ipynb) - analyses the results to produce the plots in the paper


# (Associative) Sparse Distributed Memory
The core feature of this repository is the sparse distributed memory model, which has high capacity and can learn very quickly (few-shot) and without interference between memories. It stores vector values associated with real dense vector keys:

[sparse_distributed_memory.py](/disentangled/sparse_distributed_memory.py)

The memories are stored in a sparse, distributed way which provides the desired qualities of robustness, non-interference, high orthogonality and ability to perform rapid updates.

In that file there's also an **associative** version of the memory, which allows pattern completion, recall from cue, and sampling of original input vectors. Together, these capabilities allow construction of agents which rapidly adapt to changing conditions. Note that the memory by itself is not _disentangled_, for that you need a specific architecture (like the one proposed in the paper) to extract salient variables from input data.
