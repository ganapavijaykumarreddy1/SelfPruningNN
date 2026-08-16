# Self-Pruning Neural Network: Final Report

## 1. Why an L1 Penalty on Sigmoid Gates Encourages Sparsity

In this architecture, each weight $W$ is multiplied by a gate value $G = \sigma(S)$, where $S$ is a learnable parameter. We penalize the network by adding the $L_1$ norm of all gate values to the standard classification loss:
$$Total Loss = CrossEntropy + \lambda \sum G$$

**Mathematical Intuition:**
Because the output of a sigmoid is always strictly positive ($G \in (0, 1)$), the $L_1$ norm simplifies to the sum of the gates. The gradient of this sparsity loss with respect to a gate score $S$ is:
$$\frac{\partial L_{sparsity}}{\partial S} = \lambda \cdot \sigma(S)(1 - \sigma(S))$$
Because both $\lambda$ and $\sigma(S)(1 - \sigma(S))$ are always positive, this gradient constantly applies a negative pressure on $S$. As the optimizer takes steps in the negative gradient direction, the gate scores are driven towards $-\infty$, pushing the gate values $G$ exactly towards $0.0$. Unless a specific weight is crucial for minimizing the Cross-Entropy loss (which generates an opposing positive gradient), the constant $L_1$ pressure will prune it.

## 2. Sparsity vs. Accuracy Trade-off

The following table demonstrates the results of our hyperparameter sweep over $\lambda$. We consider a weight "pruned" if its gate value falls below the hard threshold of `0.01`.

| Lambda ($\lambda$) | Test Accuracy (Soft) | Test Accuracy (Hard-Pruned) | Sparsity Level (%) | Active Parameters |
|:------------------:|:--------------------:|:---------------------------:|:------------------:|:-----------------:|
| 0 | 52.15% | 52.15% | 0.00% | 1,737,984 |
| 5e-06 | 57.37% | 57.36% | 75.55% | 424,881 |
| **8e-06** | **57.85%** | **57.99%** | **81.91%** | **314,321** |
| 1e-05 | 57.42% | 57.47% | 84.41% | 270,875 |
| 1.2e-05 | 57.53% | 57.57% | 86.29% | 238,253 |
| 1.5e-05 | 56.73% | 56.63% | 88.26% | 204,111 |
| 3e-05 | 55.39% | 55.11% | 92.94% | 122,645 |
| 5e-05 | 54.00% | 54.22% | 95.54% | 77,438 |
| 0.0001 | 52.95% | 50.92% | 97.83% | 37,770 |
| 0.0002 | 51.28% | 46.54% | 99.04% | 16,757 |
| 0.0003 | 49.90% | 42.02% | 99.41% | 10,335 |
| 0.0005 | 48.75% | 35.75% | 99.71% | 5,121 |

**Analysis:**
- **Regularization Benefit:** At $\lambda = 10^{-5}$, the network achieves 84.41% sparsity while accuracy actually *improves* from 52.15% to 57.47%. The gates act as a powerful continuous regularization mechanism (preventing over-reliance on a few parameters) before ultimately collapsing the weak connections to 0.
- **Extreme Compression:** At $\lambda = 0.0001$, the network compresses its weights by almost 98% while maintaining a highly respectable 50.92% accuracy (virtually identical to the unpruned baseline).
- **Network Collapse:** Pushing $\lambda$ beyond $0.0001$ results in sparsity approaching 100%, causing the classification accuracy to rapidly degrade as critical connections are forcefully deleted.

## 3. Accuracy vs. Sparsity Plot

The plot below visualizes the trade-off between the test accuracy and the sparsity level as the regularization coefficient $\lambda$ increases.

![Accuracy vs Sparsity](results/accuracy_vs_sparsity.png)

## 4. Final Gate Distribution

The plot below shows the distribution of the final gate values for our best model. Notice the massive spike exactly at $0$ representing the dead/pruned connections, and the long tail of continuous values away from $0$ representing the active, surviving connections.

![Gate Distribution](results/gate_distribution.png)
