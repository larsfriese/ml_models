# Analysing the importance of features in a neural network for individual predictions
**Prerequisets**

#### Dataset1:

Label | 1  | 2  | 3  |
----- | -- | -- | -- |
1     | 30 | 80 | 2  |
0     | 20 | 0  | 0  |
0     | 15 | 0  | -1 |

#### Model Structure:
![](https://miro.medium.com/max/1400/0*IlHu39jf2c7QC4kn.)

> Source: https://medium.com/@jklein694/deep-dive-in-recurrent-neural-networks-for-binary-classification-project-cd15d89694da

**Step 1:** Train a model with *Dataset1* and save it. The Saved Model will be *Model1*.

**Step 2:** Method1: First part of analyzing: For one Feature, set all other Feature values to 0.
Example: We set all input values 0 except the ones for Feature 1.
When we put Dataset1 through it now, the neuron values will only be affected by Feature 1, which will give us a clear
picture of which neurons get activated by Feature 1 and to what value.
