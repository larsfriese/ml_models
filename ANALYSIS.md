# Analysing the importance of features in a neural network for individual predictions
**Data needed for examples/explanation:**

#### Dataset1:

Label | 1  | 2  | 3  |
----- | -- | -- | -- |
1     | 30 | 80 | 2  |
0     | 20 | 0  | 0  |
0     | 15 | 0  | -1 |

#### Model Structure:

3 input neurons, 2 dense layers each 4 neurons, 1 output neuron<br>
3 features, binary classification of 1 label

**Step 1:** Train a model with *Dataset1* and save it. The Saved Model will be *Model1*.

**Step 2:** Method1: First part of analyzing: For one Feature, set all other Feature values to 0.
Example: We set all input values 0 except the ones for Feature 1.
When we put *Dataset1* through it now, the neuron values will only be affected by Feature 1, which will give us a clear
picture of which neurons get activated by Feature 1 and to what value.

![](https://raw.githubusercontent.com/larsfriese/ml_models/master/analysis/analysis1.JPG)

Doing this for every feature will give us the most important neurons for each individual feature.
The biggest valued neurons positions in the layer and values of a feature will be saved into a list like this:

```python
list=['1',[[3,6.455],[1, 7.234], ...] # (example values)
```
list=[[feature1, [highest neurons layer one], [highest neurons layer two], [highest neurons layer three],[feature2, ...]]

The amount of neurons which are put in the list per layer per feature is called *NeuronsAmount*.

**Step 3:** Prediction: Predict a Label from the dataset and save all neurons values and positions in the layer.
Compare the heighest neurons with the ones from the saved list, and count occurences of same position of neurons in corresponding layers. A typical example would look like this:

Feature | Ocurrences of neurons in both lists |
------- | ----------------------------------- | 
1       | 4                                   |
3       | 1                                   |

(example values)

From this result we can conclude that the most important feature for this prediction is feature 1, as its most important neurons
have been spotet 4 times in the new prediction. We can also calculate to what percentage this feature was important:

Feature 1: 4/5 = **80%**<br>
Feature 2: 0/5 = **0%**<br>
Feature 3: 1/5 = **20%**<br>
