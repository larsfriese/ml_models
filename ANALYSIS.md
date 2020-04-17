# Analysing the importance of features in a neural network for individual predictions

The program tries to analyze for an individual prediction what the most crucial factors<br>
were, which led to a certain decision. This currently only works for binary classification<br>
and 1-Dimensional inputs.

## Explanation:
**Data needed for examples/explanation:**

#### Dataset1:

Label | 1  | 2  | 3  |
----- | -- | -- | -- |
1     | 30 | 80 | 2  |
0     | 20 | 0  | 0  |
0     | 15 | 0  | -1 |

#### Model Structure:

3 input neurons, 2 dense layers each 4 neurons, 1 output neuron<br>
<details>
 <summary>More Details</summary>
  In the code it is actually 3 sperate input layers, as there is a problem with extracting neuron
  values out of a model with FeatureColumns.<br>
  The Code for the dense layers looks like this:
  
  ```python
   bias=False 
  
   elif 1000<=len(dataframe.index):
        x = layers.Dense(128, activation=relu, use_bias=bias)(feature_layer_outputs)
        x = layers.Dense(128, activation=relu, use_bias=bias)(x)
        x = layers.Dense(128, activation=relu, use_bias=bias)(x)
        dense_layers=3
   ```

</details>
3 features, binary classification of 1 label

**Step 1:** Train a model with *Dataset1* and save it. The Saved Model will be *Model1*.

**Step 2:** Method1: First part of analyzing: For one Feature, set all other Feature values to 0.<br>
Example: We set all input values 0 except the ones for Feature 1.<br>
When we put *Dataset1* through it now, the neuron values will only be affected by Feature 1, which will give us a clear<br>
picture of which neurons get activated by Feature 1 and to what value.

![](https://raw.githubusercontent.com/larsfriese/ml_models/master/analysis/analysis1.JPG)

Doing this for every feature will give us the most important neurons for each individual feature.<br>
The biggest valued neurons positions in the layer and values of a feature will be saved into a list like this:<br>

```python
list=['1',[[3,6.455],[1, 7.234], ...] # (example values)
```
list=[[feature1, [highest neurons layer one], [highest neurons layer two], [highest neurons layer three],[feature2, ...]]

The amount of neurons that are put in the list per layer per feature is called *NeuronsAmount*.

**Step 3:** Prediction: Predict a Label from the dataset and save all neuron's values and positions in the layer.<br>
Compare the highest neurons with the ones from the saved list, and count occurrences of the same position of neurons in<br> corresponding layers. A typical example would look like this:

Feature | Occurrences of neurons in both lists |
------- | ------------------------------------ | 
1       | 4                                    |
3       | 1                                    |

(example values)

From this result we can conclude that the most important feature for this prediction<br>
is feature 1, as its most important neurons have been spotet 4 times in the new prediction.<br>
We can also calculate to what percentage this feature was important:

Feature 1: 4/5 = **80%**<br>
Feature 2: 0/5 = **0%**<br>
Feature 3: 1/5 = **20%**<br>

## Working Example:
**Data needed:**

( [CSV Link](https://github.com/larsfriese/ml_models/blob/master/analysis/testdata1.csv "Full CSV Dataset") )
Visual Representation of Data:

![](https://raw.githubusercontent.com/larsfriese/ml_models/master/analysis/analysis2.JPG)

For this dataset, the accuracy after training is at **0.99**.

Now we want to know what the most important feature is for one prediction.
We will take Row **189** as an Example. The Label is 1, and the values are:<br>
Feature1: **0.692900156**, Feature 2: **0.673047423**

Running the code for the prediction, we get the following results:<br>
(*NeuronsAmount* set to 20)

Feature | Occurrences of neurons in both lists |
------- | ------------------------------------ | 
1       | 6                                    |
2       | 6                                    |

It seems like both features are equally important. This is correct,
because when one is going under the value of 0.5 the Label wouldn't be 1.

We can now run a few more predictions for Label 1 data to see if this pattern stays.

Row   | Feature1/Feature2 count of occurrences |
----- | -------------------------------------- | 
406   | 6/6                                    |
331   | 8/8                                    |
269   | 6/6                                    |
578   | 6/6                                    |

As you can see, the importance is always the same for both.

Lets look at another example, this time row **36**. The Values are: <br>
Feature1: **0.017075238**, Feature 2: **0.807275634**

Running the code for the prediction, we get the following results:

Feature | Occurrences of neurons in both lists |
------- | ------------------------------------ | 
1       | 12                                   |
2       | 18                                   |

This time Feature 2 is more important as it is the only high value of the 2
and therefore deciding if the Label is 0 or 1.
