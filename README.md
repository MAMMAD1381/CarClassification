# <b>Introduction</b>
In this project, we aim to explore the use of the ID3 (Iterative Dichotomiser 3) algorithm for classifying car models. By leveraging the power of machine learning and decision tree algorithms, we delve into the realm of predictive modeling to categorize car models based on their unique features and attributes.<br>

# Phase 1: How to Deal with Missing Data
## Imputation vs. Removing Data
When dealing with missing data, data scientists can use two primary methods to solve the error: imputation or data removal.

The imputation method substitutes reasonable guesses for missing data. It’s most useful when the percentage of missing data is low. If the portion of missing data is too high, the results lack natural variation that could result in an effective model.

The other option is to remove data. When dealing with data that is missing at random, the entire data point that is missing information can be deleted to help reduce bias. Removing data may not be the best option if there are not enough observations to result in a reliable analysis. In some situations, observation of specific events or factors may be required, even if incomplete.

## <b>Deletion</b>
In this method, all data for an observation that has one or more missing values are deleted. The analysis is run only on observations that have a complete set of data. If the data set is small, it may be the most efficient method to eliminate those cases from the analysis. However, in many cases, the data are not missing completely at random. Deleting the instances with missing observations can result in biased parameters and estimates and reduce the statistical power of the analysis.

## <b>Imputation</b>

### Mean, Median and Mode
One of the most common methods of imputing values when dealing with missing data. In cases where there are a small number of missing observations. The downside is that only mode works on categorical data.

|          | model | year | price | transmission | mileage | fuelType | tax | mpg | engineSize | Manufacturer |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| original    | Aygo   | <span style="color:#ff4444">None</span>   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | <span style="color:#ff4444">None</span>   | Toyota   |
| using mean   | Aygo   | **2017.02**   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | **1.60**   | Toyota   |
| using median   | Aygo   | **2017**   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | **1.5**   | Toyota   |
| using mode   | Aygo   | **2017**   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | **2**   | Toyota   |
<br>


### <b>Time-Series Specific Methods</b>
The time series methods of imputation assume the adjacent observations will be like the missing data. These methods work well when that assumption is valid.
### Last Observation Carried Forward (LOCF) & Next Observation Carried Backward (NOCB)
In these methods, every missing value is replaced with either the last observed value or the next one. Longitudinal data track the same instance at different points along a timeline. This method is easy to understand and implement.

|          | model | year | price | transmission | mileage | fuelType | tax | mpg | engineSize | Manufacturer |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| last data   | Q2   | 2019   | 31270   | Semi-Auto   | 5446   | Petrol   | 145   | 33.2   | 2   | Audi   |
| original    | Aygo   | <span style="color:#ff4444">None</span>   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | <span style="color:#ff4444">None</span>   | Toyota   |
| next data    |  GLA Class   | 2015   | 18599   | Semi-Auto   | 24927   | Diesel   | 125   | <span style="color:#ff4444">None</span>   | <span style="color:#ff4444">None</span>   | Mercedes   |
| using LOCF    | Aygo   | <b>2019</b>   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | <b>2</b>   | Toyota   |
| using NOCB    | Aygo   | <b>2015</b>   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | <span style="color:#ff4444">None</span>   | Toyota   |

<br>As you can see it's not a viable method when the neighboring data have missing values as well. Also, this method may introduce bias when data has a visible trend. It assumes the value is unchanged by the missing data.

### <b>Linear Interpolation</b>
Linear interpolation is often used to approximate a value of some function by using two known values of that function at other points. This formula can also be understood as a weighted average. The weights are inversely related to the distance from the endpoints to the unknown point. The closer point has more influence than the farther point. 

|          | model | year | price | transmission | mileage | fuelType | tax | mpg | engineSize | Manufacturer |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|
| original   | Aygo   | <span style="color:#ff4444">None</span>   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | <span style="color:#ff4444">None</span>   | Toyota   |
| After Linear Interpolation   | Aygo   | **2017.0**   | 5580   | Manual   | 27653   | Petrol   | 0   | 69   | **1.875**   | Toyota   |



# Phase 2
The ID3 algorithm works by building a decision tree, which is a hierarchical structure that classifies data points into different categories and splits the dataset into smaller subsets based on the values of the features in the dataset. The ID3 algorithm then selects the feature that provides the most information about the target variable. The decision tree is built top-down, starting with the root node, which represents the entire dataset. At each node, the ID3 algorithm selects the attribute that provides the most information gain about the target variable. The attribute with the highest information gain is the one that best separates the data points into different categories.

## <b>General steps in ID3</b>
1. Determine entropy for the overall the dataset using class distribution.
2. For each feature.<br>
    <ul>Discretization of continuous variables</ul>
    <ul>Calculate Entropy for Categorical Values</ul>
    <ul>Assess information gain for each unique categorical value of the feature</ul>
3. Choose the feature that generates highest information gain.
4. Iteratively apply all above steps to build the decision tree structure.

## <b>Implementation on our Dataset</b>
As stated in the previous section the first step is to find the best feature i.e. the one that has the maximum Information Gain(IG). in order to do this we follow this steps:

1. Calculate Entropy: The algorithm starts by calculating the entropy of the target variable (e.g., classifying cars as "good" or "bad"). Entropy is a measure of impurity in the data. The goal is to split the data in a way that reduces entropy and increases homogeneity within each subset.
2. Calculate Information Gain: For each attribute (e.g., "engine type," "fuel efficiency," "number of doors"), the algorithm calculates the information gain. Information gain measures how much the attribute contributes to reducing entropy in the dataset. The attribute with the highest information gain is selected as the splitting criterion at that node.
3. Split the Data: The dataset is split based on the selected attribute, creating subsets for each unique value of that attribute.
Repeat: The process is repeated recursively for each subset until a stopping criterion is met (e.g., maximum tree depth reached, all instances in a subset belong to the same class).
4. Build the Decision Tree: The result is a decision tree where each internal node represents an attribute test, each branch represents the outcome of the test, and each leaf node represents the class label.

### <b>Discretization of continuous variables</b>
Discretization involves dividing the range of a continuous variable into intervals or bins. Once the continuous variable is discretized, it can be treated as a categorical attribute.

In this project we selected a single threshold and divided the range of the continuous variable into two bins based on the specific threshold value.This process simplifies the continuous variable into a binary attribute, where values below the threshold are assigned to one bin and values equal to or above the threshold are assigned to another bin.

### <b>here are the steps to optimize your threshold:</b>
    1. Select a Threshold: The first step is to choose a threshold value that will be used to split the continuous variable into two bins.
    2. Split the Data:Next, the continuous variable is split into two bins based on the selected threshold.
    3. Calculating information gain: Next we need a comparison criteria to find the best threshold, which is why we use information gain to compare final results.
    4. Choosing the optimal threshold: After repeating previous steps we'll have a list of thresholds with their corresponding ig values to find an optimal threshold which is done by selecting the maximum ig.


## <b>When Entropy and Information Gain are maximized?</b>

### Entropy
    Entropy is maximized when the data set is completely impure or heterogeneous. This occurs when the distribution of the target variable classes is evenly spread across the dataset. When the entropy is at its maximum, it means that there is maximum uncertainty or disorder in the dataset, and the goal of the ID3 algorithm is to split the data in a way that reduces this entropy by creating subsets that are more homogeneous in terms of the target variable.

    example:
                                                - Root: ( 60 , 60 )
                      - Left Child: ( 40 , 40 )                     - Right Child: ( 20 , 20 )
   
    in this example the entropy would be at its maximum because the data is perfectly impure. 
               

### Information Gain
    Information gain is maximized when a split in the data results in subsets that are as pure as possible in terms of the target variable. 

                                                    - Root: ( 60 , 60 )
                      - Left Child: ( 60 , 0 )                             - Right Child: ( 0 , 60 )

# Phase 3

## Why Does Overfitting Occur in Decision Trees?

Overfitting in decision tree models occurs when the tree becomes too complex and captures noise or random fluctuations in the training data, rather than learning the underlying patterns that generalize well to unseen data. Other reasons for overfitting include:
1. **High Variance:** Decision trees have high variance, meaning they are sensitive to the noise in the training data. They can create overly complex trees that capture the noise along with the underlying patterns in the data.
2. **Deep Trees:** If a decision tree is allowed to grow without any constraints, it can become very deep and detailed, capturing even the smallest variations in the training data. This can lead to overfitting, where the model performs well on the training data but poorly on unseen data.
3. **Memorizing the Training Data:** Decision trees have the capacity to memorize the training data, especially if the tree is allowed to grow without pruning or regularization. This memorization can result in poor generalization to new, unseen data.
4. **Lack of Generalization:** Decision trees can create overly specific rules to classify the training data, which may not generalize well to new data. This lack of generalization is a common cause of overfitting.
5. **Sensitive to Outliers:** Decision trees can be sensitive to outliers in the training data, leading to the creation of split points that are influenced by these outliers. This can result in overfitting if the model tries to accommodate these outliers too closely.
6. **Sample Bias:** If the training dataset is not representative, decision trees may overfit to the training data’s idiosyncrasies, resulting in poor generalization.
7. **Discretization of continuous variables:** In this step of decision tree if we use mutiple thresholds to split the data, overfitting might occure and it is better to use a single threshold to make a binary set of data.
  
## Strategies to Overcome Overfitting in Decision Tree Models

### **Utilizing Weighted Probability for Leaf Label Selection**
Using weighted probability in choosing a leaf label in decision trees can help prevent overfitting by introducing a form of regularization that biases the model towards more generalizable predictions.<br>

 Here's how using weighted probability can aid in avoiding overfitting:

1. **Balancing Prediction Confidence:** By assigning weights to the probabilities of different leaf labels, you can control the confidence level of the model's predictions. Higher weights can be assigned to labels that are more likely to occur in the training data, while lower weights can be assigned to less frequent labels. This helps in balancing the model's predictions and prevents it from overfitting to rare or noisy patterns in the data.
2. **Regularization:** Introducing weights to the leaf labels acts as a form of regularization by penalizing overly confident predictions. By adjusting the weights based on the frequency or importance of each label, you can encourage the model to make more conservative predictions and avoid memorizing noise in the training data.
3. **Handling Class Imbalance:** Weighted probability can be particularly useful in scenarios with class imbalance, where certain labels are underrepresented in the training data. By assigning higher weights to minority class labels, the model is encouraged to pay more attention to these classes and avoid biased predictions towards the majority class.
4. **Controlling Model Complexity:** Weighted probability can help control the complexity of the decision tree model by influencing the final predictions at the leaf nodes. By adjusting the weights based on the importance of each label, you can guide the model towards more generalizable and robust predictions.
4. **Improving Generalization:** By using weighted probability to guide the model's decision-making process, you can improve its generalization performance on unseen data. The regularization effect of weighted probability helps in creating a more balanced and stable model that is less prone to overfitting.

### **Pruning Techniques**
Pruning involves removing parts of the decision tree that do not contribute significantly to its predictive power. This helps simplify the model and prevent it from memorizing noise in the training data. Pruning can be achieved through techniques such as pre-pruning and post-pruning. Here's an explanation of both:
1. **Pre-Pruning:**
        <ul>**Early Stopping:** In pre-pruning, the decision tree algorithm is stopped early before it becomes too complex. This can be done by setting a maximum depth for the tree, limiting the number of leaf nodes, or specifying a minimum number of samples required to split a node.</ul>
        <ul>**Minimum Samples per Leaf:** By setting a minimum number of samples required to be at a leaf node, pre-pruning prevents the tree from splitting further if the number of samples is below the specified threshold.</ul>
        <ul>**Maximum Features:** Limiting the number of features considered for each split can also help prevent overfitting by reducing the complexity of the tree.</ul>
2. **Post-Pruning:**
        <ul>**Cost Complexity Pruning (CCP):** Post-pruning involves growing a full decision tree and then pruning it back to find the optimal tree size. CCP assigns a cost to each node based on the improvement in impurity (e.g., Gini impurity) and the number of samples at the node. By iteratively removing nodes with the smallest cost, the tree is pruned to find the right balance between complexity and accuracy.</ul>
        <ul>**Reduced Error Pruning:** This method involves pruning the tree by removing nodes that do not improve the overall accuracy of the tree on a separate validation dataset. Nodes are removed if the error rate does not significantly increase after pruning.</ul>
### **Limiting Tree Depth**
Setting a maximum depth for the decision tree restricts the number of levels or branches it can have. By restricting the depth of the tree, you can control the complexity of the model and improve its generalization performance. Here's how limiting tree depth helps avoid overfitting:

1. **Simplifies the Model:** Limiting the tree depth prevents the model from creating overly complex and detailed decision rules. A shallow tree is simpler and captures the most important patterns in the data without memorizing noise or outliers.
2. **Reduces Variance:** By restricting the depth of the tree, you reduce the variance of the model. A deep tree with many levels can capture noise and specific details of the training data, leading to overfitting. Limiting the depth helps in creating a more generalized model.
3. **Improves Interpretability:** Shallow trees are easier to interpret and understand. Limiting the tree depth makes the decision-making process more transparent and intuitive, which can be beneficial for explaining the model to stakeholders or domain experts.
4. **Faster Training and Inference:** Deep trees can be computationally expensive to train and evaluate. By limiting the tree depth, you reduce the computational cost of building and using the model, making it more efficient for real-time applications.
5. **Better Generalization:** A shallow tree with limited depth is more likely to generalize well to unseen data. It focuses on capturing the most significant patterns in the data, leading to better performance on new instances.

### **Cross-Validation**
This is another method that could help us greatly if our test data wasn't splitted from the start.<br>
By splitting the data into training and validation sets multiple times, training the model on different combinations of data, and evaluating its performance, cross-validation helps ensure that the model generalizes well to unseen data and is not overfitting.<br>
Here's how cross-validation helps avoid overfitting:
1. **Training and Validation:** In cross-validation, the dataset is split into multiple subsets or folds. The model is trained on a subset of the data and validated on a different subset. This process is repeated multiple times, with each subset serving as both the training and validation data at different points.
2. **Model Evaluation:** By training the model on different subsets of the data and evaluating its performance on separate validation sets, cross-validation provides a more robust estimate of the model's performance. It helps in detecting overfitting by assessing how well the model generalizes to new data.
3. **Generalization Performance:** Cross-validation helps in estimating how well the model will perform on unseen data. By averaging the performance metrics across multiple folds, cross-validation provides a more reliable indication of the model's generalization performance.
4. **Hyperparameter Tuning:** Cross-validation is often used for hyperparameter tuning, where different combinations of model parameters are evaluated on different folds of the data. This helps in selecting the best hyperparameters that optimize the model's performance without overfitting.
5. **Bias-Variance Tradeoff:** Cross-validation helps in understanding the bias-variance tradeoff of the model. It allows you to assess whether the model is underfitting (high bias) or overfitting (high variance) by analyzing its performance across different folds.
6. **Data Efficiency:** Cross-validation makes efficient use of the available data by using all data points for both training and validation. This leads to a more reliable estimate of the model's performance compared to a single train-test split.


### **Ensemble Methods**
Ensemble methods such as Random Forests and Gradient Boosting combine multiple decision trees to reduce overfitting. In Random Forests, each tree is trained on a random subset of the data and features, and predictions are averaged across all trees to improve generalization. Gradient Boosting builds trees sequentially, with each tree correcting the errors of the previous ones, leading to a more accurate and robust model.<br>

Here's how ensemble methods help avoid overfitting:

1. **Reducing Variance:** Ensemble methods, such as Random Forest and Gradient Boosting, combine multiple base models to reduce variance and improve generalization performance. By aggregating the predictions of diverse models, ensemble methods can produce more robust and reliable predictions.
2. **Model Diversity:** Ensemble methods create diverse base models by using different subsets of the data or different algorithms. This diversity helps in capturing different aspects of the underlying patterns in the data and reduces the risk of overfitting to specific noise or outliers.
3. **Combining Weak Learners:** Ensemble methods can combine multiple weak learners (models that perform slightly better than random guessing) to create a strong learner with improved predictive performance. This approach helps in avoiding overfitting by leveraging the collective wisdom of multiple models.
4. **Regularization:** Ensemble methods act as a form of regularization by combining multiple models with different biases and variances. This regularization helps in preventing overfitting by promoting model simplicity and reducing the risk of memorizing noise in the training data.
5. **Voting Mechanisms:** Ensemble methods often use voting mechanisms, such as averaging or taking a majority vote, to combine the predictions of individual models. This aggregation helps in smoothing out individual model predictions and reducing the impact of outliers or noisy data points.
6. **Boosting and Bagging:** Techniques like Boosting (e.g., AdaBoost, Gradient Boosting) and Bagging (e.g., Random Forest) are popular ensemble methods that help in improving model performance and reducing overfitting. These methods leverage the strengths of multiple models to create a more accurate and stable prediction.


# **Result Comparison**
| preprocess method | criterion |  Discretization | train accuracy | test accuracy |
|----------|----------|----------|----------|----------|
| drop all   | entropy  | <span style="color:#57f066">Yes</span>   | 0.97   | 0.36   |
| drop all   | entropy  | <span style="color:#ff4444">No</span>   | 0.93   | 0.54   | 
| Forward fill   | entropy  | <span style="color:#ff4444">No</span>   | 0.81   | 0.56   |
| Forward fill   | Information gain  | <span style="color:#ff4444">No</span>   | 0.81   | 0.59   | 
| Linear Interpolation   | entropy   | <span style="color:#57f066">Yes</span>   | 0.97   | 0.36   | 
| Linear Interpolation   | Information gain   | <span style="color:#ff4444">No</span>   | 0.90   | 0.54   | 
| mean   | Information gain   | <span style="color:#ff4444">No</span>   | 0.82   | 0.60   |

## **Preproccess Comparision**
### Drop All:
With the drop all method, any data point with missing values is completely removed from the dataset before building the decision tree.
    <ul>Pros: It ensures that no assumptions are made about the missing values, and it can prevent bias in the model.</ul>
    <ul>Cons: It can lead to a significant loss of data, especially if there are many missing values, which may result in a smaller dataset for training the decision tree.</ul>
### Mean Imputation:
Mean imputation involves replacing missing values with the mean of the available values in the same column.
    <ul>Pros: It is a simple and quick method that can help retain more data points for training the decision tree.</ul>
    <ul>Cons: It may introduce bias in the data by assuming that the missing values are similar to the mean of the observed values. This can potentially impact the accuracy of the decision tree.</ul>
### Linear Interpolation:
Linear interpolation is a method that estimates missing values based on the values of neighboring data points.
    <ul>Pros: It can provide a more nuanced approach to filling in missing values by considering the trend or pattern in the data.</ul>
    <ul>Cons: It can be computationally more intensive compared to mean imputation, and the accuracy of the interpolation may depend on the distribution and nature of the data.</ul>
### Forward Fill:
Forward fill involves propagating the last observed value forward to fill in missing values.
    <ul>Pros: It can be useful for time series data where missing values are likely to follow a pattern.</ul>
    <ul>Cons: It may not be suitable for all types of data, especially if the missing values are not sequential or do not follow a specific trend.</ul>

### **Summary**
<ul>Drop all ensures no assumptions are made about the missing values but can lead to data loss.</ul>
<ul>Mean imputation is simple and quick but may introduce bias by assuming missing values are similar to the mean.</ul>
<ul>Linear interpolation provides a more nuanced approach by considering trends in the data but can be computationally intensive.</ul>
<ul>Forward fill is useful for sequential data but may not be suitable for all types of datasets.</ul>

## **Entropy Vs. Information Gain**
1. **Interpretation:**
    <ul>Entropy focuses on the disorder or uncertainty in the dataset, while information gain focuses on the information gained about the class variable.</ul>
    <ul>Entropy provides a measure of the impurity of a dataset, while information gain evaluates the usefulness of an attribute for splitting the data.</ul>
2. **Attribute Selection:**
    <ul>Entropy measures the disorder in the dataset and aims to minimize it by selecting attributes that reduce entropy after splitting.</ul>
    <ul>Information gain evaluates the potential of an attribute to provide new information about the class labels and selects attributes that maximize the information gain.</ul>
3. **Handling Attributes:**
    <ul>Entropy can be used to evaluate the homogeneity of a dataset and guide attribute selection based on reducing disorder.</ul>
    <ul>Information gain is specifically designed for decision tree learning and focuses on selecting attributes that maximize the information gained about the class variable.</ul>

## **How discretization can result in overfitting**

1. **Loss of Information:**
When continuous attributes are discretized, information about the exact values within each interval is lost. This loss of granularity can lead to a reduction in the amount of information available for the model to learn from. If the discretization is too coarse or if important patterns in the data are lost during the process, the model may struggle to capture the underlying relationships accurately, potentially leading to overfitting.
2. **Increased Model Complexity:**
Discretization can introduce additional complexity to the model, especially if the number of bins or intervals created is high. With more discrete features, the model may try to fit the training data too closely, capturing noise or outliers that are not representative of the underlying patterns in the data. This increased complexity can result in overfitting, where the model performs well on the training data but fails to generalize to unseen data.
3. **Inflexibility in Handling Outliers:**
Discretization can be sensitive to outliers, as they may fall into their own separate bins or intervals. If outliers are not handled appropriately during the discretization process, they can lead to the creation of bins that are specific to these extreme values. This can cause the model to overfit to the outliers, as it tries to capture the variability introduced by these extreme values.
4. **Limited Adaptability to New Data:**
Once continuous attributes are discretized, the resulting bins or intervals are fixed and do not adapt to new data points that fall outside the predefined ranges. This lack of adaptability can limit the model's ability to generalize to unseen data that may have values outside the original bins. As a result, the model may struggle to make accurate predictions on new data, leading to overfitting on the training set.