- [Simple Linear Regression](#simple-linear-regression)
  - [Conditional Mean and Expectation](#conditional-mean-and-expectation)
  - [Function to predict y](#function-to-predict-y)
  - [Function of Residuals](#function-of-residuals)
  - [Function of Residual Sum of Squared Error and OLS](#function-of-residual-sum-of-squared-error-and-ols)
  - [Prediction](#prediction)
  - [Interpretation](#interpretation)
    - [R-Squared](#r-squared)
    - [Calculation of R-Squared](#calculation-of-r-squared)
- [Multiple Linear Regression (MLR)](#multiple-linear-regression-mlr)
  - [Assumptions of Linear Regression](#assumptions-of-linear-regression)
    - [**Linearity**](#linearity)
    - [**Homoscedasticity**](#homoscedasticity)
    - [Normality of the Error Terms](#normality-of-the-error-terms)
    - [**No Autocorrelation between Error Terms**](#no-autocorrelation-between-error-terms)
    - [Multicollinearity among Predictors](#multicollinearity-among-predictors)
  - [Notations and Matrix Representation of Linear Regression](#notations-and-matrix-representation-of-linear-regression)
  - [Break down of the Matrix Representation](#break-down-of-the-matrix-representation)
  - [Optimal $\beta$ - Normal Equation](#optimal-beta---normal-equation)
  - [Prediction](#prediction-1)
  - [Feature Scaling](#feature-scaling)
  - [Hypothesis Testing on $\beta$](#hypothesis-testing-on-beta)
    - [T-Statistics](#t-statistics)
  - [Time Complexity](#time-complexity)
  - [**Preamble for the next series on Linear Regression**](#preamble-for-the-next-series-on-linear-regression)
  - [**Orthogonalization**](#orthogonalization)
  - [**Regularization**](#regularization)
  - [**Statistical and Interpretation of Linear Regression**](#statistical-and-interpretation-of-linear-regression)
  - [Python Implementation](#python-implementation)
  - [References and Citations](#references-and-citations)

Our dataset is fairly simple, here is a brief overview of the first five rows of it.

```python
sqft	bdrms	age	price
	2104	3	70	399900
	1600	3	28	329900
	2400	3	44	369000
	1416	2	49	232000
	3000	4	75	539900
```

The columns are:

```
sqft: the size of the house in sq. ft
bdrms: number of bedrooms
age: age in years of house
price: the price of the house
```

In the following sections, we will divide the price by $1000$, this is to pseudo-standardize the data. One note that we may bring forward in the section is that in Linear Regression, we generally don't need to standardize or center the predictors (see proof in Section: Feature Scaling).

# Simple Linear Regression

We will first start off by constructing the Simple Linear Regression (SLR).

$$y = \beta_0 + \beta_1 x + \epsilon$$

$$\text{price} = \beta_0 + \beta_1 \text{sqft} + \epsilon$$

where $\beta_0$ is the intercept, $\beta_1$ the coefficient and $\epsilon$ is the error term. The error term can be thought of as the errors in the random universe, and makes up for the deviations in the model that cannot be prevented, we will not touch too much on the $\epsilon$ and will take it as the difference between the predicted and true values which are not being explained by the variables $x$ in the SLR model.

---

Without the intercept term the regression line would always have to pass through the origin, which is almost never an optimal way to represent the relationship between our target and predictor variable. This concept is closely linked to the reason why a model must always have a bias term. 

> Bias: If there is no bias term in the model, you will lose the flexibility of your model. Imagine a simple linear regression model without a bias term, then your linear equation $y=mx$ will only pass through the origin. Therefore, if your underlying data (pretend that we know that the underlying data's actual function $y = 3x + 5$), then your Linear Regression model will never find the "best fit line" simply because we already assume that our prediction is governed by the slope $m$, and there does not exist our $c$, the intercept/bias term.

Therefore, it is usually the case whereby we always add in an intercept term. We intend to estimate the values of $y$ ***given*** $x$. Each value of $x$ is multiplied by $\beta_{1}$, with a constant intercept term $\beta_{0}$. We end this section by knowing that 1 unit increase in $x$ will correspond to a $\beta_1$ unit increase in $y$ according to the model, while always remebering to add the intercept term.

---

## Conditional Mean and Expectation

Something that most formal textbooks will mention is that a linear regression model is predicted via a conditional expectation 

$$E(y|x)=\beta_0 + \beta_1 x + \epsilon$$


In the probability model underlying linear regression, X and Y *are* random variables.

> if so, as an example, if Y = obesity and X = age, if we take the conditional expectation $E(Y|X=35)$ meaning, whats the expected value of being obese if the individual is 35 across the sample, we just take the average(arithmetic mean) of y for those observations where X=35?

But this means that each yi has in principle a different expected value : so the yi's here do not come from an identically distributed population. If they don't, then our sample {Y,X}, that contains as a random variable only the Y is not "random" (i.e. it is not i.i.d), due to the assumption that the X's are deterministic.

So given a prediction vector y hat, this set of y hat (note not just single prediction) gives rise to the lowest L2 Loss, note again, this "unique" set of y hat will give rise to the lowest L2 loss. Now, this set of y hat is also the conditional mean of y given the training set X, technically, say our yhat[0] = 2.3, then this means, the corresponding X[0], say 1.5 (only 1 feature), is corresponding to this point 2.3, and this means on average, points residing with the input x = 1.5, will give an expectation value of 2.3. Now, if we go to yhat[1] = 3.3, with X[1] = 1.8, then the same logic applies.

For more intuition, refer to the `lecture/conditional_mean`, note this is an often overlooked information and one should not neglect it.

---

## Function to predict y

An simple formula that predicts the value of the house ($\hat{y}$) given input variable square feet ($x$) is as follows:

$$\hat{y} = \beta_0 + \beta_1 x$$

There are no more error terms $\epsilon$ and rightfully so, because if our model can know the unknown errors, it will be a perfect model, however, in reality, this SLR is just an estimation of $y$.

## Function of Residuals

The definition of residuals is easy, it is simply the difference between the predicted $y$ value and the actual $y$ value. For more rigorous understanding, see the notes I made for myself. 

$$\text{Residuals}_{i} = y_i - \hat{y}_i ~ \forall i$$

## Function of Residual Sum of Squared Error and OLS

This is important to understand. Informally, we call this RSS, whereby it is a function $J(\beta)$ that we want to minimize on. We usually call $J(\beta)$ the loss function, as we want to minimize the loss. Slightly more formally, we can say that we want to find the optimal $\beta$ that gives rise to the minimum of $f = \text{RSS}$. Mathematically, we express this as $\text{argmin}_{\beta \in \R}J(\beta)$. We will touch on this later in Multiple Linear Regression, where a more rigorous form is being presented, and talking about its global minimum.

**In other words, we want to find the** $\beta_0$ and $\beta_1$ such that, $J(\beta)$ is at a minimum. We choose a reasonable function $J(\beta)$ to be the Residual Sum of Squared Error. We will solve for $J(\beta)$ and get the best $\beta$ so that our predicted $\hat{y}$ will be as close to the ground truth $y$ as possible.

As of now, we denote our loss function $J(\beta)$ as follows: 

$$J(\beta)=\sum(y-\hat{y})^2=\sum(y-\beta_0-\beta_1x)^2 = \text{RSS}=\text{SSR}$$

The reason we chose such a function is because of its convexity, and of course, that it is also the well known Ordinary Least Squares Estimator of $\beta$.  In addition, why the squared residuals instead of just the absolute value of the residuals? Well, both can be used – absolute value of residuals is often used when there are large outliers or other abnormalities in variables. [Solving for the least absolute deviations (LAD)](https://en.wikipedia.org/wiki/Least_absolute_deviations) is a type of "robust" regression.

In High School Calculus, we recall that to find a minimum of a function $J(\beta)$, we take the derivative and set it to 0. We do the exact same thing here:

$$\dfrac{dS}{d\beta_1} = -2\sum x(y-\beta_0-\beta_1x)\\
\dfrac{dS}{d\beta_0} = -2\sum (y-\beta_0-\beta_1 x)$$

After setting both the equations to 0 and solving it, we note that $J(\beta)$ is a convex function, and therefore a minima is guaranteed. We present the optimal $\beta$ that minimizes the loss function $J(\beta)$:

$$\hat{\beta}_1 = \dfrac{\sum(x-\bar{x})(y-\bar{y})}{(x-\bar{x})^2} = r_{XY} \frac{s_Y}{s_X}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

where 

- $\bar{y}$ : the sample mean of observed values $Y$
- $\bar{x}$ : the sample mean of observed values $X$
- $s_Y$ : the sample standard deviation of observed values $Y$
- $s_X$: the sample standard deviation of observed values $X$
- $r_{XY}$: the sample Pearson correlation coefficient between observed $X$ and $Y$

Note that I will not continue to put a hat notation on top of the parameters $\beta$, but keep in mind that the $\beta$ we found are not the population parameter, instead it is the sample parameter. In statistics, we often call $\beta$ as the **true population parameter**, but in reality, we do not have knowledge of what the underlying **parameter** is, and therefore we minimize the loss function to find the best estimate for the true population parameter $\beta$, we denote it as $\hat{\beta}$ and call it a **statistics**.

We end this section off with a note that this method is called the Ordinary Least Squares (OLS).

---

## Prediction

Since we have the formula to calculate $\beta_0, \beta_1$, we can use `python` to do the dirty work for us. For the full code, please refer to **appendix**. 

$$\hat{y} = 71+0.135x$$

Hereby attached is also a nice plot visualization. Some explanation is as follows: The graph below is 3 graphs stacked together - the blue dots represent $(x_i, y_i)$ where it represents a scatter plot of the original values of x and its ground truth y, one can observe that the scatter plot of the original dataset vaguely describes a linear relationship; the red dots represent $(x_i, \hat{y}_i)$ represents a scatter plot of the x and the predicted values $\hat{y}$; last but not least, we draw the best fit line across.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1333906b-d1c3-4933-84b6-f4d1baa14fbb/newplot_(1).png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1333906b-d1c3-4933-84b6-f4d1baa14fbb/newplot_(1).png)

Scatter plot

---

## Interpretation

One thing that is worth highlighting is I did a simple standardization of the $y$ value through a division of 1000. Therefore, our predicted SLR model says that given a constant intercept of $71 \times 1000 = 71000$, we expect that every unit increase of square feet brings about an increase of price of $0.135 \times 1000 = 135$ dollars. In other words, if you were to purchase a house which is $100$ square feet more than your current house, be ready to fork out an additional $13500$ bucks. (Hmm, kinda cheap though 😂, my country Singapore has way higher housing price than this 😐)

We also can calculate the loss function, or preferably the Residual Sum of Squares (SSR), to be $193464$. Note that this is the lowest number that this model can get, although it seems high, but mathematically, there does not exist a number smaller than the aforementioned, solely because we already minimized the loss function to its global minimum.

### R-Squared

We can't leave SLR without discussing the most notable metrics to access the model's performance called **[R-Squared](https://blog.minitab.com/en/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)**. Although in all seriousness, it may no longer be the "best" metric due to the following two reasons from [Reference from Minitab](https://blog.minitab.com/en/adventures-in-statistics-2/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables#:~:text=The%20adjusted%20R%2Dsquared%20is,less%20than%20expected%20by%20chance.)

1. Every time you add a predictor to a model, the R-squared increases, even if due to chance alone. It never decreases. Consequently, a model with more terms may appear to have a better fit simply because it has more terms.
2. If a model has too many predictors and higher order polynomials, it begins to model the random noise in the data. This condition is known as [overfitting the model](https://blog.minitab.com/blog/adventures-in-statistics/the-danger-of-overfitting-regression-models) and it produces misleadingly high R-squared values and a lessened ability to make predictions.

However, we are still going to go through the motion and discuss it. **(Reference from Minitab.)**

R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model → R-squared = Explained variation / Total variation

R-squared is always between 0 and 100% where

- 0% indicates that the model explains none of the variability of the response data around its mean.
- 100% indicates that the model explains all the variability of the response data around its mean.

In general, the higher the R-squared, the better the model fits your data. But do remember that the more predictors you add to a LR model, the R-Squared will going to increase regardless.

### Calculation of R-Squared

This is simple enough, the formula is given by:

$$R^2=1-\frac{\text{SSR}}{\text{SST}}$$

where 

The total sum of squares is defined:

$$\text{SST}= \sum_{i=1}^n \left(y_i - \bar{y}\right)^2$$

The residual sum of squares you are already familiar with. It is defined:

$$\text{SSR} = \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2$$

With the use of `python` , $R^2=0.73$

---

# Multiple Linear Regression (MLR)

Instead of using just one predictor to estimate a continuous target, we build a model with multiple predictor variables. You will be using MLR way more than SLR going forward. Remember the dataset on house pricing? We just now used the most obvious one to predict the **price of a house**, given the predictor variable , **square feet.** However, there are two other variables, **number of bedrooms** and **age of the house**. We can instead model these two variables alongside with **square feet** to predict the **price** of the house. In general, if these variables all play a crucial role in affecting the **price** of the house, then including these 3 variables will make the model more accurate. However, we have to take note of a few **important assumptions** of MLR (SLR included). We will mention it here.

[Reference]: [Linear Regression Assumptions by Jeff Macaluso from Microsoft](https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/)

---

## [Assumptions of Linear Regression](https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/)

### **Linearity**

This assumes that there is a linear relationship between the predictors (e.g. independent variables or features) and the response variable (e.g. dependent variable or label). This also assumes that the predictors are additive.

**Why it can happen:** There may not just be a linear relationship among the data. Modeling is about trying to estimate a function that explains a process, and linear regression would not be a fitting estimator (pun intended) if there is no linear relationship.

**What it will affect:** The predictions will be extremely inaccurate because our model is [underfitting](https://cdn-images-1.medium.com/max/1125/1*_7OPgojau8hkiPUiHoGK_w.png). This is a serious violation that should not be ignored.

**How to detect it:** If there is only one predictor, this is pretty easy to test with a scatter plot. Most cases aren’t so simple, so we’ll have to modify this by using a scatter plot to see our predicted values versus the actual values (in other words, view the residuals). Ideally, the points should lie on or around a diagonal line on the scatter plot.

**How to fix it:** Either adding polynomial terms to some of the predictors or applying nonlinear transformations . If those do not work, try adding additional variables to help capture the relationship between the predictors and the label.

---

### **Homoscedasticity**

This assumes homoscedasticity, which is the same **variance** within our error terms. Heteroscedasticity, the violation of homoscedasticity, occurs when we don’t have an even **variance** across the error terms.

**Why it can happen:** Our model may be giving too much weight to a subset of the data, particularly where the error variance was the largest.

**What it will affect:** Significance tests for coefficients due to the standard errors being biased. Additionally, the confidence intervals will be either too wide or too narrow.

**How to detect it:** Plot the residuals and see if the variance appears to be uniform.

**How to fix it:** Heteroscedasticity (can you tell I like the *scedasticity* words?) can be solved either by using [weighted least squares regression](https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares) instead of the standard OLS or transforming either the dependent or highly skewed variables. Performing a log transformation on the dependent variable is not a bad place to start.

---

### Normality of the Error Terms

More specifically, this assumes that the error terms of the model are **normally distributed**. Linear regressions other than [Ordinary Least Squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares) may also assume normality of the predictors or the label, but that is not the case here.

**Why it can happen:** This can actually happen if either the predictors or the label are significantly non-normal. Other potential reasons could include the linearity assumption being violated or outliers affecting our model.

**What it will affect:** A violation of this assumption could cause issues with either shrinking or inflating our confidence intervals.

**How to detect it:** There are a variety of ways to do so, but we’ll look at both a histogram and the p-value from the Anderson-Darling test for normality.

**How to fix it:** It depends on the root cause, but there are a few options. Nonlinear transformations of the variables, excluding specific variables (such as long-tailed variables), or removing outliers may solve this problem.

---

### **No Autocorrelation between Error Terms**

This assumes no autocorrelation of the error terms. Autocorrelation being present typically indicates that we are missing some information that should be captured by the model.

**Why it can happen:** In a time series scenario, there could be information about the past that we aren’t capturing. In a non-time series scenario, our model could be systematically biased by either under or over predicting in certain conditions. Lastly, this could be a result of a violation of the linearity assumption.

**What it will affect:** This will impact our model estimates.

**How to detect it:** We will perform a [Durbin-Watson test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic) to determine if either positive or negative correlation is present. Alternatively, you could create plots of residual autocorrelations.

**How to fix it:** A simple fix of adding lag variables can fix this problem. Alternatively, interaction terms, additional variables, or additional transformations may fix this.

---

### Multicollinearity among Predictors

This assumes that the predictors used in the regression are not correlated with each other. This won’t render our model unusable if violated, but it will cause issues with the interpretability of the model. This is why in the previous section, we need to make sure that the 3 variables, **square feet, number of bedrooms, age of house** are not highly correlated with each other, else additive effects may happen.

**Why it can happen:** A lot of data is just naturally correlated. For example, if trying to predict a house price with square footage, the number of bedrooms, and the number of bathrooms, we can expect to see correlation between those three variables because bedrooms and bathrooms make up a portion of square footage.

**What it will affect:** Multicollinearity causes issues with the interpretation of the coefficients. Specifically, you can interpret a coefficient as “an increase of 1 in this predictor results in a change of (coefficient) in the response variable, holding all other predictors constant.” This becomes problematic when multicollinearity is present because we can’t hold correlated predictors constant. Additionally, it increases the standard error of the coefficients, which results in them potentially showing as statistically insignificant when they might actually be significant.

**How to detect it:** There are a few ways, but we will use a heatmap of the correlation as a visual aid and examine the [variance inflation factor (VIF)](https://en.wikipedia.org/wiki/Variance_inflation_factor).

**How to fix it:** This can be fixed by other removing predictors with a high variance inflation factor (VIF) or performing dimensionality reduction.

---

## Notations and Matrix Representation of Linear Regression

**[Reference to Stanford and Andrew Ng, both different notations]**

We first establish that our regression model is defined as

$$\left|
\begin{array}{l}
\mathbf{y} = \mathbf{X} \mathbf{\beta} + \mathbf{\varepsilon} \\
 \mathbf{\varepsilon} \sim N(0, \sigma^2 \mathbf{I})
\end{array}
\right.$$

---

where

- **X** **is the Design Matrix:** Let **X** be the design matrix of dimensions *m* × (*n* + 1) where *m* is the number of observations (training samples) and *n* independent feature/input variables.

$$\mathbf{X} = \begin{bmatrix} 1 &  x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
                1 &  x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ 
                \vdots & \vdots & \vdots & \vdots & \vdots \\
                1 &  x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \end{bmatrix}_{m \times (n+1)} = \begin{bmatrix} (\mathbf{x^{(1)}})^{T} \\ (\mathbf{x^{(2)}})^{T} \\ \vdots \\ (\mathbf{x^{(m)}})^{T}\end{bmatrix}$$

- The *ith* column of **X** is defined as $x^{(i)}$, which is also known as the *i*th training sample, represented as a *n* × 1 vector.

    $$\mathbf{x^{(i)}} = \begin{bmatrix} x_1^{(i)} \\ x_2^{(i)} \\ \vdots \\ x_n^{(i)} \end{bmatrix}_{n \times 1}$$

    where $x^{(i)}_j$ is the value of feature *j* in the *i*th training instance.

- **y** **the output vector:** The column vector **y** contains the output for the *m* observations.

    $$\mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}_{m \times 1}$$

- **β** **the vector of coefficients/parameters:** The column vector **β** contains all the coefficients of the linear model.

    $$\mathbf{\beta}=\begin{bmatrix} \beta_ 1 \\ \beta_2 \\ \vdots \\ \beta_n\end{bmatrix}_{n \times 1}$$

- **ε** **the random vector of the error terms:** The column vector **ε** contains *m* error terms corresponding to the *m* observations.

    $$\mathbf{\varepsilon} = \begin{bmatrix} \varepsilon^{(1)} \\ \varepsilon^{(2)} \\ \vdots \\ \varepsilon^{(m)} \end{bmatrix}_{m \times 1}$$

As we move along, we will make slight modification to the variables above, to accommodate the intercept term as seen in the Design Matrix.

On a side note, we present another way to represent the above vectors and matrix, the above is the Machine Learning way, while below is the more Statistical way.

$$\mathbf{X} = \begin{bmatrix} 1 &  x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
                1 &  x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\ 
                \vdots & \vdots & \vdots & \vdots & \vdots \\
                1 &  x_{m,1} & x_{m,2} & \cdots & x_{m,n} \end{bmatrix}_{m \times (n+1)}$$

---

## Break down of the Matrix Representation

Our dataset has 47 samples, we can generalize it further to a data set with *m* independent observations, $$(x^{(1)}, y^{(1)}), (**x^{(**2)}, *y^{*(2)), ..., (**x**(*m*), *y*(*m*))

where **x**(*i*) is a *m* × 1 vector, and *y*(*i*) a scalar. 

A **multivariate linear regression** problem between an input variable $x^{(i)}$ and output variable $y^{(i)}$ can be represented as such:

$$y^{(i)} = β_0 + β_1x_1^{(i)} + ... + β_nx_n^{(i)} + ε^{(i)}   \text{where } ε^{(i)}\sim^{\text{i.i.d}}N(0, σ^2)$$

Since there exists *m* observations, we can write an equation for each observation:

$$y^{(1)} = β_0 + β_1x_1^{(1)} + ... + β_nx_n^{(1)} + ε^{(1)}\\
y^{(2)} = β_0 + β_1x_1^{(2)} + ... + β_nx_n^{(2)} + ε^{(2)}\\
\vdots\\
y^{(m)} = β_0 + β_1x_1^{(m)} + ... + β_nx_n^{(m)} + ε^{(m)}\\$$

However, linear regression model usually have an intercept term, it is necessary to include a constant variable term $\mathbf{x_{0}} = 1_{m × 1}$ such that our linear regression can be expressed compactly in matrix algebra form. Adding the intercept term $x_0$, we have the following:

$$y^{(1)} = β_0x_0^{(1)} + β_1x_1^{(1)} + ... + β_nx_n^{(1)} + ε^{(1)}\\
y^{(2)} = β_0 x_0^{(2)}+ β_1x_1^{(2)} + ... + β_nx_n^{(2)} + ε^{(2)}\\
\vdots\\
y^{(m)} = β_0x_0^{(m)}+ β_1x_1^{(m)} + ... + β_nx_n^{(m)} + ε^{(m)}\\$$

We transform the above system of linear equations into matrix form as follows:

$$
\begin{bmatrix} y^{(1)}  \\ y^{(2)} \\ y^{(3)} \\ \vdots \\ \mathbf{y}^{(m)} \end{bmatrix}_{m \times 1} = \begin{bmatrix} 1 &  x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
                1 &  x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\ 
                \vdots & \vdots & \vdots & \vdots & \vdots \\
                1 &  x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)} \end{bmatrix}_{m \times (n+1)} \begin{bmatrix} \beta_0 \\ \beta_ 1 \\ \beta_2 \\ \vdots \\ \beta_n\end{bmatrix}_{(n+1) \times 1} + \begin{bmatrix} \varepsilon^{(1)} \\ \varepsilon^{(2)} \\ \varepsilon^{(3)} \\ \vdots \\ \varepsilon^{(m)} \end{bmatrix}_{m \times 1}$$

We then write the above system of linear equations more compactly as **y** = **Xβ** + **ε**   where $ε\sim^{\text{i.i.d}}N(0, σ^2)$ recovering back the equation at the start.

This is assumed to be an accurate reflection of the real world. The model has a systematic component **Xβ** and a stochastic component **ε**. Our goal is then to obtain estimates of the population parameter **β**.

---

## Optimal $\beta$ - Normal Equation

Just as in SLR, we aim to minimize the loss function (note if you average across the samples, we also call them the cost function, but at this stage, we do not differentiate the two words). We introduce a new approach to solve the loss function analytically, called the **Normal Equation.** This method solves for us easily the optimal $\beta$ to be $\beta = (X^TX)^{-1}X^Ty$. A brief derivation will be shown below. **[Reference to Stanford]**

Slightly more formally, we can say that we want to find the optimal $\beta$ that gives rise to the minimum of $f = \text{RSS}$. Mathematically, we express this as $\text{argmin}_{\beta \in \R}J(\beta)$.

Given the notations above, we compute the Sum of Squared Residuals (SSR) to be

$$J(\beta)=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2=\sum_{i=1}^{m}(y_{i}-(\beta_0+\sum_{j=1}^{n}\beta_{j}x_{i,j}))^2$$

To understand the above summation, it is paramount to extend the idea of SLR's SSR function to here, back then, we simply calculate the difference of the `y_true` and `y_hat` for each and every sample, square it, and sum all the residuals up for all the samples. We extend this idea to MLR and realize it is the same formula, just that now the representation of $y$ and $\hat{y}$ are different, as shown above.

With the assumption that $X$is a full rank (invertible) matrix, we can even further reduce the cost/loss function into matrix multiplication (see Normal Equation Derivation II)

$$J(\beta)=\dfrac{1}{2m}(X\beta-y)^T(X\beta-y)$$

We can differentiate the above cost function with respect to each $\beta_i$ and solve the optimal $\beta$. We will not derive the equation here but instead give the result to be 

$$\beta = (X^TX)^{-1}X^Ty$$

---

## Prediction

Here is a snippet of how I calculated the optimal $\beta$ coefficients for our dataset, note that in this example, we did not divide the **house price** by $1000$ as opposed to what we did for SLR. We show that this does not matter in the interpretation. For details, read the next section Feature Scaling, which also applies here.

```python
XtX = np.dot(X.T, X)
XtX_inv = np.linalg.inv(XtX)
XtX_inv_Xt = np.dot(XtX_inv, X.T)
_optimal_betas = np.dot(XtX_inv_Xt, y)
```

$y = 92451+139x_1-8621x_2-81x_3$ where $x_1 = \text{square feet}, x_2 = \text{number of bed rooms}, x_3 = \text{age of house}$

The coefficient value signifies how much the mean of the dependent variable $y$ changes given a one-unit shift in the independent variable $x$ while **holding other variables in the model constant.** This property of holding the other variables constant is crucial because it allows you to assess the effect of each variable in isolation from the others. Thus, we see that $x_2$ actually holds an inverse relationship with the price of the house, and rightfully so, the scale/range of the variable **number of bedrooms** $x_2$ is only $1-5$. This can be confirmed by `house.bdrms.value_counts().`One unit increase of $x_2$ means one more bedroom, which signifies a decrease of $8621$ in the price of the house. The rest of the variables are easily interpreted in the same way.

---

## Feature Scaling

[Reference]([https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia](https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia))

In addition to the remarks in the other answers, I'd like to point out that the scale and location of the explanatory variables does not affect the *validity* of the regression model in any way.

Consider the model $y=β_0+β_1x_1+β_2x_2+…+ϵ$

The [least squares estimators](http://en.wikipedia.org/wiki/Linear_regression#Least-squares_estimation_and_related_techniques) of $β_1,β_2,...$ are not affected by shifting. The reason is that these are the slopes of the fitting surface - how much the surface changes if you change $x_1,x_2,...$ one unit. This does not depend on location. (The estimator of $β_0$, however, does.)

By looking at the equations for the estimators you can see that scaling $x_1$ with a factor $a$ scales $\hat{β_1}$by a factor $\frac{1}{a}$. To see this, note that

$$\hat{β_1}(x_1)=\dfrac{\sum_{i=1}^{n}(x_{1,i}−\bar{x}_1)(y_i−\bar{y})}{\sum_{i=1}^{n}(x_{1,i}−\bar{x}_1)^2}$$

Thus

$$\hat{β}_1(ax_1)=\dfrac{∑_{i=1}^{n}(ax_{1,i}−a\bar{x}_1)(y_i−\bar{y})}{∑_{i=1}^{n}(ax_{1,i}−a\bar{x}_1)^2}=\dfrac{\hat{\beta}_1(x_1)}{a}$$

By looking at the corresponding formula for $\hat{β}_2$ (for instance) it is (hopefully) clear that this scaling doesn't affect the estimators of the other slopes.

Thus, scaling simply corresponds to scaling the corresponding slopes. Because if we scale **square feet (**$x_1$) by a factor of $\frac{1}{10}$, then if the original $\hat{\beta}_1$when **square feet** is 100, then the above proof shows that the new $\hat{\beta}_1$will be multiplied by 10, becoming 1000, therefore, the interpretation of the coefficients did not change.

However, if you are using Gradient Descent (an optimization algorithm) in Regression, then centering, or scaling the variables, may prove to be faster for convergence.

---

## Hypothesis Testing on $\beta$

Recall that we are ultimately always interested in drawing conclusions about the population, not the particular sample we observed. This is an important sentence to understand, the reason we are testing our hypothesis on the population parameter instead of the estimated parameter is because we are interested in knowing our real population parameter, and we are using the estimated parameter to provide some statistical gauge. In the SLR setting, we are often interested in learning about the population intercept $\beta_0$ and the population slope $β_1$. As you know, **confidence intervals and hypothesis tests** are two related, but different, ways of learning about the values of population parameters. Here, we will learn how to calculate confidence intervals and conduct different hypothesis tests for both $\beta_0$ and $\beta_1$.We turn our heads back to the SLR section, because when we ingest and digest concepts, it is important to start from baby steps first and generalize.

As we can see above from both the fitted plot and the OLS coefficients, there does seem to be a linear relationship between the two. Furthermore, the OLS regression line's equation can be easily calculated and given by (note I have not divided the price unit by $1000$ here):

$$\hat{y} = 71000+135x$$

And so we know the **estimated** slope parameter $\hat{β_1}$ is $135$, and apparently there exhibits a "relationship" between $x$ and $y$. Remember, if there is no relationship, then our optimal **estimated** parameter $\hat{\beta}_1$ should be 0, as a coefficient of $0$ means that $x$ and $y$ has no relationship (or at least in the linear form, the same however, cannot be said for non-linear models!). But be careful, although we can be certain that there is a relationship between `house area` and the `sale price`, but it is only **limited** to the $47$ ****samples that we have!

In fact, we want to know if there is a relationship between the ***population*** of all of the `house area` and its corresponding `sale price` in the whole **population (country). It follows that we also want to ascertain that** **the true population slope** $β_1$is **unlikely** to be 0 as well. Note that $0$ is a common benchmark we use in linear regression, but it, in fact can be any number. This is why we have to draw **inferences** from $\hat{β}_1$ to make substantiate conclusion on the true population slope $β_1$.

Let us formulate our question/hypothesis by asking the question: **Do our `house area` and `sale price` exhibit a true linear relationship in our population? Can we make inferences of our true population parameters based on the estimated parameters (OLS estimates)?**

Thus, we can use the infamous scientific method **Hypothesis Testing** by defining our null hypothesis and alternate hypothesis as follows:

- Null Hypothesis $H_0$: $β_1=0$
- Alternative Hypothesis $H_1$: $β_1\neq 0$

Basically, the null hypothesis says that $β_1=0$, indicating that there is no relationship between $X$ and $y$. Indeed, if $β_1=0$, our original model reduces to $y=β_0+ε$, and this shows $X$ does not depend on $y$ at all. To test the **null hypothesis**, we **instead** need to determine whether $\hat{\beta}_1$, our OLS estimate for $β_1$, is ***sufficiently far from 0*** so that we are **confident** that the real parameter $β_1$ is non-zero. Note the distinction here that we emphasized that we are performing a hypothesis testing on the **true population parameter** but we depend on the value of the **estimate of the true population parameter since we have no way to know the underlying true population parameter.**

---

### T-Statistics

In statistics, the **t-statistic** is the ratio of the difference of the **estimated** value of a true population parameter from its **hypothesized** value to its **standard error**. A good intuitive of [explanation of t-statistics can be read here](https://blog.minitab.com/blog/statistics-and-quality-data-analysis/what-is-a-t-test-and-why-is-it-like-telling-a-kid-to-clean-up-that-mess-in-the-kitchen).

Let $\hat{\mathbf{\beta}}$ be an estimator of $\mathbf{\beta}$ in some statistical model. Then a **t-statistic** for this parameter $\mathbf{\beta}$ is any quantity of the form 

$$t_{\hat{\mathbf{\beta}}} = \dfrac{\hat{\mathbf{\beta}} - \mathbf{\beta}_H}{\text{SE}(\hat{\mathbf{\beta}})}$$

where $\mathbf{\beta}_H$ is the value we want to test in the hypothesis. By default, statistical software sets $\mathbf{\beta}_H = 0$.

In the regression setting, we further take note that the **t-statistic** for each individual coefficient $\hat{\beta}_i$ is given by 

$$t_{\hat{\mathbf{\beta}}i} = [t_{\hat{\mathbf{\beta}}}]_{(i+1) \times (i+1)}$$

***If our null hypothesis is really true, that*** $β_1=0$***, then if we calculate our t-value to be 0, then we can understand it as the number of standard deviations that*** $\hat{β}_1$ ***is 0, which means that*** $\hat{\beta}_1$ ***is 0. This might be hard to reconcile at first, but if we see the formula of the t-statistics, and that by definition we set*** $\beta_H=0$, then it is apparent that if $t_{\hat{\beta}}=0$, it forces the formula to become $t_{\hat{\beta}}=0=\dfrac{\hat{\beta}-0}{\text{SE}(\hat{\beta})} \Longrightarrow \hat{\beta}=0$ ; even more concretely with an example, we replace $\beta_{H}$ with our favorite **true population parameter** $\beta_1$ and $\hat{\beta}$ with $\hat{\beta}_1$, then it just means that if $\beta_1$ were really $0$, i.e. no relationship of $y$ and $x_1$, and if we also get $t_{\hat{\beta}_1}$to be 0 as well (To re-explain this part as a bit cumbersome). ***In which case we accept the null hypothesis; on the other hand, if our t-value is none-zero, it means that*** $\hat{\beta}_1≠0$***)***

Consequently, we can conclude that greater the magnitude of $|t|$ ($t$ can be either positive or negative), the greater the evidence to reject the null hypothesis. The closer $t$ is to 0, the more likely there isn’t a significant evidence to reject the null hypothesis.

---

## Time Complexity

Time Complexity is an important topic, you do not want your code to run for 1 billion years, and therefore, an efficient code will be important to businesses. That is also why Time Complexity questions are becoming increasingly popular in Machine Learning and Data Science interviews!

The Linear Algorithm that we used here simply uses matrix multiplication. We will also ignore the codes that are of constant time O(1). For example, `self.coef_=None` in the constructor is O(1) and we do not really wish to consider this in the *grand scheme of things.*

What is the really important ones are in code lines 37–40. Given X to be a m by n matrix/array, where m is the number of samples and n the number of features. In addition, y is a m by 1 vector. Refer to this [Wikipedia Page](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations) for a handy helpsheet on the various time complexity for mathematical operations.

Line 37: `np.dot(X.T,X)` In the dot product, we transpose the m × n matrix to become n × m, this operation takes O(m × n) time because we are effectively performing two for loops. Next up is performing matrix multiplication, note carefully that `np.dot` between two 2-d arrays does not mean [dot product](https://stackoverflow.com/questions/3744094/time-and-space-complexity-of-vector-dot-product-computation#:~:text=Assuming%20that%20multiplication%20and%20addition,computed%2C%20i.e.%20ai%20*%20bi%20.), instead they are matrix multiplication, which takes O(m × n²) time. The output matrix of this step is n× n.

Line 38: Inverting a n × n matrix takes n³ time. The output matrix is n × n.

Line 39: Now we perform matrix multiplication of n × n and n × m, which gives O(m × n²), the output matrix is n × m.

Line 40: Lastly, the time complexity is O(m × n).

Adding them all up gives you O(2mn+2mn²+n³) whereby simple triangle inequality of mn<mn² implies we can remove the less dominant 2mn term. In the end, the run time complexity of this Linear Regression Algorithm using Normal Equation is O(n²(m+n)). However, you noticed that there are two variables in the bigO notation, and you wonder if we can further reduce the bigO notation to a single variable? Well, if the number of variables is small, which means n is kept small and maybe constant, we can reduce the time complexity to O(m), however, if your variables are increasing, then your time complexity will explode if n → ∞.

This ends the first series, and also the first article published by me. Stay tuned for updates and see me code various Machine Learning Algorithms from scratch.

---

## **Preamble for the next series on Linear Regression**

Just a heads up, I may not be doing part II of the series for Linear Regression just yet, as I want to cover a wide variety of algorithms on a surface level, just enough for beginners (or intermediate) learners. However, as a preamble, I will definitely include more and touch on the following topics that are not covered in today’s session.

## **Orthogonalization**

We can speed up the Normal Equation’s time complexity by using a technique called Orthogonalization, whereby we make use of [QR Factorization](http://mlwiki.org/index.php/QR_Factorization) so we do not need to invert the annoying $X^TX$ where it took n³ time!

## **Regularization**

You basically cannot leave Linear Models without knowing L1–2 Regularization! The Ridge, Lasso, and the ElasticNet! Note that Regularization is a broad term that traverses through all Machine Learning Models. Stay tuned on understanding how Regularization can reduce overfitting. In addition, one caveat that I didn’t mention is what if $X^TX$ is not invertible in our Normal Equation? This can happen if some columns of X are linearly dependent (redundancy in our feature variables), or there are too many features whereby somehow… the number of training samples m is lesser than the number of features n. If you use say, Ridge Regression, then the Modified Normal Equation guarantees a solution. We will talk about it in Part II of Linear Regression.

## **Statistical and Interpretation of Linear Regression**

I didn’t mention much about how to interpret Linear Regression. This is important, even if you know how to code up a Linear Regression Algorithm from scratch, if you do not know how to interpret the results in a statistically rigorous way, then that is not meaningful! Learn more on Hypothesis Testing, Standard Errors, and Confidence Levels. I may delve a bit on Maximum Likelihood Estimators as well!

Conclusion on what I learnt in this few days:

1. Returning `self` by method chaining.
2. Using Decorators in Python where I have to call `raise xxx error` multiple times throughout the classes, which is annoying. Reference from [StackOverFlow](https://stackoverflow.com/questions/24024966/try-except-every-method-in-class). And [Real Python](https://realpython.com/primer-on-python-decorators/).

---

## Python Implementation

- One Hundred Page ML Book, CS229, ML Glossary, GeeksforGeeks.
- Take input $X$ and $y$ → Use either closed form solution or Gradient Descent. And remember $y = X\beta$, use this everywhere for vectorization.
- Gradient Descent
    1. Define Cost Function to be MSE = $\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2$
    2. In order to compute the gradient, we can vectorize it as such: $\nabla$MSE = $-\frac{2}{N}(y_{true} - y_{pred}) @ X$; This is because y_true - y_pred gives you a 1xN vector, whereby X gives you a N x (n+1) vector. Multiplying them give us 1x(n+1) vector, which is the gradient vector of MSE, looks like $[\beta_0, \beta_1, ..., \beta_n]$. Note $\sum_{i=1}^{N}$is omitted due to vectorizing.
    3. Note `y_pred` is calculated by `X @ B`
    4. Question: Verify by hand that the above gradient vector is true and derive it by calculus.

## References and Citations

- [Statistics by Jim - Regression](https://statisticsbyjim.com/regression/)

    [Interpreting MLR Coefficients and P-values](https://statisticsbyjim.com/regression/interpret-coefficients-p-values-regression/)

    [Goodness of Fit and R-Squared](https://blog.minitab.com/en/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)

    [T-Test](https://blog.minitab.com/en/statistics-and-quality-data-analysis/what-is-a-t-test-and-why-is-it-like-telling-a-kid-to-clean-up-that-mess-in-the-kitchen)

[Normal Equation (ML Wiki) Wholesome and Mathematically Rigorous](http://mlwiki.org/index.php/Normal_Equation) (This is a must read)

[Ordinary Least Squares Wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares)

[Linear Regression Assumptions by Jeff Macaluso from Microsoft](https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/)

[Stanford's STATS203 class - Consider downloading them before it's gone](http://statweb.stanford.edu/~jtaylo/courses/stats203/)

[Kaggle Linear Regression Assumptions](https://www.kaggle.com/shrutimechlearn/step-by-step-assumptions-linear-regression)

[Linear Regression Additive Effects (PSU STATS462)](https://online.stat.psu.edu/stat462/node/164/)

[Hands on Linear Regression](https://mylearningsinaiml.wordpress.com/concepts/regression/hands-on-linear-regression/)

[Real Python Linear Regression](https://realpython.com/linear-regression-in-python/#multiple-linear-regression)

[Normal Equation Derivation II](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/)

[Feature Scaling does not affect Linear Regressions' validity of Coefficients](https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia)

[Hypothesis Testing on Optimal Coefficients](https://online.stat.psu.edu/stat462/)

[Conditional Mean and Expectation of Linear Regression](https://stats.stackexchange.com/questions/220507/linear-regression-conditional-expectations-and-expected-values/220509#220509)