<h1 style="font-size: 36px; font-weight: bold;">Bank Customer Churn</h1>

## Business scenario and problem


This project centers on an anonymous multinational bank striving to retain its customer base and ensure ongoing business success. By harnessing the power of data analytics, the project aims to uncover customer churn patterns and key drivers, enabling the implementation of proactive retention strategies.

The primary aim is to foster customer loyalty and satisfaction by tailoring personalized services and offerings, cementing the bank's reputation as a customer-focused financial institution globally. Through a deeper understanding of customer preferences and behavior, the project seeks to create seamless and enriching banking experiences, solidifying the bank's position in the international market and ensuring long-term business sustainability.

Acquiring a new client is significantly more costly than retaining an existing one. Understanding the factors that drive a client's decision to leave the company is beneficial for banks. Implementing churn prevention enables companies to create loyalty programs and retention campaigns, aiming to retain as many customers as possible.

## Bank Dataset

In this <a href="https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data"><u>dataset</u></a></body> </html>, there are 10,000 rows and 19 columns, and these variables


<table>
  <tr>
    <th>Variable</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>RowNumber</td>
    <td>Corresponds to the record (row) number and has no effect on the output.</td>
  </tr>
  <tr>
    <td>CustomerId</td>
    <td>Contains random values and has no effect on customer leaving the bank.</td>
  </tr>
  <tr>
    <td>Surname</td>
    <td>The surname of a customer has no impact on their decision to leave the bank.</td>
  </tr>
  <tr>
    <td>CreditScore</td>
    <td>Can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.</td>
  </tr>
  <tr>
    <td>Geography</td>
    <td>A customer’s location can affect their decision to leave the bank.</td>
  </tr>
  <tr>
    <td>Gender</td>
    <td>It’s interesting to explore whether gender plays a role in a customer leaving the bank.</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>This is certainly relevant, since older customers are less likely to leave their bank than younger ones.</td>
  </tr>
  <tr>
    <td>Tenure</td>
    <td>Refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.</td>
  </tr>
  <tr>
    <td>Balance</td>
    <td>Also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.</td>
  </tr>
  <tr>
    <td>NumOfProducts</td>
    <td>Refers to the number of products that a customer has purchased through the bank.</td>
  </tr>
  <tr>
    <td>HasCrCard</td>
    <td>Denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.</td>
  </tr>
  <tr>
    <td>IsActiveMember</td>
    <td>Active customers are less likely to leave the bank.</td>
  </tr>
  <tr>
    <td>EstimatedSalary</td>
    <td>As with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.</td>
  </tr>
  <tr>
    <td>Exited</td>
    <td>Whether or not the customer left the bank.</td>
  </tr>
  <tr>
    <td>Complain</td>
    <td>Whether the customer has a complaint or not.</td>
  </tr>
  <tr>
    <td>Satisfaction Score</td>
    <td>Score provided by the customer for their complaint resolution.</td>
  </tr>
  <tr>
    <td>Card Type</td>
    <td>Type of card held by the customer.</td>
  </tr>
  <tr>
    <td>Points Earned</td>
    <td>The points earned by the customer for using a credit card.</td>
  </tr>
</table>


<h1 style="font-size: 24px; font-weight: bold;">Imports</h1>
<ul>
    <li>Import packages
    <li>Load dataset
</ul>


```python
# Import packages
### YOUR CODE HERE ### 

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve


# For saving models
import pickle 
```


```python
# Load dataset into a dataframe
df0 = pd.read_csv('customer_churn_records.csv')

df0.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>Complain</th>
      <th>Satisfaction Score</th>
      <th>Card Type</th>
      <th>Point Earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>DIAMOND</td>
      <td>464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>DIAMOND</td>
      <td>456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>DIAMOND</td>
      <td>377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>GOLD</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>GOLD</td>
      <td>425</td>
    </tr>
  </tbody>
</table>
</div>



<h1 style="font-size: 24px; font-weight: bold;">Data Exploration (Initial EDA and data cleaning)</h1>
<ul>
    <li> Understand variables
    <li> Clean dataset (missing data, redundant data, outliers)
</ul>

### Gather basic information about the data


```python
# Gather basic information about the data
df0.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 18 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   RowNumber           10000 non-null  int64  
     1   CustomerId          10000 non-null  int64  
     2   Surname             10000 non-null  object 
     3   CreditScore         10000 non-null  int64  
     4   Geography           10000 non-null  object 
     5   Gender              10000 non-null  object 
     6   Age                 10000 non-null  int64  
     7   Tenure              10000 non-null  int64  
     8   Balance             10000 non-null  float64
     9   NumOfProducts       10000 non-null  int64  
     10  HasCrCard           10000 non-null  int64  
     11  IsActiveMember      10000 non-null  int64  
     12  EstimatedSalary     10000 non-null  float64
     13  Exited              10000 non-null  int64  
     14  Complain            10000 non-null  int64  
     15  Satisfaction Score  10000 non-null  int64  
     16  Card Type           10000 non-null  object 
     17  Point Earned        10000 non-null  int64  
    dtypes: float64(2), int64(12), object(4)
    memory usage: 1.4+ MB
    

### Gather descriptive statistics about the data


```python
# Gather descriptive statistics about the data
df0.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>Complain</th>
      <th>Satisfaction Score</th>
      <th>Point Earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.00000</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5000.50000</td>
      <td>1.569094e+07</td>
      <td>650.528800</td>
      <td>38.921800</td>
      <td>5.012800</td>
      <td>76485.889288</td>
      <td>1.530200</td>
      <td>0.70550</td>
      <td>0.515100</td>
      <td>100090.239881</td>
      <td>0.203800</td>
      <td>0.204400</td>
      <td>3.013800</td>
      <td>606.515100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2886.89568</td>
      <td>7.193619e+04</td>
      <td>96.653299</td>
      <td>10.487806</td>
      <td>2.892174</td>
      <td>62397.405202</td>
      <td>0.581654</td>
      <td>0.45584</td>
      <td>0.499797</td>
      <td>57510.492818</td>
      <td>0.402842</td>
      <td>0.403283</td>
      <td>1.405919</td>
      <td>225.924839</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.556570e+07</td>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>119.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2500.75000</td>
      <td>1.562853e+07</td>
      <td>584.000000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>51002.110000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>410.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5000.50000</td>
      <td>1.569074e+07</td>
      <td>652.000000</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>97198.540000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>100193.915000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>605.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7500.25000</td>
      <td>1.575323e+07</td>
      <td>718.000000</td>
      <td>44.000000</td>
      <td>7.000000</td>
      <td>127644.240000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>149388.247500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>801.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10000.00000</td>
      <td>1.581569e+07</td>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1000.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Rename columns

Standardize column names so that they are all in `snake_case`, correct any column names that are misspelled, and make column names more concise as needed


```python
# Display all column names
df0.columns
```




    Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
           'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
           'IsActiveMember', 'EstimatedSalary', 'Exited', 'Complain',
           'Satisfaction Score', 'Card Type', 'Point Earned'],
          dtype='object')




```python
# Rename columns as needed
df0 = df0.rename(columns={'RowNumber': 'row_number',
                          'CustomerId': 'customer_id',
                          'Surname': 'surname',
                          'CreditScore': 'credit_score',
                          'Geography': 'geography',
                          'Gender': 'gender',
                          'Age': 'age',
                          'Tenure': 'tenure',
                          'Balance': 'balance',
                          'NumOfProducts': 'num_of_products',
                          'HasCrCard': 'has_cr_card',
                          'IsActiveMember': 'is_active_member',
                          'EstimatedSalary': 'estimated_salary',
                          'Exited': 'exited',
                          'Complain': 'complain',
                          'Satisfaction Score': 'satisfaction_score',
                          'Card Type': 'card_type',
                          'Point Earned': 'point_earned'})

# Dsiplay all column after update
df0.columns
```




    Index(['row_number', 'customer_id', 'surname', 'credit_score', 'geography',
           'gender', 'age', 'tenure', 'balance', 'num_of_products', 'has_cr_card',
           'is_active_member', 'estimated_salary', 'exited', 'complain',
           'satisfaction_score', 'card_type', 'point_earned'],
          dtype='object')



### Check missing values


```python
df0.isna().sum()
```




    row_number            0
    customer_id           0
    surname               0
    credit_score          0
    geography             0
    gender                0
    age                   0
    tenure                0
    balance               0
    num_of_products       0
    has_cr_card           0
    is_active_member      0
    estimated_salary      0
    exited                0
    complain              0
    satisfaction_score    0
    card_type             0
    point_earned          0
    dtype: int64



There are no missing values in the data.

### Check for duplicates

Check for any duplicate entries in the data


```python
# Check for duplicates
df0.duplicated().sum()
```




    0



There are no duplicates in the data

### Check outliers

Check for outliers in the data


```python
# setting color palette
sns.set_palette(sns.color_palette(["lightskyblue", "salmon", "lightgreen"]))

# Boxplot used to visualize distribution of 'tenure' and detect any outliers
plt.figure(figsize=(4,2))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['tenure'])
plt.show()
```


    
![png](output_25_0.png)
    



```python
# Boxplot used to visualize distribution of 'estimated_salary' and detect any outliers
plt.figure(figsize=(4,2))
plt.title('Boxplot to detect outliers for estimated_salary', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['estimated_salary'])
plt.show()
```


    
![png](output_26_0.png)
    



```python
# Boxplot used to visualize distribution of 'satisfaction_score' and detect any outliers
plt.figure(figsize=(4,2))
plt.title('Boxplot to detect outliers for satisfaction_score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['satisfaction_score'])
plt.show()
```


    
![png](output_27_0.png)
    



```python
# Boxplot used to visualize distribution of 'balance' and detect any outliers
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for balance', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['balance'])
plt.show()
```


    
![png](output_28_0.png)
    



```python
# Boxplot used to visualize distribution of 'num_of_products' and detect any outliers
plt.figure(figsize=(4,2))
plt.title('Boxplot to detect outliers for num_of_products', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['num_of_products'])
plt.show()
```


    
![png](output_29_0.png)
    


The boxplots show that there are no outliers for `tenure`, `estimated_salary`, and `satisfaction_score`. The boxplox for the `balance` column shows that there is a right skew, suggesting there is an asymmetrical distribution around the median. The boxplot for `num_of_products` show that there is an outlier.

Let's investigate if the rows may contain any outliers in the `num_of_products` column.


```python
# Determine number of rows with outliers

# 25th percentile in `num_of_products`
percentile25 = df0['num_of_products'].quantile(0.25)

# 75th percentile in `balance`
percentile75 = df0['num_of_products'].quantile(0.75)

# Interquartile range in `balance`
iqr = percentile75 - percentile25

# Upper and lower limit for non-outliers
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# subset of data containing outliers
outliers = df0[(df0['num_of_products'] > upper_limit) | (df0['num_of_products'] < lower_limit)]

# Count rows that contain outliers in `balance`
print("Number of rows in the data containing outliers in `num_of_products`:", len(outliers))
```

    Lower limit: -0.5
    Upper limit: 3.5
    Number of rows in the data containing outliers in `num_of_products`: 60
    

Based on the results of the boxplots, it seems like there are 60 outliers in the `num_of_products` column. 

## Data Exploration

How many customers exited and what percentage of all customers does it represent


```python
# Number of people who exited vs. stayed
print(df0['exited'].value_counts())
print()

# Percentage of people who exited vs. stayed
print(df0['exited'].value_counts(normalize=True))
```

    0    7962
    1    2038
    Name: exited, dtype: int64
    
    0    0.7962
    1    0.2038
    Name: exited, dtype: float64
    

A stacked boxplot to show `tenure` and `num_of_products` will be used to compare the distributions of customers who exited versus those who stayed. 

A box plot will be used to show the distribution of `num_of_products` for those who stayed and those who exited.


```python
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `tenure` distributions for 'num_of_products'
sns.boxplot(data=df0, x='tenure', y='num_of_products', hue='exited', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Tenure years by number of products purchased', fontsize='14')

# Create histogram showing distribution of `num_of_products`
stay = df0[df0['exited']==0]['num_of_products']
exited = df0[df0['exited']==1]['num_of_products']
sns.histplot(data=df0, x='num_of_products', hue='exited', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of products purchased histogram', fontsize='14')

# Display plots
plt.show()
```


    
![png](output_36_0.png)
    


The stacked boxplot displays a boxplot for customers who exited when the number of products purchased hits 4. With four products purchase, we can speculate that customers are more likely to churn or leave the bank. This could suggest that the complexity of additional services associated with having more products might contribute to higher likelihood of customers leaving the bank.

It seems as the number of products incraeses, the number of customers who exit the bank increases after 2 products purchased. 
Those who purchased a single product is more likely to exit than those purchase two products. The proportion for those who stayed and exited is significantly greater when two products are purchased versus one. 


```python
# Value count of those who stayed/exited for customers who purchased 4 products
df0[df0['num_of_products']==4]['exited'].value_counts()
```




    1    60
    Name: exited, dtype: int64



This confirms that all customers who purchased 4 products exited.

Determining whether `credit_score` is affected by the customers `age`


```python
# Scatterplot of `credit_score` versus `age`

plt.figure(figsize=(16,9))
sns.scatterplot(data=df0, x='age', y='credit_score', hue='exited', alpha=0.04)
plt.legend(labels=['exited', 'stayed'])
plt.title('Credit score by age', fontsize='14')
```




    Text(0.5, 1.0, 'Credit score by age')




    
![png](output_41_1.png)
    


The scatterplot above shows those with a credit score of around 550 to 750 were mostly customers in their late 20's to those in their early 40's. It seems like most of the bank's customers with a credit score is in that age range. 

Visualizing satisfaction score by tenure


```python
# Set figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram distribution of 'tenure'
tenure_stay = df0[df0['exited'] == 0]['tenure']
tenure_exited = df0[df0['exited'] == 1]['tenure']
sns.histplot(data=df0, x='tenure', hue='exited', multiple='dodge', shrink=1, ax=ax)
ax.set_title('Tenure histogram', fontsize='14')
ax.set_xticks(df0['tenure'].unique())  # Setting the x-axis ticks

plt.show()
```


    
![png](output_44_0.png)
    


Based on the histogram, the distribution of customers who stayed and those who left appears similar, suggesting that tenure might not be a significant factor in determining customer churn.


```python
# How many customers voted for each score in 'satisfaction_score'
score_counts = df0['satisfaction_score'].value_counts().sort_index()
print(score_counts)
```

    1    1932
    2    2014
    3    2042
    4    2008
    5    2004
    Name: satisfaction_score, dtype: int64
    


```python
df0['geography'].unique()

```




    array(['France', 'Spain', 'Germany'], dtype=object)




```python
satisfaction_score_counts = df0.groupby(['geography', 'satisfaction_score']).size().unstack().fillna(0)
print(satisfaction_score_counts)

```

    satisfaction_score    1    2     3     4    5
    geography                                    
    France              964  999  1033  1020  998
    Germany             490  504   528   475  512
    Spain               478  511   481   513  494
    


```python
# Data
data = {
    'France': [964, 999, 1033, 1020, 998],
    'Germany': [490, 504, 528, 475, 512],
    'Spain': [478, 511, 481, 513, 494]
}

geographies = list(data.keys())

# Plotting barchart to compare 'satisfaction_score' and 'geographies'
bar_width = 0.2
index = np.arange(len(data['France']))

fig, ax = plt.subplots()
for i, geography in enumerate(geographies):
    ax.bar(index + i * bar_width, data[geography], bar_width, label=geography, edgecolor='black')

ax.set_xlabel('Satisfaction Score')
ax.set_ylabel('Count')
ax.set_title('Satisfaction Score by Geography')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(index + 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.show()
```


    
![png](output_49_0.png)
    


Based on the data, it appears that both Germany and France received a slightly higher number of satisfaction scores of 3 compared to other scores. Meanwhile, Spain had a somewhat balanced distribution between scores 2 and 4. Although the differences are marginal, it's notable that approximately half of the bank's customer base is from France.


```python
# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='exited', y='age', data=df0)
plt.title('Customer Churn based on Age', fontsize=14)
plt.xlabel('Exited', fontsize=12)
plt.ylabel('Age', fontsize=12)

# Add custom legend
legend_labels = ['Stayed', 'Exited']
legend_patches = [plt.Line2D([0], [0], marker='s', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, ['lightblue', 'salmon'])]
plt.legend(handles=legend_patches, loc='upper right', fontsize=12)

plt.show()

```


    
![png](output_51_0.png)
    


The boxplot analysis indicates that customers who exited the service were generally older compared to those who stayed. The median age for those who left was higher than the median age for those who stayed. Furthermore, the interquartile range for the group that exited was wider, suggesting a greater spread of ages within that group. Additionally, the presence of outliers, particularly among the group that stayed, suggests the existence of a subset of older customers within the service.


```python
# Drop 'row_number' and 'customer_id'
df_heatmap = df0.drop(['row_number', 'customer_id'], axis=1)

# Plot a correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df_heatmap.corr(numeric_only=True), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("RdBu", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
```


    
![png](output_53_0.png)
    


Check for strong correlation between variables in the data

The correlation heatmap reveals several notable associations within the dataset.

<ul>
<li>Positive correlation between <i>age</i>, <i>complain</i>, <i>balance</i>, and <i>is_active_member</i>.</li>
<li><i>Num_of_products</i> and <i>has_cr_card</i> exhibit a negative correlation with <i>age</i>.</li>
<li><i>Num_of_products</i> also demonstrate a negative correlation with <i>exited</i> and <i>complain</i>.</li>
<li><i>Exited</i> and <i>complain</i> display a negative correlation with <i>credit_score</i>.</li>
<li><i>Is_active_member</i> showcases a stronger negative correlation with <i>exited</i> and <i>complain</i>.</li>
<li><i>Balance</i> and <i>num_of_products</i> highlight the most pronounced negative correlation in the heatmap.</li>
</ul>

### Insights

It seems that customer churn may be influenced by several factors. Notably, customers with higher balances and those with multiple products are more likely to churn. Additionally, dissatisfaction may play a role, as customers with lower satisfaction scores also tend to churn more frequently. However, interestingly, there doesn't appear to be a strong correlation between churn and other factors such as credit score, geography, or gender. This suggests that customer churn may be more related to specific behaviors and experiences rather than general demographic factors.

It's worth noting that age exhibits the highest positive correlation with customer complaints and churn. This indicates that older customers might be more likely to raise concerns or choose to discontinue their services. Age, therefore, could be an essential factor to consider when understanding and addressing customer satisfaction and retention.

## Model Building
<ul>
    <li> Fit a model that predicts outcome variable using two or more independent variables
    <li> Check for model assumptions
</ul>

Create a heatmap to visualize how variables are correlated.


```python
# Create a heatmap to visualize the correlations between variables
plt.figure(figsize=(8, 6))
sns.heatmap(df0[['age', 'balance', 'num_of_products', 'is_active_member', 'credit_score']]
           .corr(), annot=True, cmap="crest")
plt.title('Correlation Heatmap for Correlated Variables')
plt.show()

```


    
![png](output_60_0.png)
    


### Logistic Regression

Binary logistic regression involves binary classification.

Non-numeric variables must be encoded for modeling. There are four: `surname`, `geography`, `gender`, and `card_type`.

`surname` will have no use in predictive modeling and we found that `geography` does not have any correlation with customer churn.
Therefore, we will encode: `gender` and `card_type`.


```python
df0['card_type'].unique()
```




    array(['DIAMOND', 'GOLD', 'SILVER', 'PLATINUM'], dtype=object)




```python
# Copy the dataframe
df_enc = df0.copy()

# Encode the 'card_type' column as an ordinal numeric category
card_type_map = {'diamond': 3, 'platinum': 2, 'gold': 1, 'silver': 0}
df_enc['card_type'] = df_enc['card_type'].map(card_type_map)

# Encode the 'gender' column as binary
df_enc['gender'] = df_enc['gender'].map({'Female': 0, 'Male': 1})

# Dummy encode the 'department' column
df_enc = pd.get_dummies(df_enc, columns=['card_type'], drop_first=False)

# Display the new dataframe
df_enc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_number</th>
      <th>customer_id</th>
      <th>surname</th>
      <th>credit_score</th>
      <th>geography</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>num_of_products</th>
      <th>has_cr_card</th>
      <th>is_active_member</th>
      <th>estimated_salary</th>
      <th>exited</th>
      <th>complain</th>
      <th>satisfaction_score</th>
      <th>point_earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>0</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>0</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>0</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>425</td>
    </tr>
  </tbody>
</table>
</div>



Create a stacked bar plot to visualize number of customers for each card type and each gender.


```python
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot for card types
pd.crosstab(df0['card_type'], df0['exited']).plot(kind='bar', ax=axs[0])

axs[0].set_title('Counts of Customers by Card Type and Exit Status')
axs[0].set_ylabel('Customer Count')
axs[0].set_xlabel('Card Type')
axs[0].legend(loc='upper right')

# Plot for gender
pd.crosstab(df0['gender'], df0['exited']).plot(kind='bar', ax=axs[1])
axs[1].set_title('Counts of Customers by Gender and Exit Status')
axs[1].set_ylabel('Customer Count')
axs[1].set_xlabel('Gender')

# Adjust layout
plt.tight_layout()
plt.show()

```


    
![png](output_65_0.png)
    


According to the data depicted in the charts, the presence of different card types appears to have no discernible impact on the decision to stay or leave. The proportions remain consistent across all card types.

Regarding gender differences, the data suggests a disparity, with a higher likelihood of females leaving compared to males. Approximately one-third of the females in the data left, while the proportion for males leaving was approximately one-fifth.

Logistic regression is sensitive to outliers so the outliers in `num_of_products` will be removed.


```python
# Select rows without outliers in 'num_of_products' and save the resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['num_of_products'] >= lower_limit) & (df_enc['num_of_products'] <= upper_limit)]

# Display the first few rows of the new dataframe
df_logreg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_number</th>
      <th>customer_id</th>
      <th>surname</th>
      <th>credit_score</th>
      <th>geography</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>num_of_products</th>
      <th>has_cr_card</th>
      <th>is_active_member</th>
      <th>estimated_salary</th>
      <th>exited</th>
      <th>complain</th>
      <th>satisfaction_score</th>
      <th>point_earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>0</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>0</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>0</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>425</td>
    </tr>
  </tbody>
</table>
</div>



Isolating the outcome variable (variable we want to predict).


```python
# Isolate outcome variable
y = df_logreg['exited']

# Display first few rows of outcome variable
y.head()
```




    0    1
    1    0
    2    1
    3    0
    4    0
    Name: exited, dtype: int64



Select features for model to predict outcome variable, `exited`. Drop any redundant variables


```python
# Selecting features for model and dropping any redundant variables 
X = df_logreg.drop(['exited', 'surname', 'geography', 'customer_id', 'row_number'], axis=1)

# Display first few rows of selected features
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit_score</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>num_of_products</th>
      <th>has_cr_card</th>
      <th>is_active_member</th>
      <th>estimated_salary</th>
      <th>complain</th>
      <th>satisfaction_score</th>
      <th>point_earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>2</td>
      <td>464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>1</td>
      <td>3</td>
      <td>456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>0</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>3</td>
      <td>377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>0</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>5</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>0</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>5</td>
      <td>425</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(X.shape)
print(y.shape)

```

    (9940, 12)
    (9940,)
    

Split the data into training set and testing set. Set stratify based on values in `y`.


```python
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
```

Construct logisitc regression model and fit to training dataset


```python
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
```

Test logisitc regression model: use model to make predictions on test set.


```python
y_pred = log_clf.predict(X_test)
```

Create a confusion matrix to visualize the results of the logistic regression model.


```python
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()
```


    
![png](output_81_0.png)
    


The upper-left quadrant displays the number of true negatives. The upper-right quadrant displays the number of false positives. The bottom-left quadrant displays the number of false negatives. The bottom-right quadrant displays the number of true positives.

True negatives: The count of customers predicted as not leaving who actually did not leave.

False positives: The count of customers predicted as leaving but who actually did not leave.

False negatives: The count of customers predicted as not leaving but who actually left.

True positives: The count of customers predicted as leaving who actually did leave.

A perfect model would yield all true negatives and true positives, and no false negatives or false positives.

Check for class balance in the data, check value counts in `exited` column.


```python
df_logreg['exited'].value_counts(normalize=True)
```




    0    0.801006
    1    0.198994
    Name: exited, dtype: float64




```python
# Count the occurrences of each class in the target variable
class_counts = df0['exited'].value_counts()

# Display the counts
print(class_counts)

```

    0    7962
    1    2038
    Name: exited, dtype: int64
    

There is approcimately 80%-20% split. The data is not perfectly balanced but it also not too imbalanced.


```python
# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1))
```

                               precision    recall  f1-score   support
    
    Predicted would not leave       0.80      0.98      0.88      1991
        Predicted would leave       0.36      0.04      0.07       494
    
                     accuracy                           0.80      2485
                    macro avg       0.58      0.51      0.48      2485
                 weighted avg       0.72      0.80      0.72      2485
    
    

The classification report shows that logistic regression achieved a precision of 72%, recall of 80%, f1-score of 72% and accuracy of 80%. 

Based on the classification report and the confusion matrix, here is the analysis of your model's performance:

Precision of 72% implies that among the predicted positives, 72% were actually true positives, indicating that the model is moderately successful in identifying true positives.

Recall of 80% suggests that out of all the actual positive cases, the model identified 80% of them, indicating that it is fairly sensitive in detecting positive instances.

F1-score of 72% implies that the model maintains a reasonable balance between precision and recall, indicating its overall effectiveness in classifying positive cases.

With an accuracy of 80%, the model is correctly predicting the outcome for 80% of the cases, indicating that it performs well overall.

Analyzing the confusion matrix, it seems that the model is effective in identifying true negatives and true positives, but it has some difficulty in minimizing false negatives.

Perform a 5-fold cross-validation to compare the two and determine whether the model's performance remains consistent across different datasets.


```python
# Assuming X and y are your feature and target variables

# Initialize the model
log_reg = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')

# Print the accuracy for each fold
for i, score in enumerate(scores, 1):
    print(f"Accuracy for fold {i}: {score}")

# Print the mean accuracy
print(f"Mean Accuracy: {scores.mean()}")

```

    Accuracy for fold 1: 0.795774647887324
    Accuracy for fold 2: 0.7922535211267606
    Accuracy for fold 3: 0.7962776659959758
    Accuracy for fold 4: 0.795774647887324
    Accuracy for fold 5: 0.789738430583501
    Mean Accuracy: 0.7939637826961772
    

The 5-fold cross-validation results indicate that the logistic regression model achieved consistent accuracies across different subsets of the data. The average accuracy of approximately 79.4% suggests that the model performs reasonably well in predicting the outcome of interest.

## Modeling Approach: Tree-based Model

### Random Forest Tree

Isolate the outcome variable


```python
# Isolate the outcome variable
y = df_enc['exited']

# Display the first few rows of `y`
y.head()
```




    0    1
    1    0
    2    1
    3    0
    4    0
    Name: exited, dtype: int64



Select the features


```python
# Selecting features for model and dropping any redundant variables 
X = df_logreg.drop(['exited', 'surname', 'geography', 'customer_id', 'row_number'], axis=1)

# Display first few rows of `X`
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit_score</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>num_of_products</th>
      <th>has_cr_card</th>
      <th>is_active_member</th>
      <th>estimated_salary</th>
      <th>complain</th>
      <th>satisfaction_score</th>
      <th>point_earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>619</td>
      <td>0</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
      <td>2</td>
      <td>464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>0</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>1</td>
      <td>3</td>
      <td>456</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>0</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
      <td>3</td>
      <td>377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>699</td>
      <td>0</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
      <td>5</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>850</td>
      <td>0</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
      <td>5</td>
      <td>425</td>
    </tr>
  </tbody>
</table>
</div>



Split the data into training, validating, and testing sets.


```python
# Align the sizes of X and y
y = y[:9940]

# Check the shapes of X and y
print(X.shape)
print(y.shape)

# Split the data with the same sample size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
```

    (9940, 12)
    (9940,)
    

Construct a random forest model and set up cross-validated grid-search to search for the best model parameters.


```python
# Split the data into a subset
X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.9, random_state=0) # Adjust the test_size to get the desired subset size

# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3, 5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1, 2, 3],
             'min_samples_split': [2, 3, 4],
             'n_estimators': [300, 500],
             }  

# Assign a list of scoring metrics to capture, including 'zero_division' parameter
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc', error_score=0, verbose=0, n_jobs=-1)

# Fit the model on the subset
rf1.fit(X_subset, y_subset)

```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, error_score=0,
             estimator=RandomForestClassifier(random_state=0), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [3, 5, None], &#x27;max_features&#x27;: [1.0],
                         &#x27;max_samples&#x27;: [0.7, 1.0],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 3],
                         &#x27;min_samples_split&#x27;: [2, 3, 4],
                         &#x27;n_estimators&#x27;: [300, 500]},
             refit=&#x27;roc_auc&#x27;,
             scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;, &#x27;roc_auc&#x27;])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4, error_score=0,
             estimator=RandomForestClassifier(random_state=0), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [3, 5, None], &#x27;max_features&#x27;: [1.0],
                         &#x27;max_samples&#x27;: [0.7, 1.0],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 3],
                         &#x27;min_samples_split&#x27;: [2, 3, 4],
                         &#x27;n_estimators&#x27;: [300, 500]},
             refit=&#x27;roc_auc&#x27;,
             scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;, &#x27;roc_auc&#x27;])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div></div></div></div></div></div></div></div>



Fit the random forest model to the training data


```python
%%time
rf1.fit(X_train, y_train) 
```

Define functions to pickle the model and read in the model.


```python
def write_pickle(model_object, save_as:str):
    '''
    In: 
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the current working directory
    '''    
    with open(save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)
```


```python
def read_pickle(saved_model_name:str):
    '''
    In: 
        saved_model_name: filename of the pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model
```

Save model in a pickle and then read it in Jupyter notebook's directory.


```python
# Write pickle and save to Jupyter directory
write_pickle(rf1, 'hr_rf1')
```


```python
rf1 = read_pickle('hr_rf1')
```

Identify best AUC score from random forest model on training set.


```python
rf1.best_score_
```




    0.5265795946293296



Identify the optimal values for the parameters of the random forest model.


```python
rf1.best_params_
```




    {'max_depth': 3,
     'max_features': 1.0,
     'max_samples': 1.0,
     'min_samples_leaf': 3,
     'min_samples_split': 2,
     'n_estimators': 300}



Collect evaluation scores on training set for random forest models


```python
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table
```


```python
# Get CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(rf1_cv_results)
```

                  model  precision  recall   F1  accuracy      auc
    0  random forest cv        0.0     0.0  0.0  0.795976  0.52658
    

Evaluate the final model of the test set

Define function that gets all scores from model's prediction


```python
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds, zero_division=1)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table
```

Use best performing model to predict on the test set.


```python
# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>random forest1 test</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.795976</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>



## Results and Evaluation
<ul>
    <li> Interpret model
    <li> Evaluate model performance
    <li> Prepare results, visualizations and actionable steps to share with stakeholders
<ul\>

### Summary of model results

**Logistic Regression**

The logistic regression model achieved precision of 72%, recall of 80%, f1-score of 72% and accuracy of 80%. The 5-fold cross validation resulted in an average accuracy of 79.4% making it a reasonable predictor for customer churn.

**Random Forest**

The random forest model, trained on a dataset with 9940 samples, produced a best AUC score of 0.5266. While the model seems to perform relatively well with an accuracy of 79.6%, it is observed that the precision, recall, and F1 score are significantly lower. Additionally, the AUC score is at 0.5, indicating poor discriminatory power.

### Conclusion, Recommendations, Next Steps

**Conclusions:**

* The number of products purchased appears to play a significant role in customer churn, particularly when the count reaches four. This suggests that an increase in product complexity may contribute to higher churn rates.
* Age demonstrates the most substantial positive correlation with customer complaints and churn, highlighting the importance of understanding the impact of age on customer satisfaction and retention.
* While various factors such as balance, multiple products, and satisfaction scores influence customer churn, no strong correlation is observed between churn and other demographic factors like credit score, geography, or gender.
* The random forest model demonstrated relatively good overall performance but showed some limitations in accurately identifying customers at risk of churning.

**Recommmendations:**

* Implement targeted strategies to manage customer churn, focusing on those with higher product counts and older customers, as they seem to be more prone to leaving.
* Devise customer retention initiatives that address individual satisfaction concerns, considering that dissatisfaction appears to significantly impact churn rates.
* Develop customer service protocols to address complaints and concerns, especially among older customers, and tailor support systems to meet their specific needs.
* Initiate interventions and programs specifically designed to address the higher likelihood of female customers leaving, emphasizing customer engagement and support for this demographic group.
* Focus on enhancing data quality and collection methodologies to ensure a more accurate representation of customer behavior and preferences, allowing for more precise predictions.

**Next Steps:**

* Verify the potential impact of data leakage and ensure the robustness of the analysis by evaluating model performance without certain features. This approach would help determine the degree of influence that specific variables have on the predictions and whether they contribute to data leakage.
* Consider utilizing the dataset for a K-means clustering analysis
* Consider refining the model by exploring alternative machine learning approaches or improving the quality and relevance of the data.
