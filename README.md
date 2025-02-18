# 📊 RFM Analysis (RFM Analysis and Visualization for SuperStore - Global retail company) 
Author: Dat Le Tien  
Date: 2025-02-15  
Tools Used: Python  

---

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset](#-dataset)
3. [⚒️ Main Process](#-main-process)
4. [📊 Key Insights & Visualizations](#-key-insights--visualizations)  

---

## 📌 Background & Overview  

### Objective:
### 📖 What is this project about? 
RFM is a marketing analysis technique that stands for Recency, Frequency, and Monetary Value.<br>
- Recency: measures how recently a customer has made a purchase.<br>
- Frequency: measures how often a customer has made purchases.<br>
- Monetary Value: measures the total amount of money a customer has spent on purchases.<br>
RFM is used to identify and categorize customers based on their purchasing behavior and how recently and frequently they have made purchases, as well as the monetary value of those purchases.<br>

In this project, I applied RFM (Recency, Frequency, Monetary) analysis to SuperStore, a global retail company, utilizing Python to uncover meaningful customer insights. By conducting data exploration, segmentation modeling, and creating detailed visualizations, I identified critical customer groups. These findings empowered the Marketing and Sales teams to refine their strategies, enhance customer engagement, and implement more effective targeted campaigns.  

### 👤 Who is this project for?  
✔️ Marketing Management  
✔️ Sales director  
✔️ Customer Relationship Management

###  ❓Business Questions:  
✔️ How can we segment customers based on their purchasing behavior (Recency, Frequency, Monetary)?<br>
✔️ Which customer groups contribute the most to revenue, and how can we retain them?<br> 
✔️ How can we re-engage inactive customers to improve retention?<br>
✔️ What marketing strategies can be optimized for different customer segments?<br>
✔️ How can targeted campaigns improve customer engagement and sales performance?

---

## 📂 Dataset   
### 📌 Data Discription  
This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers. 
| Field       | Explaintion                                                                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| InvoiceNo   | Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode   | Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.                                                         |
| Description | Product (item) name. Nominal.                                                                                                                               |
| Quantity    | The quantities of each product (item) per transaction. Numeric.                                                                                             |
| InvoiceDate | Invoice Date and time. Numeric, the day and time when each transaction was generated.                                                                       |
| UnitPrice   | Unit price. Numeric, Product price per unit in sterling.                                                                                                    |
| CustomerID  | Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.                                                                     |
| Country     | Country name. Nominal, the name of the country where each customer resides.
### 📊 RFM Model 
RFM is a widely used method in database and direct marketing, particularly in retail and professional services, to assess customer value.

It evaluates three key factors:

- Recency – The last time a customer made a purchase.
- Frequency – The number of times they make purchases.
- Monetary Value – The total amount they spend.

Customers are ranked numerically in each category, usually on a scale of 1 to 5, with higher scores indicating greater value. The most valuable customers achieve the highest scores across all three metrics.

---

## ⚒️ Main Process
1️⃣ Data Cleaning & Preprocessing<br> 
The first step is not simply about removing unwanted data but also involves correcting syntax and spelling errors, handling missing values, identifying and eliminating duplicate records, and normalizing data. Data validation and cleaning play a crucial role in ensuring data accuracy, improving data quality, and maintaining reliability.
```python
# View the first few rows
print(transactions.head())

# Check data information
print(transactions.info())

# Check descriptive statistics
print(transactions.describe())

# Check total null values
print(transactions.isnull().sum())

# Convert columns from object to string using .apply(str)
transactions['InvoiceNo'] = transactions['InvoiceNo'].apply(str)
transactions['StockCode'] = transactions['StockCode'].apply(str)
transactions['Description'] = transactions['Description'].apply(str)
transactions['Country'] = transactions['Country'].apply(str)
transactions['CustomerID'] = transactions['CustomerID'].apply(str)

# Check column data types again
print(transactions.dtypes)

# Drop inappropriate data values
## Drop rows with negative Quantity
transactions = transactions[transactions['Quantity'] > 0]
## Drop rows with negative UnitPrice
transactions = transactions[transactions['UnitPrice'] > 0]
## Drop cancelled data
transactions['check_cancel'] = transactions['InvoiceNo'].apply(lambda x: True if x[0] == 'C' else False)
transactions = transactions[transactions['check_cancel'] == False]
transactions = transactions.replace('nan', None)
transactions = transactions.replace('Nan', None)
transactions.shape

# Statistics for columns with missing values
missing_dict = {
    'volume': transactions.isnull().sum(),
    'percentage_missing': transactions.isnull().sum() / transactions.shape[0] * 100
}
missing_df = pd.DataFrame(missing_dict)
missing_df.head(10)

# Drop 20% of users with missing values
transactions = transactions[transactions['CustomerID'].notnull()]
transactions.head()

## Total rows with missing values in CustomerID column
missing_customer_id_rows = transactions[transactions['CustomerID'].isnull()]
total_missing = len(missing_customer_id_rows)

## 20% of total missing values
num_to_drop = int(total_missing * 0.20)

## Randomly select 20% of missing values rows
rows_to_drop = missing_customer_id_rows.sample(num_to_drop, random_state=1)

## Drop selected rows
transactions = transactions.drop(rows_to_drop.index)
print(len(transactions))
print(transactions["CustomerID"].isnull().sum())

# Check duplicate values
transactions_duplicated = transactions.duplicated(subset=['InvoiceNo', 'StockCode', 'InvoiceDate', 'CustomerID'])
print(transactions[transactions_duplicated].shape)
print('')
print(transactions.shape)

# Drop duplicate values
transactions_drop_duplicates = transactions.drop_duplicates(subset=['InvoiceNo', 'StockCode', 'InvoiceDate', 'CustomerID'], keep='first')
transactions_drop_duplicates.shape
```
2️⃣ Exploratory Data Analysis (EDA)<br>
After completing data cleaning, the Recency, Frequency, and Monetary (RFM) metrics were computed for each customer. Recency indicates the number of days since a customer's most recent purchase, Frequency reflects the total count of transactions, and Monetary represents the overall amount spent by each customer.
```python
# Create RFM_transactions table
last_day = transactions['InvoiceDate'].max()
transactions['cost'] = transactions['Quantity'] * transactions['UnitPrice']
RFM_transactions = transactions.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (last_day - x.max()).days),
    Frequency=('CustomerID', 'count'),
    Monetery=('cost', 'sum')
).reset_index()
RFM_transactions.dtypes
RFM_transactions.head()

# Create R, F, M columns:
RFM_transactions['R'] = pd.qcut(RFM_transactions['Recency'], 5, labels=range(1, 6)).astype(str)
RFM_transactions['F'] = pd.qcut(RFM_transactions['Frequency'], 5, labels=range(1, 6)).astype(str)
RFM_transactions['M'] = pd.qcut(RFM_transactions['Monetery'], 5, labels=range(1, 6)).astype(str)
RFM_transactions['RFM'] = RFM_transactions.apply(lambda x: x['R'] + x['F'] + x['M'], axis=1)
RFM_transactions.head()

# Read segment data from Excel file
segments = pd.read_excel(file_path, sheet_name='Segmentation')
segments['RFM Score'] = segments['RFM Score'].str.split(',')
segments = segments.explode('RFM Score').reset_index(drop=True)

# Remove space of column 'RFM Score'
segments['RFM Score'] = segments['RFM Score'].apply(lambda x: x.replace(' ', ''))

# Merge proper segmentation
RFM_transactions_final = RFM_transactions.merge(segments, left_on='RFM', right_on='RFM Score', how='left')
RFM_transactions_final.head()
```
Scores were assigned to each RFM component using quintiles, which classified customers into segments based on their relative RFM values. These scores formed the basis for customer segmentation and subsequent analysis

---

## 📊 Key Insights & Visualizations  
**1. User profile Distribution** 
```python
# Distribution of User profile
segment_by_user_count = RFM_transactions_final[['Segment', 'CustomerID']].groupby(['Segment']).count().reset_index().rename(columns={'CustomerID': 'user_volume'})
segment_by_user_count['contribution_percent'] = round(segment_by_user_count['user_volume'] / segment_by_user_count['user_volume'].sum() * 100, 2)
segment_by_user_count['type'] = 'user contribution'

segment_by_spending = RFM_transactions_final[['Segment', 'Monetery']].groupby(['Segment']).sum().reset_index().rename(columns={'Monetery': 'spending'})
segment_by_spending['contribution_percent'] = segment_by_spending['spending'] / segment_by_spending['spending'].sum() * 100
segment_by_spending['type'] = 'spending contribution'

segment_agg = pd.concat([segment_by_user_count, segment_by_spending])

# Visualize the distribution of User profile
plt.figure(figsize=(15, 8))
sns.barplot(data=segment_agg, x='Segment', y='contribution_percent', hue='type')
plt.title('The overall distribution of user profile using RFM Model')
plt.xticks(rotation=45)
plt.show()
```
![Image](https://github.com/user-attachments/assets/a0905645-3452-4f80-aa87-bc164766a3cd)

**Data Analysis:**<br>

Group: At Risk Customer and Cannot Lose Them Customer are the two most important customer groups as they account for a large proportion of both volume and revenue. However, these customers haven't used the product for a long time, showing signs of declining interest and high risk of churning.<br>

Suggestions:<br>
- Special promotion campaigns: Target these customers with exclusive offers to motivate them to return.<br>
- Personalized notifications: Send relevant notifications, highlighting product value, or announce new features to revive interest.<br>
- Feedback surveys: Investigate reasons for declining engagement to develop appropriate solutions.

Group: Loyal, New Customer, Potential Loyalist, and Promising make up a large number of customers. However, their transaction value is low, leading to limited overall revenue contribution. This group has development potential if properly stimulated.<br>

Summary:<br>
- Cross-selling and up-selling strategies: Introduce complementary products/services to increase their transaction value.<br>
- Encourage consumption through loyalty programs: Implement point accumulation policies, discounts when reaching certain spending thresholds.<br>
- Enhanced interaction: Use email, marketing campaign notifications to build long-term relationships.

**2. Frequency Distribution** 
```python
# Distribution of Frequency
binsF = [0, 2, 5, 20, np.inf]
labelsF = ['1-2', '2-5', '5-20', '20+']
RFM_transactions['FrequencyGroup'] = pd.cut(RFM_transactions['Frequency'], bins=binsF, labels=labelsF)
fig, ax = plt.subplots(figsize=(8, 3))
sns.countplot(x='FrequencyGroup', data=RFM_transactions, ax=ax)
ax.set_title('Distribution of Frequency')
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Visualize the distribution of Frequency
for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=2)
plt.show()
```
![Image](https://github.com/user-attachments/assets/f74320fc-c323-4f92-b561-05c91d01037a)

**Data Analysis:**
- Group 1-2 times (138 customers):<br>
This is the group with the lowest number of customers.<br>
These may be customers who made trial purchases or did not return after their first purchase.

- Group 2-5 times (169 customers):<br>
The number of customers increases slightly compared to the 1-2 times group.<br>
This group may consist of potential customers who are not yet loyal or regular.

- Group 5-20 times (969 customers):<br>
This is the second-largest group, accounting for a significant proportion.<br>
Customers in this group have relatively high purchase frequency and can be considered loyal customers or those with increasing frequency trends.

- Group 20+ times (3,096 customers):<br>
This is the largest group, dominating the distribution.<br>
Customers in this group are considered loyal or bring the highest value to the business.

**Summary:** <br>
The reason for dividing the bins this way is:

- Helps identify potential customers and groups with the highest revenue potential.
- Supports marketing strategies and loyalty programs based on purchasing frequency.

The business has a large number of regular customers (20+ group) with very high purchase frequency, which is a positive sign. However, the number of customers in the 1-2 and 2-5 times groups is quite low. The business should consider strategies to convert customers with lower frequency (groups 1-2 and 2-5) into higher frequency groups (5-20 and 20+).<br>

Can implement promotional programs, customer care, or special offers to encourage more frequent shopping.

**3. Distribution throughout the time**
```python
# RFM Distribution throughout the time and visualize
transactions['YearMonth'] = transactions['InvoiceDate'].dt.to_period('M')
rfm_time = transactions.groupby(['YearMonth', 'CustomerID']).agg({
    'InvoiceDate': 'max',
    'InvoiceNo': 'count',
    'cost': 'sum'
}).reset_index()

rfm_time.columns = ['YearMonth', 'CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm_time['Recency'] = (last_day - rfm_time['Recency']).dt.days

# Define the rfm_segment function
def rfm_segment(row):
    if row['Recency'] <= 30 and row['Frequency'] >= 10 and row['Monetary'] >= 1000:
        return 'Champions'
    elif row['Recency'] <= 90 and row['Frequency'] >= 5 and row['Monetary'] >= 500:
        return 'Loyal Customers'
    elif row['Recency'] <= 180 and row['Frequency'] >= 3 and row['Monetary'] >= 300:
        return 'Potential Loyalists'
    else:
        return 'Others'

# Customer segmentation over time
rfm_time['Segment'] = rfm_time.apply(rfm_segment, axis=1)

# Calculate the number of customers in each segment over time
rfm_distribution = rfm_time.groupby(['YearMonth', 'Segment']).size().reset_index(name='Count')

# Convert YearMonth to string for plotting
rfm_distribution['YearMonth'] = rfm_distribution['YearMonth'].astype(str)

# Visualize the distribution of RFM segments over time
plt.figure(figsize=(15, 8))
sns.lineplot(data=rfm_distribution, x='YearMonth', y='Count', hue='Segment')
plt.title('RFM Distribution Throughout the Time')
plt.xticks(rotation=45)
plt.show()
```
![Image](https://github.com/user-attachments/assets/14cd2735-ed1e-4de7-ab43-ca0bae1fefcb)
**Data Analysis:** 
- Others Group (Blue):<br>
Largest group in early 2011, but customer numbers decreased significantly from mid-2011.<br>
Peak occurred around January 2011, then sharply declined approaching 0 by year-end.

- Potential Loyalists Group (Orange):<br>
Started increasing from June 2011, showing successful conversion of "Others" into potentially loyal customers.<br>
However, numbers declined from October 2011 to year-end.

- Loyal Customers Group (Green):<br>
Emerged clearly from mid-2011 with slight growth from July 2011 to October 2011.<br>
Subsequently, loyal customer numbers decreased towards year-end, indicating need for retention measures.

- Champions Group (Red):<br>
Latest to emerge, from October 2011, but showed slight decline in following months.<br>
This is a crucial customer group, but relatively small in size, highlighting need for retention and development focus.

**Summary:** <br>
- Graph shows customer migration from "Others" to higher-value groups like "Potential Loyalists", "Loyal Customers", and "Champions". However, from October 2011 onwards, all customer groups showed declining trends, especially the "Others" group.<br>
- Business needs to focus on improving customer retention strategies, particularly for "Champions" and "Loyal Customers" groups, as these provide the highest value. 

---
