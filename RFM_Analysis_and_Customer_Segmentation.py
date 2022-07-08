# author : ARDA BAYSALLAR
# DATE : 5 July 2022
# RESOURCE : MIUUL - VBO Data Science Bootcamp 2022

# -----------------------------------------------------------------------------------------------------
# MAIN TARGET : FLO Company, which is an online shoe store,
# wants to segment its customers and determine marketing strategies
# according to these segments. For this purpose, the behaviours of the customers
# will be defined, and groups will be created according to the clustering in these behaviours.
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# DATA SET : The dataset consists of the information obtained from the past shopping
# behaviours of the customers who made their last shopping from the X Company as
# Omni-Channel (both online and offline shopping) in the years 2020-2021
# -----------------------------------------------------------------------------------------------------
# DETAILS OF DATA SET FEATURES :
# master_id: Unique customer number
# order_channel: Which channel of the shopping platform is used (Android, ios, Desktop, Mobile)
# last_order_channel: The channel where the most recent purchase was made
# first_order_date: Date of the customer's first purchase
# last_order_date: Customer's last purchase date
# last_order_date_online: The date of the last purchase made by the customer on the online platform
# last_order_date_offline: The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online: The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline: Number of collections made by the customer offline
# customer_value_total_ever_offline: Total fee paid by the customer for offline purchases
# customer_value_total_ever_online: The total fee paid by the customer for their online shopping
# interested_in_categories_12: List of categories the customer has shopped in the last 12 months
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------------------------------
import datetime

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)  
pd.set_option("display.float_format", lambda x: '%.2f' % x)  
pd.set_option('display.width', 500) 

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISSION 1 : Understanding data and Data Preparation
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# Step 1 : Read data and copy for future requirements
# -----------------------------------------------------------------------------------------------------

df_0 = pd.read_csv("flo_data_20k.csv")
df = df_0.copy()

# Step 2 : Data Observation and Stats
# -----------------------------------------------------------------------------------------------------

df.head(10)  # first 10 row
df.columns  # feature names
df.shape  # (19945 observations , 12 features)
df.describe().T  # descriptive stats for numeric features

# NOTES :  order_num_total_ever_online/offline seems to have outliers because mean shows 3.11/1.91 orders
# with very moderate std and 75% quantile however, the maximum number of orders is 200 / 109
# the same situation is also showing itself for customer values
# -----------------------------------------------------------------------------------------------------

# missing values :
df.isnull().sum()  # there is no missing values present itself in this analysis

# feature type analysis
df.info()  # 4 numeric , 8 categorical
df.dtypes


# number of uniques for categorical and numerical
def basic_cat_num_analyzer(df):
    cols = df.columns
    cat_cols = [col for col in cols if str(df[col].dtype) in ('object', 'category')]
    num_cols = [col for col in cols if col not in cat_cols]
    cats_nuniq = pd.DataFrame({col: df[col].nunique() for col in cat_cols}, index=[0])
    nums_nuniq = pd.DataFrame({col: df[col].nunique() for col in num_cols}, index=[0])

    return cat_cols, num_cols, cats_nuniq, nums_nuniq


cats_unique, nums_unique = basic_cat_num_analyzer(df)[2:4]

print("categorical number of unique vals :", "\n", cats_unique)
# master_id --> unique


print("numeric number of unique vals :", "\n", nums_unique)

# Step 3 : Omnichannel : customers shop from both online and offline platforms.
# Create new variables for each customer's total shopping count and spending.
# -----------------------------------------------------------------------------------------------------


df.head()
df['order_num_total_ever_omni'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
df['customer_value_total_ever_omni'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head()

df.describe().T

# Step 4 : Feature types , date type transformation
# -----------------------------------------------------------------------------------------------------
# date columns detection from column name :
date_cols = [col for col in df.columns if 'date' in col.lower()]
df[date_cols] = df[date_cols].astype('datetime64[ns]')
df.info()
df[date_cols].head(3)

# Step 5 : See the distribution of the number of customers in shopping channels,
# the number of products collected and their total expenditures.
# -----------------------------------------------------------------------------------------------------
df_agg_channel = df.groupby('order_channel').agg({'order_num_total_ever_omni': ['sum'],
                                                  'customer_value_total_ever_omni': ['sum']
                                                  })
df_agg_channel.columns = df_agg_channel.columns.droplevel(1)
df_agg_channel.reset_index(inplace=True)
df_agg_channel['channel_order_percentage'] = 100 * (df_agg_channel.order_num_total_ever_omni) / (
    df_agg_channel.order_num_total_ever_omni.sum())
df_agg_channel['channel_value_percentage'] = 100 * (df_agg_channel.customer_value_total_ever_omni) / (
    df_agg_channel.customer_value_total_ever_omni.sum())
df_agg_channel.sort_values(by=df_agg_channel.columns.tolist(), ascending=False, inplace=True)

axes = df_agg_channel.set_index('order_channel').iloc[:, 2:].plot.bar(rot=0, subplots=True)
# NOTES :  it is showing that Android App is most used channel and the other channels are showing same
# distribution for both volume and count
# -----------------------------------------------------------------------------------------------------

# Step 6 : Top 10 customer with highest revenue value
# -----------------------------------------------------------------------------------------------------
df.sort_values(by='customer_value_total_ever_omni', ascending=False).head(10)

df.sort_values(by='customer_value_total_ever_omni', ascending=False).head(10).describe().T

# Step 7 : Top 10 customer with highest number of orders
# -----------------------------------------------------------------------------------------------------
df.sort_values(by='order_num_total_ever_omni', ascending=False).head(10)
df.sort_values(by='order_num_total_ever_omni', ascending=False).head(10).describe().T


# Step 8 : FUNCTIONAL DATA PREPERATION
# -----------------------------------------------------------------------------------------------------
# STEP 3 : OMNI CHANNEL COLUMNS
def omni_col_creator(df):
    """
    Create omni-channel count and volume columns for dataframe
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    new data frame with old dataframe + columns: order_num_total_ever_omni & customer_value_total_ever_omni
    same dataframe with added new 2 columns one for omni-channel offline and online total counts
    the other one offline and online total purchase volume per customer

    df : pandas dataframe

    """
    df['order_num_total_ever_omni'] = df['order_num_total_ever_offline'] + df['order_num_total_ever_online']
    df['customer_value_total_ever_omni'] = df['customer_value_total_ever_offline'] + df[
        'customer_value_total_ever_online']
    return df


# STEP 4 : DATE DTYPE CONVERSION
def date_column_finder(df):
    """
    Find the columns that has date in their name
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    date_columns_list : list
    """
    date_cols = [col for col in df.columns if 'date' in col.lower()]

    return date_cols


def date_convertor(df):
    """
    find date columns and covert dtype to date type
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------

    new dataframe that transforms old dataframe's date columns to dtype = datetime64[ns]
    and detected date columns as a list

    dataframe : pandas dataframe
    date_columns : list
    """

    # date columns detection from column name :
    date_cols = date_column_finder(df)
    df[date_cols] = df[date_cols].astype('datetime64[ns]')

    return df, date_cols


def chan_agg(df):
    """
    Channel based aggregation for omni channel sales and volume
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    Aggregated dataframe for channel groupby
    it has information of :
        * order_num_total_ever_omni : total count per channel
        * customer_value_total_ever_omni : total volume per channel
        * channel_order_percentage : total count percentage per channel
        * channel_value_percentage : total volume percentage per channel

    df_agg_channel : pandas dataframe

    """
    df_agg_channel = df.groupby('order_channel').agg({'order_num_total_ever_omni': ['sum'],
                                                      'customer_value_total_ever_omni': ['sum']
                                                      })

    df_agg_channel.columns = df_agg_channel.columns.droplevel(1)  # drop multiindex level 1 'sum'
    df_agg_channel.reset_index(inplace=True)  # reset indexes to iterative 0-len(df)

    # percentages
    df_agg_channel['channel_order_percentage'] = 100 * (df_agg_channel.order_num_total_ever_omni) / \
                                                 (df_agg_channel.order_num_total_ever_omni.sum())
    df_agg_channel['channel_value_percentage'] = 100 * (df_agg_channel.customer_value_total_ever_omni) / \
                                                 (df_agg_channel.customer_value_total_ever_omni.sum())
    # sorting descending
    df_agg_channel.sort_values(by=df_agg_channel.columns.tolist(), ascending=False, inplace=True)

    return df_agg_channel


# All together MAIN
def data_prep(df, channel_analysis=False):
    """
    Main function do the data preparation steps all in one function
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    return altered dataframe df_altered and date columns list

    df_altered : pandas dataframe
    date_columns : list

    if channel_analysis is True
    then it will give aggregated channel distribution analysis data-frame by volume and count
    else it will return empty data-frame
    """

    # omni channel columns
    df_omni = omni_col_creator(df)
    # date columns converter
    df_prep, date_columns = date_convertor(df_omni)

    if channel_analysis:
        df_agg_chan = chan_agg(df_prep)
    else:
        df_agg_chan = pd.DataFrame()

    return df_prep, date_columns, df_agg_chan


df = df_0.copy()
df, date_columns, df_agg_chan = data_prep(df)


# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISSION 2 : RFM Metric Calculation
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# STEP 1 AND 2 : RECENCY - FREQUENCY - MONETARY

# RECENCY : The last time customer buy something
# the last time customer buys something is the last_order_date
# analyze_date : date of analysis made after the last date of the latest purchase

def recency_calc(df, give_buffer_days=1):
    """
    Create RECENCY column for RFM SEGMENTATION
    Parameters
    ----------
    df : pandas dataframe
    give_buffer_days : buffer date for analyse, default next day after latest transaction

    Returns
    -------
    df : pandas dataframe with RECENCY additional column with number of days integer
    """
    # ANALYZE DATE
    analyze_date = max(df.last_order_date) + datetime.timedelta(days=give_buffer_days)
    df['RECENCY'] = (analyze_date - df['last_order_date']).dt.days  # GIVES INTEGER DAYS

    return df


# FREQUENCY  : Number of transaction per customer
# MONETARY : Volume of total transactions per customers

def rename_freq_and_monetary(df):
    """
    Set name of the order_num_total_ever_omni as FREQUENCY and
      customer_value_total_ever_omni as MONETARY column for RFM SEGMENTATION
    Parameters
    ----------
    df : pandas data frame

    Returns
    -------
    df : pandas dataframe with FREQUENCY AND MONETARY

    """

    df = df.rename(columns={'order_num_total_ever_omni': 'FREQUENCY',
                            'customer_value_total_ever_omni': 'MONETARY'})

    return df


def rfm_col_generator(df, give_buffer_days=1):
    """
    Main RFM column generator
    Parameters
    ----------
    df : pandas dataframe
    give_buffer_days = 1 default

    Returns
    -------
    dataframe with RFM columns
    """
    df_m = rename_freq_and_monetary(df)
    df_rfm = recency_calc(df_m, give_buffer_days=give_buffer_days)

    df_rfm = df_rfm.round({'MONETARY': 0, 'FREQUENCY': 0})
    # STEP 3 : RFM COLUMN
    df_rfm['RFM'] = (df_rfm['RECENCY'].astype(str)
                     + df_rfm['FREQUENCY'].astype(str)
                     + df_rfm['MONETARY'].astype(str))

    return df_rfm


df = rfm_col_generator(df, 1)

df.head()

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISSION 3 : RF SCORE CALCULATION
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# SINCE FREQUENCY LARGER --> BETTER it will go from 1 to 5 because qcut sorts ascending
# SINCE RECENCY SMALLER --> BETTER it will go from 5 to 1 because qcut sorts  ascending

# STEP 1 & 2  : QCUT

df['RECENCY_SCORE'] = pd.qcut(df['RECENCY'], 5, labels=[5, 4, 3, 2, 1])
df['FREQUENCY_SCORE'] = pd.qcut(df['FREQUENCY'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
df['MONETARY_SCORE'] = pd.qcut(df['MONETARY'], 5, labels=[1, 2, 3, 4, 5])

# STEP 3
df['RF_SCORE'] = (df['RECENCY_SCORE'].astype(str) +
                  + df['FREQUENCY_SCORE'].astype(str))

df.RF_SCORE.head(10)
# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISSION 4 : RF SEGMENTATION
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************
# RFM REGEX MAP :
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}


def rf_segment_mapping(dataframe, seg_map={r'[1-2][1-2]': 'hibernating',
                                           r'[1-2][3-4]': 'at_Risk',
                                           r'[1-2]5': 'cant_loose',
                                           r'3[1-2]': 'about_to_sleep',
                                           r'33': 'need_attention',
                                           r'[3-4][4-5]': 'loyal_customers',
                                           r'41': 'promising',
                                           r'51': 'new_customers',
                                           r'[4-5][2-3]': 'potential_loyalists',
                                           r'5[4-5]': 'champions'}):
    """
    Returns the corresponding mapped segment according to Recency and Frequency Scores from (1-5) for both

    Parameters
    ----------
    seg_map : Dictionary -> default segment mapping is given
    dataframe : pandas dataframe

    Returns
    given dataframe with added SEGMENT column according to Recency and Frequency mapping explained below
    -------
    dataframe : Pandas Dataframe with column SEGMENT added according to rules :

        SEGMENT column with respect to the mapped RF score segments as follows :
        R->[1-2] + F->[1-2] : hibernating
        R->[1-2] + F->[3-4] : at-Risk
        R->[1-2] + F->5     : cant_loose
        R->3     + F->[1-2] : about_to_sleep
        R->3     + F-3      : need_attention
        R->[3-4] + F->[4-5] : loyal_customers
        R->4     + F->1     : promising
        R->5     + F->1     : new_customers
        R->[4-5] + F->[2-3] : potential_loyalists
        R->5     + F->[4-5] : champions

    """
    dataframe['RF_SEGMENT'] = dataframe['RF_SCORE'].replace(seg_map, regex=True)
    return dataframe


df = rf_segment_mapping(df)

# *****************************************************************************************************
# -----------------------------------------------------------------------------------------------------
# MISSION 5 : RF SEGMENT ANALYSIS
# -----------------------------------------------------------------------------------------------------
# *****************************************************************************************************

# STEP 1 :
# SEGMENT ANALYSIS

agg_segment_analysis = pd.pivot_table(df, values=['RECENCY', 'FREQUENCY', 'MONETARY'],
                                      index=['RF_SEGMENT'],
                                      aggfunc=['mean'])
agg_segment_analysis.columns = agg_segment_analysis.columns.droplevel(0)
agg_segment_analysis.sort_values(by=['RECENCY', 'FREQUENCY', 'MONETARY'], ascending=False, inplace=True)

axes = agg_segment_analysis.plot.bar(rot=30, subplots=True, figsize=(15, 7), grid=True, ylabel='AVERAGE')


# STEP 2 :

# CASE A  : FLO is adding a new women's shoe brand to its structure.
# The product prices of the brand it includes are above the general customer preferences.
# For this reason, it is desired to contact the customers in the profile
# that will be interested in the promotion of the brand and product sales.
# Those who shop from their loyal customers (champions, loyal_customers)
# and women category are the customers to be contacted specifically.
# Save the id numbers of these customers to the csv file.
df[['interested_in_categories_12','RF_SEGMENT']].value_counts()

filter1 = df['RF_SEGMENT'].isin(['champions','loyal_customers'])
filter2 = df['interested_in_categories_12'].str.contains('KADIN')
df_case_a = df.loc[filter1 & filter2, :]
# CONTROL
df_case_a[['interested_in_categories_12','RF_SEGMENT']].value_counts()


df_case_a['master_id'].to_csv('Mission5_Step2_CaseA.csv')


# CASE B : Nearly 40% discount is planned for Men's and Children's products.
# It is aimed to specifically target customers who are good customers in the past,
# but who have not shopped for a long time, who are interested in the categories related
# to this discount, who should not be lost, who are asleep and new customers.
# Save the ids of the customers in the appropriate profile to the csv file.
filter1 = df['RF_SEGMENT'].isin(['hibernating', 'new_customers', 'cant_loose'])
filter2 = df['interested_in_categories_12'].str.contains('|'.join(['COCUK', 'ERKEK']))
df_case_b = df.loc[filter1 & filter2, :]
# CONTROL
df_case_b[['interested_in_categories_12','RF_SEGMENT']].value_counts()

df_case_b['master_id'].to_csv('Mission5_Step2_CaseB.csv')