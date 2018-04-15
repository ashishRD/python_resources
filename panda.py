import pandas as pd
import numpy as np

# Series

# data = np.array([1, 2, 3, 4, 5, 6])
# s = pd.Series(data , index=[i for i in range(100 ,160 , 10)])
# print(s)
#
# print(s[150])

# DataFrame
# pandas.DataFrame( data, index, columns, dtype, copy)
#
# data = [1, 2, 3, 4, 5]
#
# df = pd.DataFrame(data, index=[i for i in range(0, 10, 2)], columns=['Number'])
# print(df)

# Adding new column
# df['word'] = pd.Series(['one', 'two', 'three', 'four', 'five'])

# print(df)


# Deleting column


# del df['word']
#
# print(df)
#
# print(df.iloc[1])


# function on DataFrame


# Create a Dictionary of series

#
# d = {'Name':pd.Series(['Tom','James','Ricky','Vin','Steve','Smith','Jack']),'Age':pd.Series([25,26,25,23,30,29,23]),'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
#
# df = pd.DataFrame(d)

#
# print(df.T) # Transpose data
#
# print(df.ndim)
#
# print(df.size)
#
# print(df.values)
#
# print(df.shape)
#
# print(df.empty)
#
# print(df.dtypes)
#
# print(df.axes)

# pandas function applicprint(df)

# print(df.apply(np.mean))

# table-wise operation

# def adder(dataFrame, value_to_add):
#     return dataFrame + value_to_add
#
# def multy(dataFrame, value_to_mul):
#     return dataFrame * value_to_mul
#
# df = pd.DataFrame(np.random.rand(5, 3), columns=['col1', 'clo2', 'col3'])
#
# print(df)
# print(df.pipe(adder , 2))
# print(df.pipe(multy , 4))


# Row or Column wise operation

# print(df)
#
# print(df.apply(np.mean))

# element wise operation

# print(df)
# # custom function
# df['col1'] = df['col1'].map(lambda x : x * 2)
# print(df)
#
# df = df.applymap(lambda x : x * 100)
#
# print(df)


# Reindexing

# print(df)
#
# df_reindex = df.reindex(index=[i for i in range(0 ,10 , 2)])
#
# print(df_reindex)
#
# # filling blank values
#
# df_reindex1 = df.reindex(index=[i for i in range(0 ,10 , 2)], method='ffill')
#
# print(df_reindex1)


# Renaming column names

# df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'},index = {0 : 'apple', 1 : 'banana', 2 : 'durian'})


# Iterations

# iteritems()

# for col_or_key, item_or_key in df.iteritems():
#     print(col_or_key , " " , item_or_key)

# iterrows()

# for row_in_dataframe in df.iterrows():
#     print(row_in_dataframe)

# itertuples() yield the row value with index_no

# for row in df.itertuples():
#     print(row)


# options and customization

# get_option(param) #param === parameter
# examples

# print(pd.get_option('display.max_rows'))
# print(pd.get_option('display.max_column'))
#
# # set options
#
# pd.set_option('display.max_rows' , 100)
# pd.set_option('display.max_column', 100)
#
# print(pd.get_option('display.max_rows'))
# print(pd.get_option('display.max_column'))
#
# # Reset options
#
# pd.reset_option('display.max_rows')
#
# print(pd.get_option('display.max_rows'))
# print(pd.get_option('display.max_column'))

#

# describe options

# print(pd.describe_option("display.max_rows"))

# indexing and selecting data in pandas

# loc()  label based selection or column wise selection

df = pd.DataFrame(np.random.rand(8 , 4), index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                  columns=['A', 'C', 'B', 'D' ])
#
# print(df.loc[:, 'A'])
# print(df.loc['a': 'd', ['A', 'D']])
#
# print(df.loc['a'] > 0.25)



# iloc() index base selection method or row base selection

# df = df.reset_index()
# print(df.iloc[:4])
#
# print(df.iloc[1:3, 1:3])

# ix() hybrid of loc() and iloc() use give power use label as well as index to select the content


# Attribute wise Access

# print(df.A)
# print(df.D)


# Statistical Functions


# percent_change
# this function compares every element with its prior element and computes the change percentage.

# s = pd.Series([1, 2, 3, 4, 5])
# print(s.pct_change())

# By default, the pct_change() operates on columns; if you want to apply the same row wise, then use axis=1() argument.


# Covariance
# Covariance is applied on series data. The Series object has a method cov to compute covariance between series objects. NA will be excluded automatically.
'''Covariance is a measure of how changes in one variable are associated with changes in a second variable. 
Specifically, covariance measures the degree to which two variables are linearly associated. However,
 it is also often used informally as a general measure of how monotonically related two variables are.'''
#
# s1 = pd.Series(np.random.rand(10))
# s2 = pd.Series(np.random.rand(10))
# print(s1.cov(s2))
#
# frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
# print(frame['a'].cov(frame['b']))
# print(frame.cov())


# Correlation
# Correlation shows the linear relationship between any two array of values

# df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
# print(df, '\n')
# print(df['a'].corr(df.b))
# print(df.corr())

# If any non-numeric column is present in the DataFrame, it is excluded automatically.

# Data Ranking
# Data Ranking produces ranking for each element in the array of elements
#
# s = pd.Series([1, 3, 55, 2, 3, 2, 3, 4, 5])
# print(s.rank())
#
#
# # window function
# # sum, mean, median, variance, covariance, correlation, etc.
# # Window functions are majorly used in finding the trends within the
# # data graphically by smoothing the curve. If there is lot of variation
# # in the everyday data and a lot of data points are available,
# # then taking the samples and plotting is one method and applying
# # the window computations and plotting the graph on the results is another method.
# # By these methods, we can smooth the curve or the trend.
# # rolling() or moving mean
#
# df = pd.DataFrame(np.random.randn(10, 4),
# index = pd.date_range('1/1/2000', periods=10),
# columns = ['A', 'B', 'C', 'D'])
#
# print(df.rolling(window=2).mean())
#
#
# # expanding()
#
# print(df.expanding(min_periods=3).mean())
#
#
# # ewm()
#
# print(df.ewm(com=0.5).mean())
#
#
# # Aggregations on DataFrame
#
#
# # In statistics, aggregate data are data combined from several measurements.
# #  When data are aggregated,
# # groups of observations are
# # replaced with summary statistics based on those observations.
# #  In a data warehouse, the use of aggregate data dramatically
# # reduces the time to query large sets of data.
#
# r = df.rolling(window=3, min_periods=1)
# print(r.aggregate(np.sum))
#
# # applying multiple function on single column of DataFrame
#
# print(r['A'].aggregate([np.sum, np.mean]))
#
# # you can apply aggregate on multiply column with multiply functions
#
# print(r[['A','B']].aggregate([np.sum,np.mean]))
#
# # Apply Different Functions to Different Columns of a Dataframe
#
# print(r.aggregate({'A' : np.sum,'B' : np.mean}))
#
#
# # dialing with missing data or cleaning / filling data
#
# df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
# 'h'],columns=['one', 'two', 'three'])
#
# df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
#
# print(df)
#
# # checking the missing values
#
# print(df['one'].isnull())
# # or
# print(df['one'].notnull())
#
#
# # Calculations with Missing Data
# # When summing data, NA will be treated as Zero
# # If the data are all NA, then the result will be NA
#
# # filling NaN value with 0
#
# print(df.fillna(0))
#
#
# # fill NA Forward and Backware
# # for Forward filling we use pad
# # for Backward filling we use backfill
#
# print(df.fillna(method='pad'))
# print(df.fillna(method='backfill'))
#
#
# # droping missing values, its not a good idea
#
# print(df.dropna())
#
# # replace missing values, its a good idea
#
# df = df.fillna(0)
# print(df.replace(0, df.mean()))


# Grouping data

# dict = {'numbers' :[1, 2, 4, 5, 2, 1, 3, 5]}
# dict['sq'] = [i * i for i in dict['numbers']]
# df = pd.DataFrame(dict)
# print(df)
# print(df.groupby('numbers').groups)
#
#
# # groupby with multiple column
#
# print(df.groupby(['numbers', 'sq']).groups)
#
# # iterating through group
# gp = df.groupby(['numbers', 'sq'])
#
# for name, group in gp:
#     print(name)
#     print(group)
#
# # Selecting group
#
# print(df.groupby('numbers').get_group(4))


# # Transformations
#
# grouped = df.groupby('numbers')
# score = lambda x: (x - x.mean()) / x.std()*10
# print(grouped.transform(score))

#
# # Merging / Joining
#
#
# dict1 = {'id': [i for i in range(5)], 'rand_no': np.random.rand(5)}
# dict2 = {'id': [i for i in range(5)], 'id2':[i for i in range(5 ,10 ,1)], 'rand_no': np.random.rand(5)}
#
# df_left = pd.DataFrame(dict1)
# df_right = pd.DataFrame(dict2)
#
# print(df_left)
# print(df_right)
#
# print(pd.merge(df_left, df_right, on='id'))

# we can use 'how' argument for to apply joins
# left
# right
# outer
# inner
# which work same as in relation database
# example : pd.merge(left, right, on='subject_id', how='inner')


# # Categorical Data it's some what like sets
#
# s = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
#
# print(s)
#
# print(s.describe())