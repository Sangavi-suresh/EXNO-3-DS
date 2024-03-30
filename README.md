## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
 import pandas as pd
 df=pd.read_csv("/content/Encoding Data.csv")
 df
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/02049ad1-11e6-433e-a024-e8f2cf4400d9)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/21917ba1-ab30-43cd-b5b4-96406dd78088)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/c80dbd57-f457-47cd-8bea-9ffcadc68306)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/4c5015bd-0938-418c-b791-cd2b83fb5b38)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/2bc414d4-1b60-4fea-a828-8b93c7d06c6b)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/2761a9cb-d999-4985-9a9d-71eb77bcd0ba)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/c8c2226b-373d-4d28-b78d-d3978b4519b2)

pip install --upgrade category_encoders

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/fc595670-e1d8-4fe2-a627-5d049f712f15)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/d2443ab2-58f4-4e3e-b20f-a292948256b0)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/917cbfc6-4212-45b2-92b9-02cd8ac36348)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/900fca92-06be-4e24-ab2d-3317fcef9380)

df.skew()

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/ceb63cfd-8893-4067-a162-09215d6f2f27)

np.log(df["Highly Positive Skew"])

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/60babb34-e12e-4451-9cd2-dc593e812c3f)

np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/a6439aca-d4de-425a-81ce-ca8c1498a568)

np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/3637fcab-a720-4879-b71f-128714224dd8)

np.square(df["Highly Positive Skew"])

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/5316e554-1737-4e79-a47f-9fc0760d65e3)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/1c0b0414-fe63-4bc4-b388-a28476cc4064)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/4224e525-1499-441a-b4a3-ac5bcca23783)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/d3a3d1bf-86fd-4d39-a3e6-26bd4fb0dacf)

```
import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/ae9e5eb8-dec4-4273-9907-4d49ff54fc81)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/dd368302-a46b-4fee-b66d-732d8041044a)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/Sangavi-suresh/EXNO-3-DS/assets/118541861/5f50c887-2b9a-4933-be24-fba5d7ad3932)










# RESULT:

        Hence performing Feature Encoding and Transformation process is Successful.


       
