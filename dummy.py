import os
import pandas as pd
import numpy as np
os.makedirs('data', exist_ok=True)
np.random.seed(42)
n_samples = 200
data = {
    'Time': np.random.uniform(0, 100000, n_samples),
    'Amount': np.random.exponential(100, n_samples),
    'Class': np.random.choice([0, 1], p=[0.9, 0.1], size=n_samples)
}
for i in range(1, 29):
    data[f'V{i}'] = np.random.normal(0, 1, n_samples)
df = pd.DataFrame(data)
# Add some nans and outliers for testing
df.loc[10:15, 'V1'] = np.nan
df.loc[20, 'V2'] = 1000.0  # outlier
df.to_csv('data/creditcard.csv', index=False)
print('Dummy data created')
