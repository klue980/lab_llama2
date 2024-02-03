"""
title : Heatmap Generator.py
description : generate heatmap using seaborn and matplotlib
author : Kim Seong Ho
email : klue980@gmail.com 
since : 2024.02.03
update  : 2024.02.03
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    '1': [99.60, 98.39, 93.61, 74.48],
    '4': [99.56, 98.25, 93.04, 72.73],
    '16': [99.56, 98.22, 92.95, 72.86],
    '64': [99.41, 97.69, 90.90, 65.61],
    '256': [99.04, 96.28, 85.29, 55.05]
}

df = pd.DataFrame(data, index=[256, 64, 16, 4])

# 히트맵 생성
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt=".2f", cmap='Blues', linewidths=.5)
plt.title('Summarization and Generation Stage Ratio')
plt.xlabel('Input Length')
plt.ylabel('Output Length')
plt.show()