#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from preprocessor import file2frame,normalize

data = file2frame("../datass.csv","../qrsss.csv")

plt.subplot(2,1,1)
for x in data[1:10]:
    plt.plot(x[0])
plt.title("Before preprocessing")
normalize(data)
plt.subplot(2,1,2)
for x in data[1:10]:
    plt.plot(x[0])
plt.title("After preprocessing")
plt.tight_layout()
if not os.path.exists("images"):
    os.makedirs("images")
plt.savefig("images/preprocessing.png")