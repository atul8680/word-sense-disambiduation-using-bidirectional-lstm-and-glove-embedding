import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

class Preprocessing:      
    def clean_text(self,csv_file):
        data = pd.read_csv(csv_file)
        a = data.columns[0]
        data1 = data.rename(columns={a:'sentence'})
        data2 = data1.iloc[:,:].values
        np_data = np.append(data2,a)
        length = np_data.shape[0]
        documents = list()
        for i in range(length):
            comment=re.sub('[^a-zA-Z]', ' ',np_data[i])
            comment=re.sub(' s ', ' ',comment)
            comment=re.sub('art  aphb ', ' ',comment)
            comment=re.sub('aphb ', ' ',comment)
            comment=re.sub(' p ', ' ',comment)
            comment=re.sub('w ', ' ',comment)
            comment = comment.lower()
            comment = comment.split()
            ps = SnowballStemmer('english')
            comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
            comment = ' '.join(comment)
            documents.append(comment)
 
        return documents
        
        
        
        