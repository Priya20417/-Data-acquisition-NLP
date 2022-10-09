import pandas as pd
import json
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pip
pip.main(['install','seaborn']) 
pip.main(['install','textstat'])
import seaborn as sns
from textstat import flesch_reading_ease
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
#pip install textstat
#install it in command prompt
def plot_text_complexity_histogram(text):
    text.\
        apply(lambda x : flesch_reading_ease(x)).\
        hist()

def plot_parts_of_speach_barchart(text):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')


    def _get_pos(text):
        pos=nltk.pos_tag(word_tokenize(text))
        pos=list(map(list,zip(*pos)))[1]
        return pos
    
    tags=text.apply(lambda x : _get_pos(x))
    tags=[x for l in tags for x in l]
    counter=Counter(tags)
    x,y=list(map(list,zip(*counter.most_common(7))))
    
    sns.barplot(x=y,y=x)
    
def main():
    ## YOUR CODE HERE
    
    df = pd.read_csv("/mnt/data/dagstuhl-15512-argquality-corpus-annotated.csv", sep = "	",encoding = "ISO-8859-1")
    df.columns = df.columns.str.replace(' ','_')
    df.effectiveness=df.effectiveness.str.extract('(\d+)').astype(float).replace(np.nan,0) #extract the digit string and convert to digit using regx
    df.overall_quality=df.overall_quality.str.extract('(\d+)').astype(float).replace(np.nan,0)
    rd=df[['#id','issue','stance','argumentative','overall_quality','effectiveness','argument']]
    ds = rd.groupby(["#id","issue","stance","argument"]).agg(list).reset_index() #grouped data is returned as series thus apply reset for getting a df back
    mask = ds.argumentative.apply(lambda x: any(item for item in ['y'] if item in x and x.count(item)>=2))
    ds = ds[mask]
    dict1={"#id":"id","issue":"issue","stance":"stance_on_topic","argument":"text","argumentative":"argumentative","overall_quality":"argument_quality_scores","effectiveness":"effectiveness_scores"}
    ds.rename(columns=dict1,inplace=True) #renaming as per the output required
    ds=ds.reindex(columns=["id","issue","stance_on_topic","argumentative","argument_quality_scores","effectiveness_scores","text"])#reindexing according to json required
    ds.to_csv("full.csv")
    ds.to_json("full.json",orient='records')
    train, test = train_test_split(ds,test_size = 0.2, random_state = 42)
    train1, val = train_test_split(train, test_size = 0.125, random_state = 42)
    train1.to_json("train.json",orient='records')
    test.to_json("test.json",orient='records')
    val.to_json("val.json",orient='records')
    sns.set(style="whitegrid")
    sns.barplot(x=df["effectiveness"], y=df["overall_quality"], hue=df["annotator"],data=df)
    plt.show()
    plot_text_complexity_histogram(ds['text'])
    plt.show()
    plot_parts_of_speach_barchart(ds['text'])
    plt.show()
    sns.countplot(x ='overall_quality', data = df) 
    plt.show() 
    print("it works!")
    pass


if __name__ == '__main__':
    main()