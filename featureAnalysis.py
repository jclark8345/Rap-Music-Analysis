"""
Justin Clark
CSYS 300
Final Project
featureAnalysis.py

Exploratory Data Analysis
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd
import lyricsgenius
import re
import nltk
import numpy as np
from nltk.sentiment import SentimentAnalyzer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
from nltk.stem import WordNetLemmatizer 
import pickle
import itertools

lemmatizer = WordNetLemmatizer() 

def calculateTotalFreq(df):
    """

    
    """
    totalFrq = np.sum([Counter(df.iloc[i,-1]) for i in range(df.shape[0]) if df.iloc[i,-1] != None ])
    #totalFrq.most_common(10)
    sortedTotalFreq = sorted(totalFrq.items(), key=lambda pair: pair[1], reverse=True)
    return sortedTotalFreq

def calulateStrongestFeatureCorrelations(df):
    c= df.corr().abs()
    s = c.unstack()
    so = s.sort_values(kind = 'quicksort')
    print(so.tolist()[-20])
    
    
with open('word_list.pkl','rb') as f:
    running_word_list = pickle.load(f)
    
with open('word_list_1993-2019.pkl'
,'rb') as f:
    total_word_list = pickle.load(f)
    
    
#########################################################################
list_all_words = list(itertools.chain.from_iterable(total_word_list))
#while '' in list_all_words:
#    list_all_words.remove('')
counter_all_words = Counter(list(itertools.chain.from_iterable(total_word_list)))
sorted_word_freq = sorted(counter_all_words.items(), key=lambda pair: pair[1], reverse=True)
for i in range(100):
    print("Rank: {} Word: {} Frequency: {}".format(i+1,sorted_word_freq[i][0],sorted_word_freq[i][1]))
    
#########################################################################

count_values = sorted(np.array(list(counter_all_words.values())),reverse = True)
cv = count_values[1:]
frequency = dict()
#counter_all_words_counter = Counter(counter_all_words.values())
#total_list = list(itertools.chain.from_iterable(total_word_list))
for value in cv:
    if value in frequency:
        frequency[value] += 1
    else:
        frequency[value] = 1    
Nk = sorted(np.array(list(frequency.values())),reverse = True)#/np.sum(np.array(list(frequency.values())))
k = sorted(np.array(list(frequency.keys())))
plt.scatter(np.log10(k),np.log10(Nk))
plt.show()
#np.cumsum(Nk)
#cumulative_sum = np.cumsum(Nk)

#log10k = np.log10(np.array(k))
#log10cumsum = np.log10(cumulative_sum)

#plt.scatter(log10k,log10cumsum)
#plt.show()
#Nk = np.array(list(frequency.values()))/np.sum(np.array(list(frequency.values())))
#cumsum = 1 - np.cumsum(Nk)
#nk = np.log10(cumsum)

#k = np.log10(list(sorted(frequency.keys())))
slope_g,intercept_g = np.polyfit(k,nk,1)



fig,ax = plt.subplots()
ax.scatter(k,nk)
ax.plot(k,slope_g*k+intercept_g,color = 'red',label = 'Slope = {:.3f}'.format(slope_g))
plt.xlabel(r"Number of Distinct Names ($\log_{10}k$)",fontsize = 16)
plt.ylabel("Frequency ($\log_{10}P_{\geq}(k)$)",fontsize = 16)
plt.title("Boys 1952 CCDF (log-log space)",fontsize = 16)
plt.tight_layout()
plt.legend()
plt.savefig(os.getcwd() + '/CCDF_.png', dpi=900)
plt.show()

    


### ZIPF PLOT ###
frequency = sorted(list(counter_all_words.values()),reverse=True)
rank = list(range(1,len(frequency)+1))        
rank = np.log10(rank)
frequency = np.log10(frequency)
slope,intercept = np.polyfit(rank,frequency,1)
fig,ax = plt.subplots()
ax.plot(rank,slope*rank+intercept,color = 'red',label='Regression Slope = {}'.format(slope))
ax.scatter(rank,frequency,color = 'blue',label = 'Original Word Data')
plt.xlabel(r"$\log_{10}(Word Rank$)",fontsize = 12)
plt.ylabel(r"$\log_{10}$(Frequency)",fontsize = 12)
plt.title('ZIPF distribution')
#plt.title(r"ZIPF $\rho = {:.3f}$ (log-log space)".format(rho),fontsize = 16) #$\alpha$ = {:.2f}".format(slope),fontsize = 16)
plt.tight_layout()
plt.legend()
plt.savefig(os.getcwd() + '/Plots/ZIPF.png', dpi=900)
plt.show()

    

data = pd.read_csv("rap_1993-2019.csv")

data_pop_sorted = data.sort_values(by = ['Avg Sentiment'])


year_grouped_df = data.groupby(['Year']).mean()
plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Avg Sentiment'].tolist())
plt.show()


plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Word Count'].tolist())
plt.show()

plt.plot(year_grouped_df.index.tolist(),year_grouped_df['danceability'].tolist())
plt.show()


plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Prop Unique Words'].tolist())
plt.show()

plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Prop Lines Pos'].tolist())
plt.show()

plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Prop Lines Neg'].tolist())
plt.show()

plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Prop Lines Neu'].tolist())
plt.show()


### FIGURE ####
plt.plot(year_grouped_df.index.tolist(),year_grouped_df['speechiness'].tolist(),color = 'black',linestyle = 'solid',label = 'speechiness')
plt.plot(year_grouped_df.index.tolist(),year_grouped_df['Prop Unique Words'].tolist(),color = 'black',linestyle = 'dotted',label = 'unique words/total words')
plt.legend(loc = 'center left',fontsize = 12)
plt.xlabel('Year',fontsize = 12)
plt.xlim(1990,2020)
plt.title('Features of  Word Frequency by Year',fontsize = 12)
plt.tight_layout()
plt.savefig(os.getcwd() + '/Plots/speechiness_unique.png',dpi = 900)
plt.show()



#data = data[data['popularity'] != 0]

data.describe()

year_grouped_df = data.groupby(['Year']).mean()
plt.plot(year_grouped_df.index.tolist(),year_grouped_df['popularity'].tolist())
plt.show()
plt.scatter(data['Year'].tolist(),data['popularity'].tolist())
plt.plot()

#merged_df = pd.read_csv(merged_df)
#df_above_50 = merged_df[merged_df.popularity > 50]

#calulateStrongestFeatureCorrelations(merged_df)
#calulateStrongestFeatureCorrelations(df_above_50)

sns.distplot(data['popularity'])
plt.title("Loudness for Sounds above 50 Popularity")

artist_grouped_df = data.groupby(['artist']).mean()
artist_grouped_sorted = artist_grouped_df.sort_values(by = ['Avg Sentiment'])
plt.scatter(artist_grouped_sorted['Prop Unique Words'],artist_grouped_sorted['popularity'])



list_artists = artist_grouped_df.index.values.tolist()
plt.scatter(artist_grouped_df['Avg Sentiment'],artist_grouped_df['popularity'])
#for i,text in enumerate(list_artists):
#    plt.annotate(str(text),(artist_grouped_df['Avg Sentiment'][i],artist_grouped_df['popularity'][i]))
    
#sortedTotalFreq = calculateTotalFreq(merged_df)

#artis = merged_df.groupby(['artist']).sum()