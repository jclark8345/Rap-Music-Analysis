# Rap-Music-Analysis
Exploratory analysis of Rap/Hip-Hop Music Genre through the use of NLP and Machine Learning techniques. For a full writeup of project, please see [CSYS_300_Final_Paper_Justin_Clark.pdf](https://github.com/jclark8345/Rap-Music-Analysis/blob/main/CSYS_300_Final_Paper_Justin_Clark.pdf)

## Contents

- [Project Introduction](#project-introduction)
- [File Directory](#file-directory)
- [Data](#data)
- [Results](#results)
- [References](#references)

## Project Introduction

The music industry as a whole and more recently, the rap/hip-hop genre, has consistently displayed a close tie with contemporary pop culture. Trends and relationships between this genre of music and song popularity have influence on many social and economic factors within the United States. Spotify and Genius API was used to extract audio/lyrical features of 2000+ popular songs in the years 1993 to 2019 as we look examine popularity related to rap/hip-hop music through the use of both exploratory analysis and machine learning techniques. Three algorithms were applied towards popularity classification and results indicated a general importance of track release and lyrical repetition in predicting song popularity. General trends of audience attention towards the recency of music was discovered and specific factors of this genre were examined in relationship to features within the data set.

## File Directory
    - featureAnalysis.py: 
    - mainScrpit.py:
    - popularityPrediction.py
    - rap_1993-2019.csv:
    - spotify_api.py: 

## Data

Billboard Top 100 charts were found for the years 1993 to 2019 and equivalent Spotify playlists were found through novel search. [Spotipy](https://github.com/plamere/spotipy/blob/master/LICENSE.md), a lightweight Python library for the [Spotify Web API](https://developer.spotify.com/documentation/web-api/), was used to obtain full access to all of the music data provided by the Spotify platform based on the given Spotify playlist. This included artist and track data such as album title, track features, song duration and other audio features. Next, [Genius API](https://docs.genius.com/) was used to extract song lyrics from Genius.com for each song within the data set using artist name and track title. Basic data preprocessing steps were applied to lyric data to obtain proper accounts of word frequency and lyric sentiment for each track.

## Results

## References

    - Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ArXiv:1502.03167 [Cs]. http://arxiv.org/abs/1502.03167 
    
    - North, A., & Hargreaves, D. (2008). The Social and Applied Psychology of Music. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780198567424.001.0001

    - North, A. C., Hargreaves, D. J., & O’Neill, S. A. (2000). The importance of music to adolescents. British Journal of Educational Psychology, 70(2), 255–272. https://doi.org/10.1348/000709900158083

    - Raviv, A., Bar-Tal, D., Raviv, A., & Ben-Horin, A. (1996). Adolescent idolization of pop singers: Causes, expressions, and reliance. Journal of Youth and Adolescence, 25(5), 631–650. https://doi.org/10.1007/BF01537358

    - Rokach, L., & Maimon, O. (2005). Top-Down Induction of Decision Trees Classifiers—A Survey. IEEE Transactions on Systems, Man and Cybernetics, Part C (Applications and Reviews), 35(4), 476–487. https://doi.org/10.1109/TSMCC.2004.843247

    - Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.

    -Tibbs, D. F. (2012). From Black Power to Hip Hop: Discussing Race, Policing, and the Fourth Amendment Through the "War on" Paradigm. The Journal of Gender, Race, and Justice, 15(1), 47-79. https://search-proquest-com.ezproxy.uvm.edu/scholarly-journals/black-power-hip-hop-discussing-race-policing/docview/1508067282/se-2?accountid=14679

