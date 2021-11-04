 * Introduction *
 As the name suggests, this code is used to analyze reviews of a hotel using various libraries. Some main features are listed as follows
    - Polarity (positive/negative/neutral/overallScore) of reviews using 
            - SentiStrength
            - VaderNLTK
    - Correlation among ratings of users and polarity returned by analyzers
    - Assigning categories of Empath Client to each review
    - Generation of Histogram from Empath Categories 
    - WordCloud Illustrations
    - Topic Modelling using LDA library
    - Testing various hypothesis
    - Finf Automated Readability Index using Textstat library
 
 * Requirements *
https://github.com/RamishaMunir/SentimentAnalysisOnHotelReviews/blob/main/requirements.txt
 
 * Installation *
 Installation of all required packages can be done using the "requirement.txt" file mentioned above.
 
 * Configuration *
env.py file contains the variable that should be set before running the applications. Such variables varry user to user e.g username & password to connect with Database 

*How to Run" 
For the computations and storing data into the data base, the files (analyzers, points_6_9_13, points 7_8_10) should be run separately. 
To visualize the outputs using UI, run UI.py file
