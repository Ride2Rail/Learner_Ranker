# TravelOffer_Recommendation
RANKER &amp; LEANER part of the travel offer recommendation.

==================================================================
File Description:

Recommender Core:
       1. TEST_DATA
       2. API_Main.py
       3. Cluster.py
       4. LEANER.py
       5. one_hot_encoding.py
       6. ParametersTunning.py
       7. Step1_user2TSP.py
       8. Step2_TSP2OfferCategorizer.py
       9. Step3_EnrichedOfferRanker.py

       And it's main process and all the APIs are shown in 2.API_Main.py

Sara TEST File:
       1. SARA_modified.csv
       2. SaraTest.ipynb

       It shows a real data test, we design the data of a person named "SARA",and test the results to confirm that whether it satisfies our predictions or not. More detailed steps is shown in 2.SaraTest.ipynb

data Generation Methods:
       1. dataGenerationMethods.ipynb

       Three mehods of generate your own dataset are introduced in this file

Accuracy Evaluation:
       1. AccuracyEval.py

       This file can help to generate the dataset and do some analysis on it.

Parameters Range definition:
       1. Alogorithm_Parameters_and_Ranges.pdf

       This file described the details about the algorithms used in our system, and gave the parameters range of each of them.

Guide for using:
       1. Guide for using

       This doc. can guide the user to use this core quickly.


===================================================================
 API LIST:
                       

                       
>>>"API_USER_TRAIN":
                       
       -used by the administrator to train or update the model for user
                       

                       
>>>"API_USER_PREDICT":
                       
       -used by the user to get the recommendation list
                       

                       
>>>"API_USER_FEEDBACK":
                       
       -used by the administrator to update the historical records according to the user's decision
                       

                       
----------------------------------------
                       
 CHECK LIST:
                       

                       
>>>"CHECK_Req2TSP":
                       
       -check the phase from user to tsp 
                       
        
                       
>>>"CHECK_Req2Categorizer":
                       
       -check the pahse from user to categorizer
                       

                       
>>>"CHECK_USER2UPDATE_PROFILE":
                       
       -check the functionality that the user want to update his/her profile
                       

                       
>>>"CHECK_USER2RESPONSE":
                       
       -check the predict function for an old user
                       

                       
>>>"CHECK_ColdUserResponse":
                       
       -check the predict function for a cold user
                       

                       
>>>"CHECK_USER_FEEDBACK":
                       
       -check the feedback function 
                       

                       
>>>"CHECK_ADMINISTRATOR2CLASSIFIER_TRAIN":
                       
       -check the functionality that the administrator want to train/update the classifier model
                       

                       
>>>"CHECK_ADMINISTRATOR2CLUSTER_TRAIN":
                       
       -check the functionality that the administrator want to train/update the cluster model
                       

                       
 -----------------------------------------
                       
 [Using help() to check the details of the function ]
