#sleep 1h

# health
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_submissions_2016-19.json submission &> to5gram_health_submissions_2016-19.log

python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2010-12.json comment &> to5gram_health_comments_2010-12.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2013.json comment &> to5gram_health_comments_2013.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2014.json comment &> to5gram_health_comments_2014.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2017-18_0.json comment &> to5gram_health_comments_2017-8_0.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2017-18_1.json comment &> to5gram_health_comments_2017-8_1.log

python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/health_comments_2019.json comment &> to5gram_health_comments_2019.log

# food
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/food_submissions_2016-18.json submission &> to5gram_food_submissions_2016-18.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/food_submissions_2019.json submission &> to5gram_food_submissions_2019.log

python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/food_comments_2010-12.json comment &> to5gram_food_comments_2010-12.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/food_comments_2013.json comment &> to5gram_food_comments_2013.log
python -u /mnt/dhr/CreateChallenge_ICC_0821/Compounding/social-media-data/Reddit_to_5grams.py /mnt/dhr/CreateChallenge_ICC_0821/reddit-data/food_comments_2014.json comment &> to5gram_food_comments_2014.log



