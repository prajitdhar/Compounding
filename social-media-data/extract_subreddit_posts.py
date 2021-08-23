import os
import json
import sys
import csv

subreddits_food = ['bakeoff',
      'baking',
      'cakes',
      'cakewin',
      'cupcakes',
      'deliciousfoods',
      'dessert',
      'desserts',
      'icecreamery',
      'icepops',
      'trollcooking',
      'veganbaking',
      'artisanbread',
      'breadit',
      'breadmachines',
      'sourdough',
      'beerrecipes',
      'coffee',
      'espresso',
      '52weeksofcooking',
      '7dollardinners',
      'aquafaba',
      'archaiccooking',
      'atkgear',
      'bachelorchef',
      'budgetveggie',
      'campfirecooking',
      'castiron',
      'ccrecipe',
      'charcoal',
      'cheap_meals',
      'chefit',
      'chefknives',
      'cli5',
      'collegecooking',
      'cookbooks',
      'cooking',
      'cookingcollaboration',
      'cookingforbeginners',
      'cookingforone',
      'cookingvideos',
      'cookmesomething',
      'cookwithbeer',
      'culinary',
      'culinaryplating',
      'dixiefood',
      'dope_as_fuck_cooking',
      'familyrecipes',
      'fastmealprep',
      'fooddev',
      'foodhacks',
      'fromscratch',
      'fuckingcooking',
      'gifrecipes',
      'goofykitchen',
      'grilling',
      'insidethefridge',
      'instantpot',
      'kitchenconfidential',
      'mealprep',
      'mealprepsunday',
      'menuhacker',
      'mimicrecipes',
      'minimeals',
      'onceamonthcooking',
      'onepotmeals',
      'pastabakes',
      'pickling',
      'pressurecooking',
      'recipeclub',
      'recipegifs',
      'recipes',
      'redditcookbook',
      'ricecookerrecipes',
      'risottomasterrace',
      'shittygifrecipes',
      'shots',
      'singleguymeals',
      'slackerrecipes',
      'slowcooking',
      'smoking',
      'sousvide',
      'steak',
      'studentfood',
      'topsecretrecipes',
      'veggieslowcooking',
      'weeklyfoodcirculars',
      'whatsinmycupboard',
      'whatthefridge',
      'asianeats',
      'bento',
      'chinesefood',
      'indianfood',
      'japanesefood',
      'greentea',
      'tea',
      'teaexchange',
      'teaporn',
      'eatcheapandvegan',
      'glutenfreevegan',
      'meatlessmealprep',
      'plantbaseddiet',
      'raw',
      'rawfoodreddit',
      'shittyveganfoodporn',
      'soylent',
      'spotthevegan',
      'veg',
      'vegan',
      'vegan_food',
      'vegan1200isplenty',
      'veganbeauty',
      'vegancirclejerk',
      'veganfitness',
      'veganfoodporn',
      'vegangifrecipes',
      'veganparenting',
      'veganrecipes',
      'vegents',
      'vegetarian',
      'vegetarian_food',
      'vegetarianfoodporn',
      'vegetarianism',
      'vegetarianrecipes',
      'vegetariansaremetal',
      'veghumor',
      'vegmeme',
      'vegproblems',
      'vegrecipes',
      '80sfastfood',
      'appetizers',
      'askculinary',
      'askfoodhistorians',
      'atlantaeats',
      'bachelor_chow',
      'bacon',
      'bagels',
      'barista',
      'bbq',
      'beerandpizza',
      'breakfast',
      'breakfastfood',
      'budgetfood',
      'burgerking',
      'burgers',
      'butchery',
      'cafe',
      'candymakers',
      'canning',
      'celiac',
      'cereal',
      'charcuterie',
      'cheaphealthyvegan',
      'cheese',
      'cheeseburgers',
      'cheesemaking',
      'chicagofood',
      'chicagosupperclub',
      'chiliconcarne',
      'chipotle',
      'ciderexchange',
      'ciderporn',
      'coffee_shop',
      'coffeestations',
      'coffeetrade',
      'coffeewithaview',
      'columbusfood',
      'condiments',
      'cookiedecorating',
      'cookingwithcondiments',
      'crepes',
      'culinaryporn',
      'curry',
      'dehydrating',
      'deltaco',
      'dessertporn',
      'drupes',
      'eatcheapandhealthy',
      'eatsandwiches',
      'eatwraps',
      'energydrinks',
      'ethiopianfood',
      'fastfood',
      'fastfoodreview',
      'fermentation',
      'fffffffuuuuuuuuuuuud',
      'findascoby',
      'firewater',
      'fishtew',
      'food',
      'food_and_wine',
      'foodie',
      'foodnews',
      'foodonfilm',
      'foodporn',
      'foodporn_ja',
      'foodscience',
      'foodtheory',
      'foodvideos',
      'foodwriting',
      'fried',
      'glutenfree',
      'glutenfreecooking',
      'grease',
      'grilledcheese',
      'grocerystores',
      'healthyfood',
      'honey',
      'hotsauce',
      'ifilikefood',
      'irishfood',
      'ironchef',
      'jello',
      'jerky',
      'juicing',
      'ketchup',
      'kfc',
      'kimchi',
      'knightsofpineapple',
      'kombucha',
      'koreanfood',
      'makesushi',
      'malefoodadvice',
      'masonjars',
      'mcdonalds',
      'mclounge',
      'meat',
      'melts',
      'microwaveporn',
      'morganeisenberg',
      'morning',
      'muglife',
      'offal',
      'onofffood',
      'ourrecipes',
      'pasta',
      'peanutbutter',
      'pho',
      'pickle',
      'pizza',
      'pizzahut',
      'popeyes',
      'publix',
      'ramen',
      'randomactsofcoffee',
      'roasting',
      'rootbeer',
      'russianfood',
      'sandwiches',
      'seafood',
      'seriouseats',
      'sexypizza',
      'shittyfoodporn',
      'smoothies',
      'snackexchange',
      'soda',
      'sodaswap',
      'spicy',
      'sriracha',
      'starbucks',
      'streeteats',
      'streetfoodartists',
      'stupidfood',
      'subway',
      'survivalfood',
      'sushi',
      'sushiabomination',
      'sushiroll',
      'taco',
      'tacobell',
      'tacos',
      'tastyfood',
      'themcreddit',
      'thingstomakeyoudrool',
      'timhortons',
      'todayiate',
      'todaysbreakfast',
      'tonightsdinner',
      'tonightsvegdinner',
      'traderjoes',
      'truesexypizza',
      'trythisfood',
      'veganfood',
      'vitamix',
      'wendys',
      'wewantplates',
      'whataburger',
      'whatyoueat',
      'withrice',
      'worldofpancakes']
subreddits_health = ['biotech', 'biotechnology', 'pharmacy', 'health', 'medical', 'askdocs', 'adverseeffects', 'science', 'longevity', 'scientific', 'tryingforababy', 'stilltrying', 'infertility', 'ttc30', 'ttc_pcos', 'cancer', 'breastcancer', 'thyroidcancer', 'coloncancer', 'cancerfamilysupport', 'cancersupportgroup', 'braincancer']

def get_submissions(infile):
    # write posts in csv format
    outfile_food = infile.split(".")[0] + "_food.json"
    outfile_health = infile.split(".")[0] + "_health.json"
    with open(infile, 'r', encoding='utf-8') as fin, \
            open(outfile_food, "w", encoding="utf-8") as fout_food, \
            open(outfile_health, "w", encoding="utf-8") as fout_health:

        counter = 0
        no_text_counter = 0
        skipped_texts = 0
        skipped_misses_information = 0
        self_post_counter = 0
        for line in fin:
            data = json.loads(line)
            if data['is_self']:
                self_post_counter += 1
            # skip all non-selfposts
            else:
                continue
            text = data["selftext"].strip()
            # print("Text: ", not(text), text)
            if not(text):
                no_text_counter += 1
                continue
                # print("Post without text\n%s" %str(data))
            elif text == "[deleted]":
                continue
            text = data["title"].strip()+ "\n" + data["selftext"].strip()
            # strip again in case selftext was empty, so don't end with newline
            text = text.strip()
            # check if have all data to write the post
            try:
                subreddit = data["subreddit"].lower()
                post_data = {"id": data["id"], "text": text, "created_utc": data["created_utc"], "subreddit": data["subreddit"]}
            except Exception as e:
                print("%s\nPost misses information\n%s" % (str(e), str(line)))
                skipped_misses_information += 1
                continue

            # check from which category the post is (if at all)
            if subreddit.isin(subreddits_food):
                json.dump(post_data, fout_food)
            elif subreddit.isin(subreddits_health):
                json.dump(post_data, fout_health)
            counter += 1
    print("Found %d empty, and %d non-empty posts in %s; skipped %d posts; skipped %d posts with missing information"
          % (no_text_counter, counter, infile, skipped_texts, skipped_misses_information))

def skip_text(text):
    if text.startswith("This comment has been overwritten by an open source script to protect this user"):
        return True
    if text.startswith("Sorry, but this just got posted. And while reposts are allowed, we do remove multiples, which are a clone of an existing post."):
        return True
    if text.startswith("Post removed for not being flaired."):
        return True
    if text == "[deleted]":
        return True
    return False

def get_comments(infile):
    # write posts in csv format
    outfile_food = infile.split(".")[0] + "_food.csv"
    outfile_health = infile.split(".")[0] + "_health.csv"
    with open(infile, 'r', encoding='utf-8') as fin,\
            open(outfile_food, "w", encoding="utf-8") as fout_food, \
            open(outfile_health, "w", encoding="utf-8") as fout_health:

        counter = 0
        no_text_counter = 0
        skipped_texts = 0
        skipped_misses_information = 0
        for line in fin:
            data = json.loads(line)
            text = data["body"].strip()
            if not text:
                no_text_counter += 1
                continue
                # print("Post without text\n%s" %str(data))
            text = text.replace("\n", "\\n").replace("\r", "\\n")
            if skip_text(text):
                skipped_texts += 1
                continue
            # check if have all data to write the post
            try:
                subreddit = data["subreddit"].lower()
                post_data = {"id": data["id"], "text": text, "created_utc": data["created_utc"], "subreddit": data["subreddit"]}
            except Exception as e:
                print("%s\nPost misses information\n%s" %(str(e), str(line)))
                skipped_misses_information += 1
                continue

            # check from which category the post is (if at all)
            if subreddit.isin(subreddits_food):
                json.dump(post_data, fout_food)
            elif subreddit.isin(subreddits_health):
                json.dump(post_data, fout_health)
            counter += 1
    print("Found %d empty, and %d non-empty posts in %s; skipped %d posts; skipped %d posts with missing information"
          % (no_text_counter, counter, infile, skipped_texts, skipped_misses_information))

if __name__ == '__main__':
    json_file = sys.argv[1]
    type = sys.argv[2]

    # get_only_target_user_posts(json_file)

    if type == "submission":
        get_submissions(json_file)
    elif type == "comment":
        get_comments(json_file)