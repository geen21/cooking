import pandas as pd
import nltk
import string
import ast
import re
import unidecode

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from MeanEmbeddingVectorizer import MeanEmbeddingVectorizer

nltk.download('wordnet')

df_recipes = pd.read_csv("input/df_recipes.csv")

def ingredient_parser(ingreds):
  # Words for removal

  words_to_remove = ["fresh","minced","chopped" "oil","a","red","bunch","and","clove","or","leaf","chilli","large","extra","sprig","ground","handful","free","small","pepper","virgin","range","from","dried","sustainable","black","peeled","higher","welfare","seed","for","finely","freshly","sea","quality","white","ripe","few","piece","source","to","organic","flat","smoked","ginger","sliced","green","picked","the","stick","plain","plus","mixed","mint","bay","basil","your","cumin","optional","fennel","serve","mustard","unsalted","baby","paprika","fat","ask","natural","skin","roughly","into","such","cut","good","brown","grated","trimmed","oregano","powder","yellow","dusting","knob","frozen","on","deseeded","low","runny","balsamic","cooked","streaky","nutmeg","sage","rasher","zest","pin","groundnut","breadcrumb","turmeric","halved","grating","stalk","light","tinned","dry","soft","rocket","bone","colour","washed","skinless","leftover","splash","removed","dijon","thick","big","hot","drained","sized","chestnut","watercress","fishmonger","english","dill","caper","raw","worcestershire","flake","cider","cayenne","tbsp","leg","pine","wild","if","fine","herb","almond","shoulder","cube","dressing","with","chunk","spice","thumb","garam","new","little","punnet","peppercorn","shelled","saffron","other" "chopped","salt","olive","taste","can","sauce","water","diced","package","italian","shredded","divided","parsley","vinegar","all","purpose","crushed","juice","more","coriander","bell","needed","thinly","boneless","half","thyme","cubed","cinnamon","cilantro","jar","seasoning","rosemary","extract","sweet","baking","beaten","heavy","seeded","tin","vanilla","uncooked","crumb","style","thin","nut","coarsely","spring","chili","cornstarch","strip","cardamom","rinsed","honey","cherry","root","quartered","head","softened","container","crumbled","frying","lean","cooking","roasted","warm","whipping","thawed","corn","pitted","sun","kosher","bite","toasted","lasagna","split","melted","degree","lengthwise","romano","packed","pod","anchovy","rom","prepared","juiced","fluid","floret","room","active","seasoned","mix","deveined","lightly","anise","thai","size","unsweetened","torn","wedge","sour","basmati","marinara","dark","temperature","garnish","bouillon","loaf","shell","reggiano","canola","parmigiano","round","canned","ghee","crust","long","broken","ketchup","bulk","cleaned","condensed","sherry","provolone","cold","soda","cottage","spray","tamarind","pecorino","shortening","part","bottle","sodium","cocoa","grain","french","roast","stem","link","firm","asafoetida","mild","dash","boiling","oil","chopped","vegetable oil","chopped oil","garlic","skin off","bone out" "from sustrainable sources",]

  if isinstance(ingreds, list):
        ingredients = ingreds
  else:
    ingredients = ast.literal_eval(ingreds)
    # We first get rid of all the punctuation. We make use of str.maketrans. It takes three input
    # arguments 'x', 'y', 'z'. 'x' and 'y' must be equal-length strings and characters in 'x'
    # are replaced by characters in 'y'. 'z' is a string (string.punctuation here) where each character
    #  in the string is mapped to None.
    translator = str.maketrans("", "", string.punctuation)
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
      i.translate(translator)
      # We split up with hyphens as well as spaces
      items = re.split(" |-", i)
      # Get rid of words containing non alphabet letters
      items = [word for word in items if word.isalpha()]
      # Turn everything to lowercase
      items = [word.lower() for word in items]
      # remove accents
      items = [
          unidecode.unidecode(word) for word in items
      ]
      # Lemmatize words so we can compare words to measuring words
      items = [lemmatizer.lemmatize(word) for word in items]
      # Get rid of common easy words
      items = [word for word in items if word not in words_to_remove]
      if items:
        ingred_list.append(" ".join(items))
    return ingred_list
    
# df_recipes["ingredients_parsed"] = df_recipes["ingredients"].apply(
#     lambda x: ingredient_parser(x)
# )
# df_recipes.to_csv(r"dataset/parsed_recipes.csv", index=False)

def get_and_sort_corpus(data):
    """
    Get corpus with the documents sorted in alphabetical order
    """
    corpus_sorted = []
    for doc in data.ingredients_parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def get_recommendations(N, scores):
    """
    Top-N recomendations order by score
    """
    # load in recipe dataset
    # df_recipes = pd.read_csv("dataset/RAW_recipes.csv")
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score"])
    count = 0
    for i in top:
        recommendation.at[count, "recipe"] = title_parser(df_recipes["name"][i])
        recommendation.at[count, "ingredients"] = ingredient_parser_final(
            df_recipes["ingredients"][i]
        )
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title

def ingredient_parser_final(ingredient):
    """
    neaten the ingredients being outputted
    """
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def get_recs(ingredients, N=5, mean=True):
    # load in word2vec model
    model = Word2Vec.load("models/model_cbow.bin")
    if model:
        print("Successfully loaded model")
    # load in data
    # df_parsed = pd.read_csv("dataset/parsed_recipes.csv")
    
    # parse ingredients
    df_recipes["ingredients_parsed"] = df_recipes["ingredients"].apply(
        lambda x: ingredient_parser(x)
    )

    # create corpus
    corpus = get_and_sort_corpus(df_recipes)

    if mean:
        # get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]

        assert len(doc_vec) == len(corpus)
    
    # create embessing for input text
    input1 = ingredients
    # create tokens with elements
    input1 = input1.split(",")
    
    input = []
    
    for i in input1:
      input.append(i)
    
    # print(input)
    df_input = pd.DataFrame(columns=['ingredients'])
    df_input = df_input.append({'ingredients': str(input)}, ignore_index=True)
    
    df_input["ingredients_parsed"] = df_input["ingredients"].apply(
        lambda x: ingredient_parser(x)
        # lambda x: print(type(x))
    )

    print(df_input.head())

    parsed_input = []
    for i in df_input['ingredients_parsed']:
      parsed_input = i
    print(parsed_input)

    if mean:
        input_embedding = mean_vec_tr.transform([parsed_input])[0].reshape(1, -1)
        # print(input_embedding)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations

if __name__ == "__main__":
    input = "jambon, sel"
    rec = get_recs(input)
    print(rec)