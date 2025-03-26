# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity

# # Read Dataset
# dataframe = pd.read_csv(r"C:\Users\91639\OneDrive\Desktop\TCS\Zomato_dataSet.csv")

# # Data Cleaning
# dataframe.fillna("Unknown", inplace=True)
# dataframe['Cost'] = dataframe['Cost'].astype(float)

# # Exploratory Data Analysis Example
# plt.figure(figsize=(13,6))
# sns.countplot(x=dataframe["Cuisine"])
# plt.xlabel("Type of Cuisine")
# plt.title("Cuisine Distribution")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Content-Based Filtering
# # Creating a food profile using Cuisine and Location
# dataframe['food_profile'] = dataframe['Cuisine'] + " " + dataframe['Location']

# # Remove duplicates based on Restaurant Name
# df_unique = dataframe.drop_duplicates(subset=['Restaurant_Name']).reset_index(drop=True)

# vectorizer = TfidfVectorizer(stop_words='english')
# food_matrix = vectorizer.fit_transform(df_unique['food_profile'])
# food_sparse = csr_matrix(food_matrix)

# nn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
# nn_model.fit(food_sparse)

# def recommend_content(restaurant_name, top_n=2):
#     if restaurant_name not in df_unique['Restaurant_Name'].values:
#         return "Restaurant not found!"
#     idx = df_unique[df_unique['Restaurant_Name'] == restaurant_name].index[0]
#     distances, indices = nn_model.kneighbors(food_sparse[idx], n_neighbors=top_n+1)
#     return [df_unique.iloc[i]['Restaurant_Name'] for i in indices.flatten()[1:]]

# print("\n✅ Content-Based Recommendations for 'Burger King':")
# print(recommend_content("Burger King"))

# # Collaborative Filtering based on Rating
# pivot_table = dataframe.pivot_table(index='Restaurant_Name', values='Rating', aggfunc='mean')
# pivot_matrix = pivot_table.values
# similarity_matrix = cosine_similarity(pivot_matrix)
# restaurant_indices = {name: idx for idx, name in enumerate(pivot_table.index)}

# def recommend_user_based(restaurant_name, top_n=2):
#     if restaurant_name not in restaurant_indices:
#         return "Restaurant not found!"
#     idx = restaurant_indices[restaurant_name]
#     sim_scores = list(enumerate(similarity_matrix[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
#     return [pivot_table.index[i[0]] for i in sim_scores]

# print("\n✅ Collaborative-Based Recommendations for 'Burger King':")
# print(recommend_user_based("Burger King"))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Read Dataset
dataframe = pd.read_csv(r"C:\Users\91639\OneDrive\Desktop\TCS\Zomato_dataSet.csv")

# Data Cleaning
dataframe.fillna("Unknown", inplace=True)
dataframe['Cost'] = dataframe['Cost'].astype(float)

# ✅ Adding Dummy Reviews for Sentiment Analysis
dataframe['Reviews'] = [
    "Great food and service!", "Tasty but expensive.", "Average experience.",
    "Loved it, will come again!", "Not good, bad taste.", "Nice ambiance.",
    "Highly recommended!", "Healthy and tasty!", "Perfect BBQ night!", "Delicious kababs!"
]

# ✅ Sentiment Analysis
dataframe['Sentiment'] = dataframe['Reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)

# ✅ Exploratory Data Analysis (Cuisine Distribution)
plt.figure(figsize=(13,6))
sns.countplot(x=dataframe["Cuisine"])
plt.xlabel("Type of Cuisine")
plt.title("Cuisine Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ✅ Content-Based Filtering
dataframe['food_profile'] = dataframe['Cuisine'] + " " + dataframe['Location']
df_unique = dataframe.drop_duplicates(subset=['Restaurant_Name']).reset_index(drop=True)

vectorizer = TfidfVectorizer(stop_words='english')
food_matrix = vectorizer.fit_transform(df_unique['food_profile'])
food_sparse = csr_matrix(food_matrix)

nn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
nn_model.fit(food_sparse)

def recommend_content(restaurant_name, top_n=2):
    if restaurant_name not in df_unique['Restaurant_Name'].values:
        return "Restaurant not found!"
    idx = df_unique[df_unique['Restaurant_Name'] == restaurant_name].index[0]
    distances, indices = nn_model.kneighbors(food_sparse[idx], n_neighbors=top_n+1)
    return [df_unique.iloc[i]['Restaurant_Name'] for i in indices.flatten()[1:]]

print("\n✅ Content-Based Recommendations for 'Burger King':")
print(recommend_content("Burger King"))

# ✅ Collaborative Filtering based on Rating
pivot_table = dataframe.pivot_table(index='Restaurant_Name', values='Rating', aggfunc='mean')
pivot_matrix = pivot_table.values
similarity_matrix = cosine_similarity(pivot_matrix)
restaurant_indices = {name: idx for idx, name in enumerate(pivot_table.index)}

def recommend_user_based(restaurant_name, top_n=2):
    if restaurant_name not in restaurant_indices:
        return "Restaurant not found!"
    idx = restaurant_indices[restaurant_name]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [pivot_table.index[i[0]] for i in sim_scores]

print("\n✅ Collaborative-Based Recommendations for 'Burger King':")
print(recommend_user_based("Burger King"))

# ✅ Location-Based Filtering
def recommend_nearby(location, top_n=5):
    nearby = dataframe[dataframe['Location'].str.lower() == location.lower()]
    return nearby.sort_values(by='Rating', ascending=False).head(top_n)[['Restaurant_Name', 'Location', 'Rating']]

print("\n✅ Nearby Recommendations for 'Delhi':")
print(recommend_nearby('Delhi'))

# ✅ Budget Filtering
def recommend_budget(budget, top_n=5):
    budget_restaurants = dataframe[dataframe['Cost'] <= budget]
    return budget_restaurants.sort_values(by='Rating', ascending=False).head(top_n)[['Restaurant_Name', 'Cost', 'Rating']]

print("\n✅ Budget-Friendly Recommendations (Budget: 400):")
print(recommend_budget(400))

# ✅ Top Restaurants Based on Sentiment Score
top_sentiment = dataframe.sort_values(by='Sentiment', ascending=False)[['Restaurant_Name', 'Sentiment']].head(5)
print("\n✅ Top 5 Restaurants by Positive Sentiment:")
print(top_sentiment)

