import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Dados de exemplo
frases = [
    "Eu amo viajar para o Japão",
    "Viajar é incrível e divertido", 
    "Quero conhecer o Japão e a Coreia",
    "Viajar expande nossa mente"
]

print("📊 Frases originais:")
for i, frase in enumerate(frases, 1):
    print(f"{i}. {frase}")

# Bag of Words
print("\n🎯 Bag of Words:")
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(frases)

bow_df = pd.DataFrame(X_bow.toarray(), 
                     columns=vectorizer_bow.get_feature_names_out())
print(bow_df)

# TF-IDF
print("\n📈 TF-IDF:")
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(frases)

tfidf_df = pd.DataFrame(X_tfidf.toarray(),
                       columns=vectorizer_tfidf.get_feature_names_out())
print(tfidf_df.round(3))

# Word Cloud
texto_completo = " ".join(frases)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Palavras - Frases de Viagem')
plt.show()