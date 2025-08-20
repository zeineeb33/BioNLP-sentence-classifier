# Sentence Classification of PubMed 200k RCT Dataset

##  Description
Ce projet a pour objectif de développer un modèle de classification de phrases issues d’articles de recherche biomédicale.  
L’enjeu est d’assigner chaque phrase à sa section correspondante (Objectif, Méthodes, Résultats, Conclusions, Contexte).  
Cela permet d’automatiser la structuration d’articles scientifiques et d’améliorer la recherche d’information dans la littérature biomédicale.

##  Dataset
- **Nom :** PubMed 200k RCT
- **Composition :**
  - 180 040 phrases pour l’entraînement
  - 30 212 pour la validation
  - 30 135 pour le test
- **Labels :**
  - `BACKGROUND`
  - `OBJECTIVE`
  - `METHODS`
  - `RESULTS`
  - `CONCLUSIONS`

## ⚙ Méthodologie
1. **Prétraitement :**
   - Tokenisation et nettoyage
   - Représentation vectorielle avec :
     - TF-IDF
     - CountVectorizer
     - BioWordVec (embeddings spécifiques au domaine biomédical)

2. **Modèles testés :**
   - Naïve Bayes (baseline, TF-IDF & CountVectorizer)
   - Logistic Regression avec BioWordVec
   - SVM (abandonné pour performance insuffisante)

3. **Optimisation :**
   - GridSearchCV pour les hyperparamètres (`alpha` pour Naïve Bayes, `C` pour Logistic Regression)

##  Résultats
- **Meilleur modèle :** Logistic Regression + BioWordVec  
- **Observations :**
  - Naïve Bayes limité par son hypothèse d’indépendance des mots.
  - Les embeddings biomédicaux (BioWordVec) capturent la sémantique et améliorent significativement la performance.

##  Conclusion
Les représentations vectorielles spécialisées (BioWordVec) surpassent largement les méthodes classiques de type Bag-of-Words pour la classification de phrases biomédicales.  
Ce projet montre l’importance d’utiliser des embeddings adaptés au domaine dans les tâches de NLP.

##  Outils
- Python (scikit-learn, numpy, pandas)
- BioWordVec
- Matplotlib / Seaborn pour la visualisation

## 📂 Organisation
- `notebooks/` : code principal (Jupyter Notebook)  
- `reports/` : rapport PDF détaillant le projet  

---
