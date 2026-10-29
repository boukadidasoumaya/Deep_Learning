# TP5 — Image Captioning avec RNN et Attention sur ResNet

## Exercice 1 : Chargement des données et Embeddings

### Question 1.a — Chargement du dataset Flickr30k

On redimensionne les images en 224 × 224 pour pouvoir regrouper des images de tailles différentes dans un même batch (qui nécessite des dimensions identiques) et obtenir des cartes de caractéristiques de taille fixe, adaptées au module d'attention de ResNet50.

## Exercice 2 : Extraction de caractéristiques et Transfer Learning

### Question 2.a — ResNet50 pré-entraîné

**1. Pourquoi geler les poids de ResNet50 ?**

Il y a deux raisons principales:

- **Raison matérielle :** cela réduit la consommation de mémoire GPU et les calculs nécessaires pendant l'entraînement.
- **Raison algorithmique :** cela permet de conserver les caractéristiques visuelles déjà apprises sur ImageNet et d'éviter de les dégrader pendant l'apprentissage du modèle de captioning.

**2. Dimension du tenseur de sortie**

Pour une entrée de forme `(32, 3, 224, 224)`, la sortie est de forme **`(32, 2048, 7, 7)`** : 32 images, 2048 canaux de caractéristiques et une grille spatiale de 7 × 7.

## Exercice 3 : Module d'Attention et LSTM personnalisé

### Question 3.a — Module d'attention spatial

Le `context_vector` représente une combinaison pondérée des caractéristiques visuelles de l'image, permettant au LSTM de se concentrer sur les régions les plus pertinentes à chaque étape de génération, plutôt que de dépendre d'un unique vecteur fixe comme dans un Seq2Seq classique.

### 4.b. Analyse des résultats de l'entraînement

**1. Comparaison des prédictions aux époques 1 et 10**

À l'époque 1, la légende générée est peu cohérente, avec des répétitions et une syntaxe incorrecte (*"man men are playing are on the with..."*). À l'époque 10, le modèle reconnaît mieux certains éléments de l'image, notamment les personnes et la piscine (*"two men are in in in pool pool..."*), mais les répétitions persistent et la phrase reste grammaticalement incorrecte.

On observe donc une légère amélioration du contenu, mais pas de réelle amélioration de la syntaxe. De plus, la perte d'entraînement diminue de **5.6075 à 2.5731**, tandis que la loss de validation passe de **4.8440 à 4.3698**. Cette dernière atteint son minimum à l'époque 4 (4.2825), puis augmente légèrement, ce qui suggère un début de surapprentissage.

**2. Impact du planificateur StepLR**

Le planificateur **StepLR** divise le learning rate par 10 toutes les 5 époques : **0.001 → 0.0001 → 0.00001**. Cela permet de réduire progressivement la taille des mises à jour des poids.

Si on gardait un **lr=0.001** constant, les mises à jour pourraient rester trop importantes en fin d'entraînement et provoquer des oscillations autour du minimum. Avec **Adam**, qui utilise des moyennes mobiles des gradients, réduire le learning rate permet de stabiliser les mises à jour et d'affiner l'apprentissage.

Dans nos résultats, la **Train Loss** diminue jusqu'à **2.4851**, tandis que la **Validation Loss** remonte à **4.3869** à l'époque 15. On observe donc un début de surapprentissage que StepLR ne suffit pas à éviter.