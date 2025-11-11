# ** Observation **
Lors de mes expérimentations, j’ai remarqué que la normalisation des images entre 0 et 1 aide le modèle à apprendre beaucoup plus facilement, en stabilisant l’entraînement.
L’ajout de Dropout a également permis de réduire le surapprentissage (overfitting), ce qui a amélioré la généralisation du modèle sur les données de test.

En revanche, l’utilisation de convolutions sans padding='same' rend le modèle légèrement moins performant. Les images se rétrécissent couche après couche, ce qui fait perdre de l’information sur les bords et impacte la précision finale.