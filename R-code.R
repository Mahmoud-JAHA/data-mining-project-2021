##  Mahmoud JAHA  /  L3 MASS  ##
## Projet : Prédiction de défauts de paiements ##

#--------------------------------------#
# ACTIVATION DES LIRAIRIES NECESSAIRES #
#--------------------------------------#
library(ggplot2)
library("cowplot")
library(rpart)
library(C50)
library(randomForest)
library(kknn)
library(ROCR)
library(e1071)
library(naivebayes)
library(nnet)

#-------------------------#
# PREPARATION DES DONNEES #
#-------------------------#
#Intégrer la data frame "Data Projet" sur R
projet <- read.csv("C:/Users/Dadou/Documents/Documents importants/Data Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = TRUE)

#Intégrer la data fram "Data Projet New" sur R
projet_P <- read.csv("C:/Users/Dadou/Documents/Documents importants/Data Projet New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = TRUE)

#Retirer la variable inutile "customer"
projet <- projet[,-3]

#Effectuer un échantillonage aléatoire sur "projet"
projet_tab <- projet[sample(1:nrow(projet), 1200), ]

#Créer les ensembles d'apprentissage et de test
projet_EA0 <- projet_tab[1:800,]
projet_ET0 <- projet_tab[801:1200,]

#Retirer la variable "ncust"
projet_EA <- projet_EA0[,-2]
projet_ET <- projet_ET0[,-2]
projet_P <- projet_P[, -2]

#Analyser les ensembles d'app et de test
#Comparaison des effectifs des classes dans les 3 ensembles
windows()
a <- qplot(default, data=projet, main="Effectifs des classes (default) dans Projet", fill=default)
b <- qplot(default, data=projet_EA, main="Effectifs des classes (default) dans Projet_EA", fill=default)
c <- qplot(default, data=projet_ET, main="Effectifs des classes (default) dans Projet_ET", fill=default)

#Comparaison des proportions des classes pour une variable prédictive par histogrammes  (si souhaité, changer la variable utilisée).
d <- qplot(ed, data=projet, main="Proportions des classes pour ed dans Projet", fill=default)
e <- qplot(ed, data=projet_EA, main="Proportions des classes pour ed dans Projet_EA", fill=default)
f <- qplot(ed, data=projet_ET, main="Proportions des classes pour ed dans Projet_ET", fill=default)
plot_grid(a, b, c, d, e, f, ncol = 3, nrow = 2)


#Comparaison des distributions des valeurs d'une variable prédictive pour chaque classe par boxplots (si souhaité, changer la variable utilisée).
windows()
layout(matrix(1 :3, 1, 3))
boxplot(age~default, data=projet, col=c("red","green"), main="Proportions des classes pour age dans Projet", ylab="age", xlab="default")
boxplot(age~default, data=projet_EA, col=c("red","green"), main="Proportions des classes pour age dans Projet_EA", ylab="age", xlab="default")
boxplot(age~default, data=projet_ET, col=c("red","green"), main="Proportions des classes pour age dans Projet_ET", ylab="age", xlab="default")



#Analyse exploratoire des données
# Nuage de points des variables continues ou discrètes (quelconques) avec couleurs selon classe
windows()
qplot(age, income, data=projet, main="Nuage de point de income et age", xlab="Valeur de age", ylab="Valeur de income", color=default)
#Analyse similaire qui compte comme justification de la création d'une nouvelle variable "Quotient debtinc/employ"
windows()
qplot(debtinc, employ , data=projet, main="Nuage de point de employ  et debtinc", xlab="Valeur de debtinc", ylab="Valeur de employ ", color=default)
# Ajout du Jitter vertical et horizontal afin de mieux analyser le graph 
qplot(debtinc, employ , data=projet, main="Nuage de point de employ  et debtinc", xlab="Valeur de debtinc", ylab="Valeur de employ", color=default) + geom_jitter(width = 1, height = 1)

#Création d'une variable "Quotien debtinc/employ"
projet_Q <- projet_tab
projet_Q$Quotient_employ_debtinc <- ifelse(projet_tab$employ==0, projet_tab$debtinc, projet_tab$debtinc/projet_tab$employ)
windows()
boxplot(Quotient_employ_debtinc~default, data=projet_Q, col=c("red","blue"), main="Quotien debtinc/employ selon default dans projet_Q", ylab="Quotien employ/debtinc", xlab="default")
#Création des ensembles d'apprentissage et de test avec Quotien_employ_debtinc
projet_Q_EA <- projet_Q[1:800, ]
projet_Q_ET <- projet_Q[801:1200, ]

#-------------------------#
# ARBRE DE DECISION RPART #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [avec Quotient]
test_Q_rpart <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  dtQ <- rpart(default~., projet_Q_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  
  # Tests du classifieur : classe predite
  dtQ_class <- predict(dtQ, projet_Q_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, dtQ_class))
  
  # Tests du classifieur : probabilites pour chaque prediction
  dtQ_prob <- predict(dtQ, projet_Q_ET, type="prob")
  
  # Courbes ROC
  dtQ_pred <- prediction(dtQ_prob[,2], projet_Q_ET$default)
  dtQ_perf <- performance(dtQ_pred,"tpr","fpr")
  plot(dtQ_perf, main = "Arbres de décision rpart()", add = arg3, col = arg4, lty=2, lwd=2)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  dtQ_auc <- performance(dtQ_pred, "auc")
  cat("AUC = ", as.character(attr(dtQ_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [sans Quotient]
test_rpart <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  dt <- rpart(default~., projet_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  
  # Tests du classifieur : classe predite
  dt_class <- predict(dt, projet_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_ET$default, dt_class))
  
  # Tests du classifieur : probabilites pour chaque prediction
  dt_prob <- predict(dt, projet_ET, type="prob")
  
  # Courbes ROC
  dt_pred <- prediction(dt_prob[,2], projet_ET$default)
  dt_perf <- performance(dt_pred,"tpr","fpr")
  plot(dt_perf, main = "Arbres de décision rpart()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  dt_auc <- performance(dt_pred, "auc")
  cat("AUC = ", as.character(attr(dt_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#----------------#
# RANDOM FORESTS #
#----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Sans Quotient]
test_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rf <- randomForest(default~., projet_EA, ntree = arg1, mtry = arg2)
  
  # Test du classifeur : classe predite
  rf_class <- predict(rf,projet_ET, type="response")
  
  # Matrice de confusion
  print(table(projet_ET$default, rf_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  rf_prob <- predict(rf, projet_ET, type="prob")
  
  # Courbe ROC
  rf_pred <- prediction(rf_prob[,2], projet_ET$default)
  rf_perf <- performance(rf_pred,"tpr","fpr")
  plot(rf_perf, main = "ROC Random Forests [Sans Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  rf_auc <- performance(rf_pred, "auc")
  cat("AUC = ", as.character(attr(rf_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Avec Quotient]
test_Q_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rfQ <- randomForest(default~., projet_Q_EA, ntree = arg1, mtry = arg2)
  
  # Test du classifeur : classe predite
  rfQ_class <- predict(rfQ,projet_Q_ET, type="response")
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, rfQ_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  rfQ_prob <- predict(rfQ, projet_Q_ET, type="prob")
  
  # Courbe ROC
  rfQ_pred <- prediction(rfQ_prob[,2], projet_Q_ET$default)
  rfQ_perf <- performance(rfQ_pred,"tpr","fpr")
  plot(rfQ_perf, main = "ROC Random Forests [Avec Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  rfQ_auc <- performance(rfQ_pred, "auc")
  cat("AUC = ", as.character(attr(rfQ_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}


#---------------------#
# K-NEAREST NEIGHBORS #
#---------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Sans Quotient]
test_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors
  knn <- kknn(default~., projet_EA, projet_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, knn$fitted.values))
  
  # Courbe ROC
  knn_pred <- prediction(knn$prob[,2], projet_Q_ET$default)
  knn_perf <- performance(knn_pred,"tpr","fpr")
  plot(knn_perf, main = "ROC Classifeurs K-plus-proches-voisins [Sans Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knn_auc <- performance(knn_pred, "auc")
  cat("AUC = ", as.character(attr(knn_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}
# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Avec Quotient]
test_Q_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors
  knnQ <- kknn(default~., projet_Q_EA, projet_Q_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, knnQ$fitted.values))
  
  # Courbe ROC
  knnQ_pred <- prediction(knnQ$prob[,2], projet_Q_ET$default)
  knnQ_perf <- performance(knnQ_pred,"tpr","fpr")
  plot(knnQ_perf, main = "ROC Classifeurs K-plus-proches-voisins [Avec Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knnQ_auc <- performance(knnQ_pred, "auc")
  cat("AUC = ", as.character(attr(knnQ_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Sans Quotient]
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(default~., projet_EA, probability=TRUE, kernel = arg1)
  
  # Test du classifeur : classe predite
  svm_class <- predict(svm, projet_ET, type="response")
  
  # Matrice de confusion
  print(table(projet_ET$default, svm_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, projet_ET, probability=TRUE)
  
  # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities")
  
  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,1], projet_ET$default)
  svm_perf <- performance(svm_pred,"tpr","fpr")
  plot(svm_perf, main = "ROC Support vector machines [Sans Quotient]", add = arg2, col = arg3)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  svm_auc <- performance(svm_pred, "auc")
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}
# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Avec Quotient]
test_Q_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svmQ <- svm(default~., projet_Q_EA, probability=TRUE, kernel = arg1)
  
  # Test du classifeur : classe predite
  svmQ_class <- predict(svmQ, projet_Q_ET, type="response")
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, svmQ_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  svmQ_prob <- predict(svmQ, projet_Q_ET, probability=TRUE)
  
  # Recuperation des probabilites associees aux predictions
  svmQ_prob <- attr(svmQ_prob, "probabilities")
  
  # Courbe ROC 
  svmQ_pred <- prediction(svmQ_prob[,1], projet_Q_ET$default)
  svmQ_perf <- performance(svmQ_pred,"tpr","fpr")
  plot(svmQ_perf, main = "ROC Support vector machines [Avec Quotient]", add = arg2, col = arg3)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  svmQ_auc <- performance(svmQ_pred, "auc")
  cat("AUC = ", as.character(attr(svmQ_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------#
# NAIVE BAYES #
#-------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Sans Quotient]
test_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nb <- naive_bayes(default~., projet_EA, laplace = arg1, usekernel = arg2)
  
  # Test du classifeur : classe predite
  nb_class <- predict(nb, projet_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_ET$default, nb_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  nb_prob <- predict(nb, projet_ET, type="prob")
  
  # Courbe ROC
  nb_pred <- prediction(nb_prob[,2], projet_ET$default)
  nb_perf <- performance(nb_pred,"tpr","fpr")
  plot(nb_perf, main = "ROC Classifieurs bayésiens naïfs [Sans Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  nb_auc <- performance(nb_pred, "auc")
  cat("AUC = ", as.character(attr(nb_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Avec Quotient]
test_Q_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nbQ <- naive_bayes(default~., projet_Q_EA, laplace = arg1, usekernel = arg2)
  
  # Test du classifeur : classe predite
  nbQ_class <- predict(nbQ, projet_Q_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, nbQ_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  nbQ_prob <- predict(nbQ, projet_Q_ET, type="prob")
  
  # Courbe ROC
  nbQ_pred <- prediction(nbQ_prob[,2], projet_Q_ET$default)
  nbQ_perf <- performance(nbQ_pred,"tpr","fpr")
  plot(nbQ_perf, main = "ROC Classifieurs bayésiens naïfs [Sans Quotient]", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  nbQ_auc <- performance(nbQ_pred, "auc")
  cat("AUC = ", as.character(attr(nbQ_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Sans Quotient]
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nn <- nnet(default~., projet_EA, size = arg1, decay = arg2, maxit=arg3)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nn_class <- predict(nn, projet_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_ET$default, nn_class))
  
  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, projet_ET, type="raw")
  
  # Courbe ROC 
  nn_pred <- prediction(nn_prob[,1], projet_ET$default)
  nn_perf <- performance(nn_pred,"tpr","fpr")
  plot(nn_perf, main = "ROC Réseaux de neurones [Sans Quotient]", add = arg4, col = arg5)
  
  # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("AUC = ", as.character(attr(nn_auc, "y.values")))
  
  # Return ans affichage sur la console
  invisible()
}

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC [Avec Quotient]
test_Q_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nnQ <- nnet(default~., projet_Q_EA, size = arg1, decay = arg2, maxit=arg3)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nnQ_class <- predict(nnQ, projet_Q_ET, type="class")
  
  # Matrice de confusion
  print(table(projet_Q_ET$default, nnQ_class))
  
  # Test des classifeurs : probabilites pour chaque prediction
  nnQ_prob <- predict(nnQ, projet_Q_ET, type="raw")
  
  # Courbe ROC 
  nnQ_pred <- prediction(nnQ_prob[,1], projet_Q_ET$default)
  nnQ_perf <- performance(nnQ_pred,"tpr","fpr")
  plot(nnQ_perf, main = "ROC Réseaux de neurones [Avec Quotient]", add = arg4, col = arg5)
  
  # Calcul de l'AUC
  nnQ_auc <- performance(nnQ_pred, "auc")
  cat("AUC = ", as.character(attr(nnQ_auc, "y.values")))
  
  # Return ans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Arbres de decision
windows()
layout(matrix(1 :2, 1, 2))
test_rpart("gini", 10, FALSE, "red")
test_rpart("gini", 5, TRUE, "blue")
test_rpart("information", 10, TRUE, "green")
test_rpart("information", 5, TRUE, "orange")
test_Q_rpart("gini", 10, FALSE, "red")
test_Q_rpart("gini", 5, TRUE, "blue")
test_Q_rpart("information", 10, TRUE, "green")
test_Q_rpart("information", 5, TRUE, "orange")

# Forets d'arbres decisionnels aleatoires
windows()
layout(matrix(1 :2, 1, 2))
test_rf(300, 3, FALSE, "red")
test_rf(300, 5, TRUE, "blue")
test_rf(500, 3, TRUE, "green")
test_rf(500, 5, TRUE, "orange")
test_Q_rf(300, 3, FALSE, "red")
test_Q_rf(300, 5, TRUE, "blue")
test_Q_rf(500, 3, TRUE, "green")
test_Q_rf(500, 5, TRUE, "orange")

# K plus proches voisins
windows()
layout(matrix(1 :2, 1, 2))
test_knn(10, 1, FALSE, "red")
test_knn(10, 2, TRUE, "blue")
test_knn(20, 1, TRUE, "green")
test_knn(20, 2, TRUE, "orange")
test_Q_knn(10, 1, FALSE, "red")
test_Q_knn(10, 2, TRUE, "blue")
test_Q_knn(20, 1, TRUE, "green")
test_Q_knn(20, 2, TRUE, "orange")

# Support vector machines
windows()
layout(matrix(1 :2, 1, 2))
test_svm("linear", FALSE, "red")
test_svm("polynomial", TRUE, "blue")
test_svm("radial", TRUE, "green")
test_svm("sigmoid", TRUE, "orange")
test_Q_svm("linear", FALSE, "red")
test_Q_svm("polynomial", TRUE, "blue")
test_Q_svm("radial", TRUE, "green")
test_Q_svm("sigmoid", TRUE, "orange")

# Naive Bayes
windows()
layout(matrix(1 :2, 1, 2))
test_nb(0, FALSE, FALSE, "red")
test_nb(20, FALSE, TRUE, "blue")
test_nb(0, TRUE, TRUE, "green")
test_nb(20, TRUE, TRUE, "orange")
test_Q_nb(0, FALSE, FALSE, "red")
test_Q_nb(20, FALSE, TRUE, "blue")
test_Q_nb(0, TRUE, TRUE, "green")
test_Q_nb(20, TRUE, TRUE, "orange")

# Réseaux de neurones nnet()
windows()
layout(matrix(1 :2, 1, 2))
test_nnet(50, 0.01, 100, FALSE, "red")
test_nnet(50, 0.01, 300, TRUE, "tomato")
test_nnet(25, 0.01, 100, TRUE, "blue")
test_nnet(25, 0.01, 300, TRUE, "purple")
test_nnet(50, 0.001, 100, TRUE, "green")
test_nnet(50, 0.001, 300, TRUE, "turquoise")
test_nnet(25, 0.001, 100, TRUE, "grey")
test_nnet(25, 0.001, 300, TRUE, "black")
test_Q_nnet(50, 0.01, 100, FALSE, "red")
test_Q_nnet(50, 0.01, 300, TRUE, "tomato")
test_Q_nnet(25, 0.01, 100, TRUE, "blue")
test_Q_nnet(25, 0.01, 300, TRUE, "purple")
test_Q_nnet(50, 0.001, 100, TRUE, "green")
test_Q_nnet(50, 0.001, 300, TRUE, "turquoise")
test_Q_nnet(25, 0.001, 100, TRUE, "grey")
test_Q_nnet(25, 0.001, 300, TRUE, "black")

#----------------------------#
# PREDICTION PAR NAIVE BAYES #
#----------------------------#
# Apprentissage du classifeur 
nb <- naive_bayes(default~., projet_EA, laplace = 0, usekernel = FALSE)

# Application de Naive Bayes aux prospects dans 'projet_P' : classe predite
pred_nb <- predict(nb, projet_P, type="class")

# Test du classifeur : probabilites pour chaque prediction
prob_nb <- predict(nb, projet_P, type="prob")

# Affichage des résultats (predictions)

# Affichage du nombre de predictions pour chaque classe
table(pred_nb)

# Ajout dans le data frame produit_pro d'une colonne Predition contenant la classe prédite 
projet_P$Prediction <- pred_nb
projet_P$Probabilité <- prob_nb
projet_P <- projet_P[, -1]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P <- projet_P[, -2]
projet_P
write.csv(projet_P, file = "M JAHA Prédiction Projet Data.csv")
