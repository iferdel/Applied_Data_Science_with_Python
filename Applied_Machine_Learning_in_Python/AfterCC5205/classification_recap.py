from ssl import ALERT_DESCRIPTION_UNEXPECTED_MESSAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, permutation_test_score, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import set_config
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, VarianceThreshold, SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay ,accuracy_score, auc, ConfusionMatrixDisplay, confusion_matrix, hinge_loss, plot_confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesClassifier
from textwrap import wrap
import copy as cp


def main():
    
    cancer = load_breast_cancer()
    print('directorio de data: ', cancer.__dir__())
    features = cancer['feature_names']
    print('features de la data: ', pd.Series(features)) 
    df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']
    target_summary = zip(cancer['target_names'],df['target'].value_counts().sort_values())
    print('conteo de valores objetivo: ', list(target_summary))
    X, y = df.drop('target', axis=1), df['target']
    print('shape de la data: {}; shape del target value {}'.format(X.shape, y.shape))

    # RANDOM STATE FOR REPRODUCIBLE RESULTS
    rng = np.random.RandomState(0) # Note We do not recommend setting the global numpy seed by calling np.random.seed(0)
    # When you go to production, you should remove the random_state and/or random_seed settings, or set to None, then do some cross validation. This will give you more realistic results from your model.
    
    # NORMALIZATION
    zscore = StandardScaler()
    minmax = MinMaxScaler()

    # FEATURE SELECTION
    varsel = VarianceThreshold(threshold=0.5)
    tree_selector = ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_split=5, criterion='entropy', random_state=rng, n_jobs=-1) 

    # DATA REDUCCTION
    # PCA for dense data, TSVD for sparse data, T-SNE for visualize data (probabilistic approach) 
    kpca = KernelPCA(n_components=2, kernel='linear', random_state=rng, n_jobs=-1)
    tsvd = TruncatedSVD(n_components=2, random_state=rng) 

    # K-FOLDS
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=rng)
    #It is preferable to evaluate the cross-validation performance by letting the estimator use a different RNG on each fold. This is done by passing a RandomState instance (or None) to the estimator initialization.
    print(skf.get_n_splits(X, y))
    print(skf)
    
    # MODELS
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', n_jobs=-1) 
    lr = LogisticRegression(dual=False, penalty='l2', random_state=rng) # Prefer dual=False when n_samples > n_features.
    svc = OneVsRestClassifier(SVC(C=0.01, cache_size=300, gamma=0.001, kernel='sigmoid', random_state=rng)) # possibly changeable with SGD
    # dtree = DecisionTreeClassifier(random_state=rng)
    # nbgss = GaussianNB(random_state=rng)
    # gauss = GaussianProcessClassifier(random_state=rng)
    
    # PIPES
    knnpipe = Pipeline(steps=[('normalization', zscore), ('feature selection', varsel), ('feature reduction', kpca), ('classification model', knn)])
    lrpipe = Pipeline(steps=[('normalization', minmax), ('feature selection', SelectFromModel(tree_selector)), ('feature reduction', tsvd), ('classification model', lr)])
    svcpipe = Pipeline(steps=[('normalization', zscore), ('feature selection', SequentialFeatureSelector(estimator=svc, n_features_to_select=5, direction='backward', n_jobs=-1)), ('classification model', svc)])
                                                                             #issues with input in SelectFromModel as threshold='mean'
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=rng)
    
    #pipeline performance
    scoring_metrics = ['accuracy', 'average_precision', 'f1', 'precision', 'recall', 'jaccard', 'roc_auc']
    
    # KNN
    knnscores = cross_validate(knnpipe, X, y, scoring=scoring_metrics, cv=skf, n_jobs=-1) 
    knnpredicted = cross_val_predict(knnpipe, X, y, cv=skf, n_jobs=-1)
    # plot may work with another classfifier approach such as regressional, for this case there is no possible visualization of amount of data, only FP, FN, TP, TN and there is only one value for every one on those.
    # Plotting Cross-Validated Predictions
    plot_predictions = 0
    if plot_predictions == 1:    
        fig, ax = plt.subplots()
        ax.scatter(y, knnpredicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        plt.show(block=False)
    knnpermutation = permutation_test_score(knnpipe, X, y, n_permutations=(100+1)*skf.get_n_splits(), scoring='roc_auc', random_state=rng, cv=skf, n_jobs=-1)
    # It provides a permutation-based p-value, which represents how likely an observed performance of the classifier would be obtained by chance.
    # A low p-value provides evidence that the dataset contains real dependency between features and labels and the classifier was able to utilize this to obtain good results. 
    # A high p-value could be due to a lack of dependency between features and labels (there is no difference in feature values between the classes) or because the classifier was not able to use the dependency in the data. In the latter case, using a more appropriate classifier that is able to utilize the structure in the data, would result in a lower p-value.
    # Cross-validation provides information about how well a classifier generalizes, specifically the range of expected errors of the classifier. However, a classifier trained on a high dimensional dataset with no structure may still perform better than expected on cross-validation, just by chance. 
    # permutation_test_score provides information on whether the classifier has found a real class structure and can help in evaluating the performance of the classifier.
    # OPTIONAL APPROACH
    optional = 0
    if optional == 1:
        p_grid = {"n_neighbors": [5, 10, 15]}
        knncrossnested = KNeighborsClassifier()
        # Availability of SelectKBest()
        NUM_TRIALS = 20
        nested_scores = np.zeros(NUM_TRIALS)
        # Loop for each trial
        for i in range(NUM_TRIALS):
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=i)
            clf = GridSearchCV(estimator=knncrossnested, param_grid=p_grid, cv=inner_cv)
            clf.fit(X, y)
            print('for trial number {}, the best paramameters are: {}'.format(i, clf.best_params_))
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
            nested_scores[i] = nested_score.mean()
        print(sorted(clf.cv_results_.keys()))
        plt.figure()
        plt.subplot(111)
        (nested_line,) = plt.plot(nested_scores, color="b")
        plt.xlabel("trial number", fontsize="14")
        plt.ylabel("score", fontsize="14")
        plt.legend(
            [nested_line],
            ["Nested CV"],
            bbox_to_anchor=(0, 0.4, 0.5, 0),
        )
        plt.title(
            "Nested Cross Validation on Dataset",
            x=0.5,
            y=1.1,
            fontsize="15",
        )
        plt.show()

    # LOGISTIC REGRESSION, SUPPORT VECTOR CLASSIFIER
    lrscores = cross_validate(lrpipe, X, y, scoring=scoring_metrics, cv=skf, n_jobs=-1)
    svcscores = cross_validate(svcpipe, X, y, scoring=scoring_metrics, cv=skf, n_jobs=-1) 
    #  In the case of the Iris dataset, the samples are balanced across target classes hence the accuracy and the F1-score are almost equal.
    
    ROC_AUC_PLOT = 1
    if ROC_AUC_PLOT == 1:
        pipes = [cp.deepcopy(knnpipe), cp.deepcopy(lrpipe)]
        # pipes = [knnpipe, lrpipe, svcpipe]
        for key, pipe in enumerate(pipes):
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            curr_fold = 1
            for train, test in skf.split(X, y):
                pipe.fit(X.iloc[train], y.iloc[train])
                PrecisionRecallDisplay.from_estimator(
                    pipe, 
                    X.iloc[test],
                    y.iloc[test])
                plt.show()
                plt.subplot(1, skf.get_n_splits(), curr_fold)
                ax_folds = plt.gca()
                fig_folds = plt.gcf()
                fig_folds.set_size_inches(18, 7)
                fig_folds.tight_layout()
                
                viz = RocCurveDisplay.from_estimator(
                    pipe,
                    X.iloc[test],
                    y.iloc[test],
                    name="ROC fold {}".format(curr_fold),
                    alpha=0.3,
                    lw=1,
                    ax=ax_folds,
                )
                curr_fold += 1
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            fig, ax = plt.subplots()
            ax = plt.gca()
            fig = plt.gcf()
            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            ax.set(

                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title="\n".join(wrap(text="Receiver operating characteristic for {} model".format(pipe.__getitem__(-1)), width=40)),
            )
            fig.set_size_inches(7,7)
            ax.legend(loc="lower right")
            fig.tight_layout()

            plt.show()
   
    # t-SNE plotter
    t_sne_ = 0
    if t_sne_ == 1:
        for perplexity in range(5, 50, 5):
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, learning_rate=200, n_jobs=-1, random_state=rng) 
            X_pca = PCA(n_components=10, random_state=rng).fit_transform(X)
            rows = np.arange(569)
            np.random.shuffle(rows)
            n_select = 100    
            tsne_results = tsne.fit_transform(X_pca[rows[:n_select],:])
            df_tsne = pd.DataFrame(tsne_results, columns=['comp1', 'comp2'])
            df_tsne['label'] = y[rows[:n_select]]
            sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)
            plt.show()
            # If choosing different values between 5 and 50 significantly change your interpretation of the data, then you should consider other ways to visualize or validate your hypothesis.
            # for further explanations https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0

    # ConfusionMatrixDisplay
    fig_dep, (ax1_dep, ax2_dep) = plt.subplots(1, 2, figsize=(10, 10))
    split_data = 0
    def knnpipe_function():
        for i in scoring_metrics:
            print('cross validation values for score metric {} are: '.format(i), knnscores['test_{}'.format(i)])
        set_config(display="text")
        print(knnpipe)
        no_classes = len(np.unique(y))
        actual_classes = np.empty([0], dtype=int)
        predicted_classes = np.empty([0], dtype=int)
        predicted_proba = np.empty([0, no_classes]) 
        #stratification performance
        knnlst_accu_stratified = []
        for train, test in skf.split(X, y):
            if split_data == 1: print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
            X_train_fold, X_test_fold = X.iloc[train], X.iloc[test]
            y_train_fold, y_test_fold = y.iloc[train], y.iloc[test]
            
            actual_classes = np.append(actual_classes, y_test_fold)
            knnpipe.fit(X_train_fold, y_train_fold)
            predicted_classes = np.append(predicted_classes, knnpipe.predict(X_test_fold))
            try:
                predicted_proba = np.append(predicted_proba, knnpipe.predict_proba(X_test_fold), axis=0)
            except:
                predicted_proba = np.append(predicted_proba, np.zeros((len(X_test_fold), no_classes), dtype=float), axis=0)
            print('feature names out: ', knnpipe['feature selection'].get_feature_names_out())
            knnlst_accu_stratified.append(knnpipe.score(X_test_fold, y_test_fold))
            # PartialDependenceDisplay.from_estimator(knnpipe, X_test_fold, ["mean texture", "mean perimeter"], n_jobs=-1, random_state=rng, ax=ax1_dep)
            # plt.suptitle(knnpipe['classification model'])
            # plt.show() 
        knn_disp = PartialDependenceDisplay.from_estimator(knnpipe, X_test_fold, ["mean texture", "mean perimeter"], n_jobs=-1, random_state=rng)
        knn_disp.plot(ax=[ax1_dep, ax2_dep], line_kw={"label": "KNN"})
        print("KNN score list: ", knnlst_accu_stratified)
        print("KNN mean score: ", np.mean(knnlst_accu_stratified))
        return (actual_classes, predicted_classes, predicted_proba)

    def lrpipe_function():
        for i in scoring_metrics:
            print('cross validation values for score metric {} are: '.format(i), knnscores['test_{}'.format(i)])
        set_config(display="text")
        print(lrpipe)
        #stratification performance
        lrlst_accu_stratified = []
        for train, test in skf.split(X, y):
            if split_data == 1: print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
            X_train_fold, X_test_fold = X.iloc[train], X.iloc[test]
            y_train_fold, y_test_fold = y.iloc[train], y.iloc[test]
            lrpipe.fit(X_train_fold, y_train_fold)
            print('feature names out: ', lrpipe['feature selection'].get_feature_names_out())
            lrlst_accu_stratified.append(lrpipe.score(X_test_fold, y_test_fold))
            # PartialDependenceDisplay.from_estimator(lrpipe, X_test_fold, ["mean texture", "mean perimeter"], n_jobs=-1, random_state=rng, ax=ax2_dep)
            # plt.suptitle(lrpipe['classification model'])
            # plt.show()     
        lr_disp = PartialDependenceDisplay.from_estimator(lrpipe, X_test_fold, ["mean texture", "mean perimeter"], n_jobs=-1, random_state=rng)
        lr_disp.plot(ax=[ax1_dep, ax2_dep], line_kw={"label": "Logarithmic regression", "color": "red"})
        print("Logarithmic regression score list: ",lrlst_accu_stratified)
        print("Logarithmic regression mean score: ", np.mean(lrlst_accu_stratified))

    def svcpipe_function():
        for i in scoring_metrics:
            print('cross validation values for score metric {} are: '.format(i), knnscores['test_{}'.format(i)])
        set_config(display="text")
        print(svcpipe)
        #stratification performance
        svclst_accu_stratified = []
        for train, test in skf.split(X, y):
            print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))
            X_train_fold, X_test_fold = X.iloc[train], X.iloc[test]
            y_train_fold, y_test_fold = y.iloc[train], y.iloc[test]
            svcpipe.fit(X_train_fold, y_train_fold)
            print('feature names out: ', svcpipe['feature selection'].get_feature_names_out())
            svclst_accu_stratified.append(svcpipe.score(X_test_fold, y_test_fold))
        print(svclst_accu_stratified)
        print(np.mean(svclst_accu_stratified))

    def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array): #, sorted_labels : list):
        matrix = confusion_matrix(y_true=actual_classes, y_pred=predicted_classes)#, labels=sorted_labels) 
        plt.figure(figsize=(12.8,6))
        sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")#, xticklabels=sorted_labels, yticklabels=sorted_labels)
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
        plt.show()

    knnpipe_function()
    lrpipe_function()
    ax1_dep.legend()
    ax2_dep.legend()
    plt.show()

    actual_classes, predicted_classes, _ = knnpipe_function()
    print(actual_classes, predicted_classes, _)
    plot_confusion_matrix(actual_classes, predicted_classes)#, ["Positive", "Negative"])

if __name__ == '__main__':
    main()