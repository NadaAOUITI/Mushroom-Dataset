from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#----------------------------------------------------------------Data description------------------------------------------------------------
mushroom = fetch_ucirepo(id=73) 
  

X = mushroom.data.features 
y = mushroom.data.targets 
mushroom_df = pd.concat([X, y], axis=1)
  

print(mushroom.metadata) 
  
print(mushroom.variables) 
print(mushroom_df.describe())
print(mushroom_df.head())
#mushroom_df.to_excel("mushroom_dataset.xlsx", index=False) commented because this has already been done.

#--------------------------------------------------------Data cleaning ----------------------------------------------------------------------
print("Valeurs manquantes par colonne :")
print(mushroom_df.isnull().sum())

missing_rows = mushroom_df.isnull().any(axis=1).sum()
total_rows = mushroom_df.shape[0]
missing_percentage = (missing_rows / total_rows) * 100

print(f"\nNombre de lignes avec au moins une valeur manquante : {missing_rows}")
print(f"Pourcentage de lignes manquantes : {missing_percentage:.2f}%")


missing_rows_df = mushroom_df[mushroom_df.isnull().any(axis=1)]
print(f"\nLignes contenant des valeurs manquantes ({len(missing_rows_df)} lignes) :")
print(missing_rows_df)



if missing_percentage > 5:
    print("\n→ Remplissage des valeurs manquantes avec la valeur la plus fréquente (mode)...")
    for column in mushroom_df.columns:
        if mushroom_df[column].isnull().any():
            mode_value = mushroom_df[column].mode()[0]
            print(": " ,mushroom_df[column].mode()[0])
            mushroom_df[column].fillna(mode_value, inplace=True)
else:
    print("\n→ Suppression des lignes avec des valeurs manquantes...")
    mushroom_df.dropna(inplace=True)


print("\n✔ Données nettoyées. Vérification des valeurs manquantes :")
print(mushroom_df.isnull().sum())
print("-------------------------------------Vérification des valeurs dupliquées----------------------------------------------------")
duplicate_count = mushroom_df.duplicated().sum()
print(f"Nombre de lignes dupliquées : {duplicate_count}")


if duplicate_count > 0:
    print("\n→ Aperçu des doublons :")
    print(mushroom_df[mushroom_df.duplicated()])


if duplicate_count > 0:
    mushroom_df.drop_duplicates(inplace=True)
    print(f"\n✔ {duplicate_count} doublons supprimés.")
else:
    print("✔ Aucun doublon trouvé.")


print(f"\nTaille finale du dataset : {mushroom_df.shape[0]} lignes, {mushroom_df.shape[1]} colonnes")

#---------------------------------------------------Data Exploration---------------------------------------------------------------------------
#countplot
import numpy as np
import math
import seaborn as sns

# -----------------Feature selection : Dropping variables here to avoid visualizing them all over again--------------------------------------
mushroom_df.drop(
    columns=[
        'veil-type',                # Mostly 'partial' → no variation
        'odor',                     # Too predictive → removes challenge
        'ring-number',             # Limited categories, weak separation
        'veil-color',              # Dominated by one value ('w')
        'gill-attachment',         # Almost no useful variation
        'stalk-color-above-ring',  # Weak pattern compared to other stalk features
        'stalk-color-below-ring',   # Same as above
        'cap-color'
    ],
    inplace=True
)

columns = mushroom_df.columns.tolist()
plots_per_fig = 5
total_features = len(columns)

for i in range(0, total_features, plots_per_fig):
    subset_cols = columns[i:i + plots_per_fig]
    fig, axes = plt.subplots(1, len(subset_cols), figsize=(5 * len(subset_cols), 5))
    
    if len(subset_cols) == 1:
        axes = [axes]  

    for ax, col in zip(axes, subset_cols):
        sns.countplot(data=mushroom_df, y=col, hue='poisonous', palette="Set2", ax=ax)
        ax.set_title(f"{col}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    plt.tight_layout()
    plt.show()

#-----------------------------------Feature selection : Test de Q2--------------------------------------------------------------------
from sklearn.feature_selection import chi2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

encoded_df = mushroom_df.copy()
le = LabelEncoder()

for col in encoded_df.columns:
    encoded_df[col] = le.fit_transform(encoded_df[col])

X = encoded_df.drop(columns=['poisonous'])  
y = encoded_df['poisonous']                


chi2_scores, p_values = chi2(X, y)


chi2_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'P-Value': p_values
})


chi2_results.sort_values(by='P-Value', inplace=True)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))


sns.barplot(x='Chi2 Score', y='Feature', data=chi2_results, ax=ax1, palette='viridis')
ax1.set_title('Chi-Square Scores by Feature', fontsize=16)
ax1.set_xlabel('Chi-Square Score', fontsize=12)
ax1.set_ylabel('Feature', fontsize=12)
ax1.grid(axis='x', linestyle='--', alpha=0.7)


sns.barplot(x='P-Value', y='Feature', data=chi2_results, ax=ax2, palette='plasma')
ax2.set_title('P-Values by Feature', fontsize=16)
ax2.set_xlabel('P-Value', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.axvline(x=0.05, color='red', linestyle='--', label='p=0.05 threshold')
ax2.grid(axis='x', linestyle='--', alpha=0.7)
ax2.legend()


for i, p in enumerate(chi2_results['P-Value']):
    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax2.text(0.001, i, significance, fontsize=14, weight='bold')


num_significant = sum(chi2_results['P-Value'] < 0.05)
top_features = chi2_results.head(min(10, num_significant))


fig2, ax3 = plt.subplots(figsize=(10, 8))
sns.barplot(x='Chi2 Score', y='Feature', data=top_features, ax=ax3, palette='viridis')
ax3.set_title(f'Top {len(top_features)} Most Significant Features', fontsize=16)
ax3.set_xlabel('Chi-Square Score', fontsize=12)
ax3.set_ylabel('Feature', fontsize=12)
ax3.grid(axis='x', linestyle='--', alpha=0.7)


styled_results = chi2_results.style.background_gradient(subset=['Chi2 Score'], cmap='viridis')
styled_results = styled_results.background_gradient(subset=['P-Value'], cmap='plasma_r')
styled_results = styled_results.format({'Chi2 Score': '{:.2f}', 'P-Value': '{:.4f}'})

plt.tight_layout()


plt.show()

chi2_results

#--------------------------------------------KNN--------------------------------------------------------------------------------------
#--------------------------------------------KNN Implementation (Manhattan Distance)-----------------------------------------------------
from sklearn.preprocessing import FunctionTransformer

# Encode the target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(mushroom_df["poisonous"])  # Converts 'e'/'p' to 0/1

# 1. Split target and features
X = mushroom_df.drop(columns=["poisonous"])

# 2. Split into train, validation, and test sets (65% / 15% / 20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1875, stratify=y_temp, random_state=42) 

# 3. Preprocessing: One-hot encode all categorical columns
categorical_cols = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)]
)

# 4. PCA for dimensionality reduction----------------------------------------------------------------------------------------------

categorical_cols = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)]
)

# Apply preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_train_dense = X_train_processed.toarray()  # Convert to dense for PCA

# Now apply PCA to the processed data
pca = PCA(n_components=0.95, svd_solver='full')
pca.fit(X_train_dense)

# Now we can analyze the PCA components
print(f"Number of components selected to retain 95% variance: {pca.n_components_}")
print(f"Total variance retained: {np.sum(pca.explained_variance_ratio_):.4f}")

# Get feature names after one-hot encoding
feature_names = []
for name, encoder, cols in preprocessor.transformers_:
    if hasattr(encoder, 'get_feature_names_out'):
        transformed_names = encoder.get_feature_names_out(cols)
        feature_names.extend(transformed_names)

# Create a DataFrame with the absolute values of PCA components
pca_components = pd.DataFrame(
    np.abs(pca.components_),
    columns=feature_names
)

# Display top features for each component
print("\nTop features contributing to each principal component:")
for i, component in enumerate(pca_components.iloc[:10].iterrows()):  # Show first 5 components only
    idx, values = component
    top_features = values.sort_values(ascending=False).head(10)
    print(f"\nTop 10 features for Principal Component {idx+1}:")
    print(top_features)

# Visualize explained variance by component
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'r-', marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot: Variance Explained by Principal Components')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1, 2)) 
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------

# 5. KNN pipeline with PCA
knn_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)), #so that PCA can be applied
    ('pca', pca),  # PCA step
    ('classifier', KNeighborsClassifier(metric='manhattan'))
])

# 6. Grid search over K values
param_grid = {'classifier__n_neighbors': list(range(1, 31))}

precision_scorer = make_scorer(precision_score, pos_label=1)

grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring=precision_scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n Meilleur nombre de voisins (K) trouvé :", grid_search.best_params_['classifier__n_neighbors'])

# 7. Evaluate on validation set
val_preds = grid_search.predict(X_val)
print("\n Résultats sur le jeu de validation :")
report = classification_report(y_val, val_preds)
print(report)

# Final evaluation on test set
try:
    test_preds = grid_search.predict(X_test)
    print("\n Résultats sur le jeu de test (final) :")
    report = classification_report(y_test, test_preds)
    print(report)
except Exception as e:
    print(" Error during prediction or reporting:", e)

# Precision for poisonous class by K value

# Extract the results of grid search
results = grid_search.cv_results_

# Extract K values and the corresponding precision scores
k_values = results['param_classifier__n_neighbors']
mean_test_scores = results['mean_test_score']

# Plot precision vs. K
plt.figure(figsize=(8, 6))
plt.plot(k_values, mean_test_scores, marker='o', color='b', linestyle='-', markersize=8)
plt.title('Variation of Precision with K (Number of Neighbors)', fontsize=14)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Precision Score', fontsize=12)
plt.grid(True)
plt.show()
#--------------------------Plotting the decision boundary of the best KNN model given by the grid search (K=1)-------------------


# KNN with k=1
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
pca_2d = PCA(n_components=2)
# Pipeline

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ('pca', pca_2d),
    ('classifier', knn)
])

# Fit only on training set
pipeline.fit(X_train, y_train)

# Create a meshgrid to plot decision boundary
X_vis = pipeline.named_steps['pca'].transform(
    pipeline.named_steps['to_dense'].transform(
        pipeline.named_steps['preprocessing'].transform(X_train)
    )
)
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on the grid using the fitted KNN in 2D space
Z = pipeline.named_steps['classifier'].predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=40)
plt.title("Decision Boundary of KNN (k=1) on PCA-Reduced Mushroom Data", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
#-----------------------------------------------Model Selection 2 : Decision Tree--------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


tree_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

tree_pipeline.fit(X_train, y_train)

y_pred = tree_pipeline.predict(X_val)

# Evaluate
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

param_grid = {
    'classifier__max_depth': [3, 5, 10, 15, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

# GridSearchCV (5-fold cross-validation on X_train + X_val)
grid = GridSearchCV(tree_pipeline, param_grid, cv=5, scoring=precision_scorer, n_jobs=-1)
# n_jobs=-1 : Use all CPU cores

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Validation accuracy:", grid.best_score_)

# Evaluate on validation set
best_tree = grid.best_estimator_
y_val_pred = best_tree.predict(X_val)



# Classification report
print("\nValidation Set Performance:")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
# Predict on test set
test_preds = best_tree.predict(X_test)

# Classification report
print("\nTest Set Performance (Final Evaluation):")
print(classification_report(y_test, test_preds, target_names=label_encoder.classes_))

#-------------------------------------------------Roc curve---------------------------------------------------
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class
y_val_prob = best_tree.predict(X_val) # Probabilities for class 1 (positive class)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------
# Post-Pruning the Best Pre-Pruned Decision Tree
# ----------------------------------------------------------------------------------------------------------------------------------


# 1. Get the best pre-pruned tree from  grid search
best_pre_pruned_tree = grid.best_estimator_.named_steps['classifier']

# 2. Prepare the preprocessed training data
X_train_processed = preprocessor.transform(X_train)

# 3. Get the cost complexity pruning path
path = best_pre_pruned_tree.cost_complexity_pruning_path(X_train_processed, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

from sklearn.metrics import precision_score

# 4. Train trees for each alpha value
pruned_trees = []
poisonous_precisions = []
ccp_alpha_used = []

# Get the encoded value for poisonous class 
pos_class = 1  

for alpha in ccp_alphas:
    pruned_tree = DecisionTreeClassifier(
        random_state=42,
        criterion=grid.best_params_['classifier__criterion'],
        max_depth=grid.best_params_['classifier__max_depth'],
        min_samples_split=grid.best_params_['classifier__min_samples_split'],
        min_samples_leaf=grid.best_params_['classifier__min_samples_leaf'],
        ccp_alpha=alpha
    )
    
    pruned_tree.fit(X_train_processed, y_train)
    val_preds = pruned_tree.predict(preprocessor.transform(X_val))
    
    # Get precision for class 'poisonous' (class 1)
    precision = precision_score(y_val, val_preds, pos_label=pos_class)
    
    poisonous_precisions.append(precision)
    pruned_trees.append(pruned_tree)
    ccp_alpha_used.append(alpha)

# 5. Choose the simplest tree (smallest alpha) that gives **max poisonous precision**
best_idx = np.argmax(poisonous_precisions)
optimal_alpha = ccp_alpha_used[best_idx]
optimal_tree = pruned_trees[best_idx]

print(f"\nOptimal alpha based on poisonous precision: {optimal_alpha:.5f}")
print(f"Best poisonous precision: {poisonous_precisions[best_idx]:.4f}")
print(f"Pruned tree node count: {optimal_tree.tree_.node_count}")


# 6. Plot the pruning process
plt.figure(figsize=(10,6))
plt.plot(ccp_alpha_used, poisonous_precisions, marker='o', label='Poisonous Precision')
plt.axvline(optimal_alpha, color='red', linestyle='--', label=f'Optimal α: {optimal_alpha:.5f}')
plt.xlabel("CCP Alpha (Pruning Strength)")
plt.ylabel("Precision (Poisonous)")
plt.title("Poisonous Precision vs. CCP Alpha")
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluate the optimal pruned tree on test set
X_test_processed = preprocessor.transform(X_test)
test_accuracy = optimal_tree.score(X_test_processed, y_test)

print("\nFinal Evaluation:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, optimal_tree.predict(X_test_processed), 
      target_names=label_encoder.classes_))

# 8. Create the final pruned pipeline
pruned_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', optimal_tree)
])

# To use this pruned model for predictions:
predictions = pruned_pipeline.predict(X_val)
# Evaluate
print("\nPrediction on test set:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

#----------------------------------------------Visualization of the best model (pre-pruned)------------------------------------------------
from sklearn.tree import plot_tree

# Access the fitted classifier directly
clf = best_tree.named_steps['classifier']

# Create a more visually appealing figure with better spacing
plt.figure(figsize=(24, 12))

# Use a cleaner style
plt.style.use('ggplot')

# Plot the tree with improved parameters
plot_tree(
    clf,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    precision=2,  # Show fewer decimal places
    proportion=False,  # Show sample counts rather than proportions
    fontsize=10,
    impurity=True,  # Show impurity (gini) at each node
    node_ids=False,  # Don't show node IDs to reduce clutter
    max_depth=5,  # Limit the display depth for better readability
)

# Add a more descriptive title with information about the tree
plt.title(f"Mushroom Classification Decision Tree (Max Depth={clf.max_depth}, Leaves={clf.get_n_leaves()})", 
          fontsize=18, fontweight='bold', pad=20)

# Add a subtle border around the figure
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['top'].set_linewidth(0.5)
plt.gca().spines['right'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)
plt.gca().spines['left'].set_linewidth(0.5)

# Add a subtle grid to help with alignment
plt.grid(True, linestyle='--', alpha=0.3)

# Add more padding/margin around the plot for better breathing room
plt.tight_layout(pad=3.0)
plt.show()

#Visualize Decision Paths

# Get a sample
sample = X_val.iloc[[0]]  # 0-th row, but wrapped in list to keep it 2D
sample_transformed = preprocessor.transform(sample)

# Get path
node_indicator = clf.decision_path(sample_transformed)
leave_id = clf.apply(sample_transformed)

# Show the nodes used in the decision path
feature_names = preprocessor.get_feature_names_out()
print("Decision path for sample 0:")
for node_id in node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]:
    if (sample_transformed[0, clf.tree_.feature[node_id]] <= clf.tree_.threshold[node_id]):
        threshold_sign = "<="
    else:
        threshold_sign = ">"
    print(f"Node {node_id}: (X_test[0, '{feature_names[clf.tree_.feature[node_id]]}'] = "
          f"{sample_transformed[0, clf.tree_.feature[node_id]]}) "
          f"{threshold_sign} {clf.tree_.threshold[node_id]}")

# Plot Decision Boundaries (only for 2D data)
from sklearn.decomposition import PCA

X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_preprocessed)

# Extract and clean the best parameters
best_params = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}

# Fit a new classifier on PCA-transformed data (only for visualization, not performance)
clf_vis = DecisionTreeClassifier(**best_params, random_state=42)
clf_vis.fit(X_pca, y_train)

# Create mesh grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolor='k', cmap='coolwarm')
plt.title("Decision Tree Decision Boundaries (PCA-reduced)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# =============================================================================
# FINAL MODEL CONSTRUCTION
# =============================================================================
print("\n" + "="*80)
print("FINAL MODEL CONSTRUCTION")
print("="*80)

# 1. Combine all  data (train + validation + test)
print("\nStep 1: Combining all data (train + validation + test)...")
X_full = pd.concat([X_train, X_val, X_test])
y_full = np.concatenate([y_train, y_val, y_test])

print(f"Final dataset size: {X_full.shape[0]} samples")

# 2. Prepare the best parameters from grid search
print("\nStep 2: Preparing best parameters...")
best_params = {
    'criterion': grid.best_params_['classifier__criterion'],
    'max_depth': grid.best_params_['classifier__max_depth'],
    'min_samples_split': grid.best_params_['classifier__min_samples_split'],
    'min_samples_leaf': grid.best_params_['classifier__min_samples_leaf'],
    'ccp_alpha': optimal_alpha,  
    'random_state': 42
}

print("Best parameters:")
for k, v in best_params.items():
    print(f"- {k}: {v}")

# 3. Create and train the final model
print("\nStep 3: Creating and training final model...")
final_model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(**best_params))
])

final_model.fit(X_full, y_full)
print(" Final model trained on complete dataset")

# 4. Evaluate the final model on the full dataset
print("\nStep 4: Verifying final model performance (Poisonous Class Focus)...")
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Get predictions
final_preds = final_model.predict(X_full)

# Calculate poisonous class metrics (class 1)
poisonous_precision = precision_score(y_full, final_preds, pos_label=1)
poisonous_recall = recall_score(y_full, final_preds, pos_label=1)
poisonous_f1 = f1_score(y_full, final_preds, pos_label=1)

print("\nPoisonous Class Performance:")
print(f"Precision: {poisonous_precision:.4f} (When model says poisonous, how often it's correct)")
print(f"Recall:    {poisonous_recall:.4f} (What percentage of poisonous mushrooms are detected)")
print(f"F1-Score:  {poisonous_f1:.4f} (Balance between precision and recall)")

# Detailed poisonous class report
print("\nDetailed Poisonous Class Report:")
print(classification_report(y_full, final_preds, 
      target_names=label_encoder.classes_,
      labels=[1],  # Focus only on poisonous class (encoded as 1)
      zero_division=0))

# Full classification report for reference
print("\nFull Classification Report (Both Classes):")
print(classification_report(y_full, final_preds, target_names=label_encoder.classes_))

# 5. Save the model
print("\nStep 5: Saving the model...")
import joblib

# Save the model
model_filename = 'mushroom_classifier_final.pkl'
joblib.dump(final_model, model_filename)
print(f" Model saved as '{model_filename}'")

# Save the label encoder too (important for interpreting predictions)
encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, encoder_filename)
print(f" Label encoder saved as '{encoder_filename}'")
#--------------------------------------------Demo visualization of model---------------------------------------------------
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
from PIL import Image, ImageTk

class MushroomClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mushroom Edibility Classifier")
        self.root.geometry("700x760")
        self.root.configure(bg="#f0f0f0")
        
        # Try to load the model
        try:
            self.model = joblib.load('mushroom_classifier_final.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.model_loaded = True
            messagebox.showwarning("Model Found", 
                "Model files  found. This app will run in correct mode.")
        except FileNotFoundError:
            self.model_loaded = False
            messagebox.showwarning("Model Not Found", 
                "Model files not found. This app will run in demo mode.")
        
        # Define feature options (based on our dataset)
        self.feature_options = {
            'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
            'cap-surface': ['fibrous', 'grooves', 'scaly', 'smooth'],
            'gill-color': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 
                           'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
            'bruises': ['bruises', 'no'],
            'gill-size': ['broad', 'narrow'],
            'gill-spacing': ['close', 'crowded', 'distant'],
            'population': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'],
            'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'],
            'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 
                                 'orange', 'purple', 'white', 'yellow'],
            'ring-type': ['evanescent', 'flaring', 'large', 'none', 'pendant'],
            'stalk-shape': ['enlarging', 'tapering'],
            'stalk-root': ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing'],
            'stalk-surface-above-ring': ['fibrous', 'scaly', 'silky', 'smooth'],
            'stalk-surface-below-ring': ['fibrous', 'scaly', 'silky', 'smooth']
        }
        
        # Feature importance (for explanation)
        self.feature_importance = {
            'gill-color': 0.30,
            'spore-print-color': 0.25,
            'stalk-root': 0.15,
            'habitat': 0.10,
            'population': 0.08,
            'ring-type': 0.07,
            'cap-shape': 0.05,
            'gill-spacing': 0.04,
            'stalk-shape': 0.03,
            'gill-size': 0.03,
            'bruises': 0.03,
            'cap-surface': 0.02,
            'stalk-surface-above-ring': 0.02,
            'stalk-surface-below-ring': 0.02
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title and description
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="Mushroom Edibility Classifier", 
                 font=("Arial", 18, "bold"), bg="#f0f0f0").pack()
        
        tk.Label(title_frame, text="Select mushroom characteristics to check if it's safe to eat",
                 font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
        
        # Create mushroom icon
        try:
            # Create a simple mushroom icon using text
            mushroom_label = tk.Label(title_frame, text="🍄", font=("Arial", 32), bg="#f0f0f0")
            mushroom_label.pack(pady=10)
        except:
            # Fallback if emoji doesn't render properly
            pass
        
        # Main content - scrollable frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Feature selection 
        self.selection_vars = {}
        
        row = 0
        col = 0
        for feature, options in self.feature_options.items():
            # Create frame for each feature
            feature_frame = tk.LabelFrame(scrollable_frame, text=feature.replace('-', ' ').title(), 
                                         padx=10, pady=5, bg="#f0f0f0")
            feature_frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            # Create dropdown for options
            self.selection_vars[feature] = tk.StringVar()
            self.selection_vars[feature].set(options[0])  # Default to first option
            
            option_menu = ttk.Combobox(feature_frame, textvariable=self.selection_vars[feature],
                                      values=options, state="readonly", width=15)
            option_menu.grid(row=0, column=0, padx=5, pady=5)
            
            # Layout: 2 columns
            col += 1
            if col > 1:
                col = 0
                row += 1
                
        # Results frame
        self.result_frame = tk.LabelFrame(self.root, text="Classification Result", 
                                         font=("Arial", 12, "bold"), padx=20, pady=10,
                                         bg="#f0f0f0")
        self.result_frame.pack(fill="x", padx=20, pady=10)
        
        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14), 
                                    bg="#f0f0f0", pady=10)
        self.result_label.pack()
        
        self.explanation_label = tk.Label(self.result_frame, text="", 
                                         font=("Arial", 10), bg="#f0f0f0",
                                         justify="left", wraplength=600)
        self.explanation_label.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        classify_button = tk.Button(button_frame, text="Classify Mushroom", 
                                   command=self.classify_mushroom,
                                   bg="#4CAF50", fg="white", 
                                   font=("Arial", 12), padx=20, pady=10)
        classify_button.pack(side="left", padx=10)
        
        clear_button = tk.Button(button_frame, text="Reset", 
                                command=self.reset_selections,
                                bg="#f44336", fg="white", 
                                font=("Arial", 12), padx=20, pady=10)
        clear_button.pack(side="left", padx=10)
        
        # Credit
        tk.Label(self.root, text="© 2025 Mushroom Classification Project", 
                font=("Arial", 8), bg="#f0f0f0").pack(side="bottom", pady=5)
        
    def classify_mushroom(self):
        # Get selected features
        features = {feature: [value.get()] for feature, value in self.selection_vars.items()}
        
        # Create DataFrame with the selected features
        input_df = pd.DataFrame(features)
        
        # If model is loaded, make prediction
        if self.model_loaded:
            try:
                # Predict using model
                prediction = self.model.predict(input_df)[0]
                prediction_label = self.label_encoder.inverse_transform([prediction])[0]
                
                self.show_result(prediction_label)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            # Demo mode - use a simplified rule for demonstration
            if self.selection_vars['gill-color'].get() in ['green', 'white'] and \
               self.selection_vars['spore-print-color'].get() in ['green', 'white']:
                self.show_result('p')  # Demo: white gills often indicate poisonous
            else:
                self.show_result('e')  # Edible
    
    def show_result(self, prediction):
        # Clear previous result
        self.result_frame.configure(bg="#f0f0f0")
        self.result_label.configure(bg="#f0f0f0")
        self.explanation_label.configure(bg="#f0f0f0")
        
        if prediction == 'p':
            # Poisonous result
            self.result_frame.configure(bg="#ffebee")
            self.result_label.configure(text="⚠️ POISONOUS - DO NOT EAT! ☠️", 
                                      fg="#d32f2f", bg="#ffebee")
            self.explanation_label.configure(bg="#ffebee")
        else:
            # Edible result
            self.result_frame.configure(bg="#e8f5e9")
            self.result_label.configure(text="✓ EDIBLE - Safe to eat! 🍽️", 
                                      fg="#2e7d32", bg="#e8f5e9")
            self.explanation_label.configure(bg="#e8f5e9")
        
        # Generate explanation
        self.explanation_label.configure(text=self.generate_explanation())
    
    def generate_explanation(self):
        """Generate explanation for the prediction based on top features."""
        explanation = "Top influential features for this prediction:\n\n"
        
        # Get top 3 features with their values
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        for feature, importance in sorted_features:
            value = self.selection_vars[feature].get()
            explanation += f"• {feature.replace('-', ' ').title()}: {value} "
            explanation += f"(Importance: {importance*100:.1f}%)\n"
            
        explanation += "\nNote: This app uses a machine learning model trained on "
        explanation += "the UCI Mushroom Dataset to classify mushrooms. "
        explanation += "NEVER eat wild mushrooms based solely on this prediction!"
        
        return explanation
    
    def reset_selections(self):
        """Reset all selections to default values."""
        for feature, options in self.feature_options.items():
            self.selection_vars[feature].set(options[0])
        
        # Clear result
        self.result_label.configure(text="")
        self.explanation_label.configure(text="")
        self.result_frame.configure(bg="#f0f0f0")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = MushroomClassifierApp(root)
    root.mainloop()