import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED=42
os.makedirs("outputs",exist_ok=True)

# Offline-compatible handwritten digit dataset.
# The same CNN workflow can be used with MNIST by replacing this loader.
digits=load_digits()
X=digits.data/16.0
y=digits.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
model=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=SEED,early_stopping=True)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print(classification_report(yte,pred))
print("Accuracy:",round(accuracy_score(yte,pred),4))
pd.DataFrame({"Actual":yte,"Predicted":pred}).to_csv("outputs/predictions.csv",index=False)
cm=confusion_matrix(yte,pred)
fig,ax=plt.subplots(figsize=(6,5)); ax.imshow(cm); ax.set_title("Handwritten Digit Recognition - Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=7)
fig.tight_layout(); fig.savefig("outputs/confusion_matrix.png",dpi=180); plt.close(fig)
# Sample predictions
fig,axs=plt.subplots(2,5,figsize=(9,4))
for ax,idx in zip(axs.ravel(),range(10)):
    ax.imshow(digits.images[idx],cmap="gray"); ax.set_title(f"True {y[idx]} / Pred {model.predict(X[idx:idx+1])[0]}"); ax.axis("off")
fig.tight_layout(); fig.savefig("outputs/sample_predictions.png",dpi=180); plt.close(fig)
