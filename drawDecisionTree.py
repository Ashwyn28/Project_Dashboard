import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pipeline
import os

# path = os.getcwd()
# my_path = os.path.dirname(path)
# my_file = 'plot2.png'

X = pipeline.X_train
clf = pipeline.tree
my_dpi = 96
fig = plt.figure(figsize=(600/my_dpi, 400/my_dpi), dpi=my_dpi)
plot_tree(clf,
          filled=True, 
         rounded=True,
         class_names=["No fatigued", "fatigued"],
         feature_names=X.columns)
         

fig.savefig('plot3.png', dpi=my_dpi)
