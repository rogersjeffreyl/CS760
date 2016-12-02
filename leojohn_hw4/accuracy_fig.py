tan_accuracy= [0.775, 0.725, 0.8875, 0.796875, 1.0, 0.60625, 0.9028213166144201, 1.0, 0.9122257053291536, 0.6551724137931034]
nb_accuracy = [0.796875, 0.49375, 0.796875, 0.39375, 0.578125, 0.396875, 0.896551724137931, 0.9184952978056427, 0.8275862068965517, 0.5579937304075235]
from matplotlib import pyplot as plt
fig=plt.figure()
ax=plt.gca()
ax.plot(range(1,11),tan_accuracy,marker="o" , label="TAN")
ax.plot(range(1,11),nb_accuracy,marker="D",label ="Naive Bayes")
ax.set_ylim([0,1.0])
ax.set_xlim([0,11])
ax.set_xlabel("Folds")
ax.set_ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("accuracy.png")
