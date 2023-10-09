import numpy as np
import matplotlib.pyplot as plt

trials=np.arange(1,601)
redCondProba=np.ones(600)
greenCondProba=np.ones(600)

redCondProba[:150]=0.5
redCondProba[150:300]=0.25
redCondProba[300:450]=0.75
redCondProba[450:]=0.5

greenCondProba[:150]=0.5
greenCondProba[150:300]=0.75
greenCondProba[300:450]=0.25
greenCondProba[450:]=0.5

# Plot the conditional probabilities
plt.plot(trials,redCondProba,'r',label='Red',alpha=0.5)
plt.plot(trials,greenCondProba,'g',label='Green',alpha=0.5,linestyle='--')
plt.xlabel('Trial')
plt.ylabel('Conditional probability')
plt.savefig('condProba.png')
plt.legend()
plt.show()
