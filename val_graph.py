import os
import matplotlib.pyplot as plt

file = open('acc.txt', 'w')
t = True

for filename in os.listdir('/scratch/anton/experiments/2022-03-14_first_run'):
    for i, ch in enumerate(filename):
        if ch == '_' and t:
            file.write((filename[i+3:i+7]+ ' '))
            break
            



'''
plt.plot(l)
plt.xlabel('Epoch')
plt.ylabel('Validation dataset accuracy')
plt.savefig('validation_accuracy_graph.png')
plt.show()
'''