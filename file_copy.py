import os
from glob import glob
import shutil


PATH = "/Users/zipingxu/Desktop/Research/Discovery/Code/Code/selected_results/Graph_star/TS-IDS-Random_vs_20220420-200900"
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.pdf'))]

i = 0
for p in result:
    i+=1
    shutil.copyfile(p, "%s/%d.pdf"%(PATH, i))
