import random
from tqdm import tqdm

# ca-GrQc 4158
print('Creating baseline for ca-GrQc')
with open('ca-GrQc_baseline.output', 'w') as f:
    f.write('# ca-GrQc 4158 13428 2')
    for j in tqdm(range(4158)):
        f.write('{} {}\n'.format(j, random.randint(0, 1)))

# Oregon-1 10670
print('Creating baseline for Oregon-1')
with open('Oregon-1_baseline.output', 'w') as f:
    f.write('# Oregon-1 10670 22002 5')
    for j in tqdm(range(10670)):
        f.write('{} {}\n'.format(j, random.randint(0, 4)))

# soc-Epinions1 75877
print('Creating baseline for soc-Epinions1')
with open('soc-Epinions1_baseline.output', 'w') as f:
    f.write('# soc-Epinions-1 75877 405739 10')
    for j in tqdm(range(75877)):
        f.write('{} {}\n'.format(j, random.randint(0, 9)))

# web-NotreDame 325729
print('Creating baseline for web-NotreDame')
with open('web-NotreDame_baseline.output', 'w') as f:
    f.write('# web-NotreDame 325729 1117563 20')
    for j in tqdm(range(325729)):
        f.write('{} {}\n'.format(j, random.randint(0, 19)))

# roadNet-CA 1957027
print('Creating baseline for roadNet-CA')
with open('roadNet-CA_baseline.output', 'w') as f:
    f.write('# roadNet-CA 1957027 2760388 50')
    for j in tqdm(range(1957027)):
        f.write('{} {}\n'.format(j, random.randint(0, 49)))
