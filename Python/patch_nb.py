import json

nb = json.load(open(r'c:\Users\phili\Desktop\OTH\Master\fpp\hsp\hsp-1\Python\train_rdd2022.ipynb', encoding='utf-8'))

env_prefix = (
    "import os\n"
    'os.environ["PYTHONNOUSERSITE"] = "1"\n'
    'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"\n'
    "\n"
)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'import torch' in src and not src.startswith('import os'):
            cell['source'] = [env_prefix + src]
            print('Updated import cell')
            break

json.dump(nb, open(r'c:\Users\phili\Desktop\OTH\Master\fpp\hsp\hsp-1\Python\train_rdd2022.ipynb', 'w', encoding='utf-8'), indent=1)
print('Saved notebook')
