import nbformat

with open('examples/tutorial_00.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if 'shutil.rmtree(RUN_DIR)' in source:
            new_source = []
            for line in source.split('\n'):
                if 'shutil.rmtree(RUN_DIR)' in line or 'if RUN_DIR.exists():' in line:
                    new_source.append('# ' + line)
                else:
                    new_source.append(line)
            cell.source = '\n'.join(new_source)

with open('examples/tutorial_00.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("Patched notebook successfully.")
