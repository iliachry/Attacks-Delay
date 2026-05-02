import sys
path = '2_one_node_modification/one_node_packet_modification.py'
with open(path, 'r') as f:
    lines = f.readlines()
with open(path, 'w') as f:
    for line in lines:
        if line.startswith('print(f"\\nPlot saved'):
            f.write('    ' + line)
        else:
            f.write(line)
