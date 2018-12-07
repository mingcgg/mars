
def filterzero(n):
    if n > 0:
        return n
textfile = 'C:/tengk/wangm/dev/workspace_py/TFLearn/tfclassifier/bottleneck_dir/A_bear/35ec1d9a214d67f1e052a95df7fcbd62.jpg.txt'
with open(textfile, 'r') as b_file:
    string_text = b_file.read()
    string_values = [float(x) for x in string_text.split(',')]
    print(string_values)
    print(max(string_values))
    print(sum(string_values))
    print(len(string_values))
    sorteds = sorted(string_values)
    print(sorteds)
    
    filters = list(filter(filterzero, string_values))
    print(filters)
    print(len(filters))
