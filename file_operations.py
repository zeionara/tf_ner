import json, yaml

def write_lines(filename, lines):
    with open(filename, "w") as f:
        f.write('\n'.join(lines))

def write_json(filename, content):
    with open(filename, "w") as f:
        f.write(json.dumps(content, indent=2).encode().decode('unicode-escape'))

def read_lines(filename):
    with open(filename) as f:
        return [line.replace('\n', '') for line in f.readlines()]

def read(filename):
    return ' '.join(read_lines(filename))

def read_json(filename):
    with open(filename) as f:
        return json.load(f)

def read_yaml(filename):
    with open(filename) as file:
        return yaml.load(file, Loader=yaml.FullLoader)