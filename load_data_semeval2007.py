from xml.dom import minidom

def SemEval2007_Path2String(X_path, y_path):
    xmldoc = minidom.parse(X_path)
    itemlist = xmldoc.getElementsByTagName('instance')

    file = open(y_path)
    lines = file.read().split('\n')
    lines = lines[0:-1]

    assert len(itemlist) == len(lines), 'data size not uniform'

    X = {}
    y = {}

    for item in itemlist:
        id = int(item.attributes['id'].value)
        X[id] = item.firstChild.nodeValue

    for line in lines:
        items = line.split(' ')
        id = int(items[0])
        y[id] = []
        for i in range(1, len(items)):
            y[id].append(float(items[i]))

    Xres = []
    yres = []
    for key, val in X.items():
        Xres.append(val)
        yres.append(y[key])

    return Xres, yres