def line_prepend(filename, line):
    """ Credit to https://stackoverflow.com/a/5917395/5449970 """
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
