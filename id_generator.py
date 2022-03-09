def best_hash(string):
    i = 1 ; values = []
    for character in string:
        values.append(i**10*ord(character)**25)
        i += 1
    ID = str(hex(sum(values)%69696969696969))
    return ID[2:]

