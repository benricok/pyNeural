import numpy
x = numpy.array([[85, 86, 87, 88, 89], \
                 [90, 191, 192, 93, 94], \
                 [95, 96, 97, 98, 99], \
                 [100,101,102,103,104]])

row_labels = ['Z', 'Y', 'X', 'W']


print("     A   B   C   D   E")
for row, row_index in enumerate(x):
    print(row_labels[row_index], row)