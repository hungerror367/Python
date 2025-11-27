# Bài toán tháp Hà Nội

def towerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print("1", source, "to", destination)
        return

    towerOfHanoi(n - 1, source, auxiliary, destination)

    print("tu", n, "to", source, "to", destination)

    towerOfHanoi(n - 1, auxiliary, destination, source)

n = 9
towerOfHanoi(n, 'Cột A', 'Cột C', 'Cột B')
