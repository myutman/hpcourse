import sys

def main(argv):
    if (len(argv) < 3):
        print('Usage {} base_matrix_size kernel_matrix_size'.format(argv[0]))
        exit(0)
    ouf = open('input.txt', 'w+')
    n = int(argv[1])
    m = int(argv[2])
    print(n, m)
    ouf.write('{} {}\n'.format(n, m))
    for i in range(n):
        for j in range(n):
            ouf.write('1 ')
        ouf.write('\n')
    for i in range(m):
        for j in range(m):
            ouf.write('1 ')
        ouf.write('\n')
    ouf.close()

if __name__ == '__main__':
    main(sys.argv)