def yeild_test():
    for i in range(10):
        yield i

a = yeild_test()
for i in range(10):
    print(next(a))