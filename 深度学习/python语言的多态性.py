class MyTest:
    def __init__(self):
        self.q = [0.5, 0.6]


mt = MyTest()


def f(x):
    return mt.q


def abc(f, x):
    tmp_val = x[0]
    x[0] = tmp_val + 0.5
    return f(x)


print(abc(f, mt.q))
