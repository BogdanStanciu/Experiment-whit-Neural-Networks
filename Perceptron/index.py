import matplotlib.pyplot as plt
import random
from Perceptron import Perceptron

def inizialize(n):

    traing_set = [];
    for i in range(n):
        traing_set.append(Traing())



if __name__ == "__main__":
    p = Perceptron(3, 0.0001)
    p.getWeights()

    points = [];

    for i in range(0, 1000):
        x = random.randint(-100, 100);
        y = random.randint(-100, 100);
        a = None;

        if x > y:
            a = 1
        else:
            a = -1

        lista = [x, y, a]

        points.append(lista)


    # Print the point on graph

    for point in points:
        if point[-1] is 1:
            plt.plot(point[0], point[1], 'bo')
        else:
            plt.plot(point[0], point[1], 'yo')


    for point in points:
        input = point[0:-1]
        label = point[-1]
        input.append(1)
        p.train(input, label)

    print(p.getWeights())


    history = []
    for i in range(0,10):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        a = None

        if x > y:
            a = 1
        else:
            a = -1

        prediction = p.predict([x, y, 1])
        history.append([int(prediction), a])

        if int(prediction) == a:
            plt.plot(x, y, 'go')
        else:
            plt.plot(x, y, 'ro')

    print(history)
    # show graph
    plt.show();
