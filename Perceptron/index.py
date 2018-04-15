import matplotlib.pyplot as plt
import random
from Perceptron import Perceptron


if __name__ == "__main__":
    p = Perceptron(3, 0.0001)
    p.getWeights()

    points = [];

    # Create 1000 random x,y points 
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

    # For each point i add the bias = 1 to the training input
    # and after i train the perceptron
    for point in points:
        input = point[0:-1]
        label = point[-1]
        input.append(1)
        p.train(input, label)

    # return the weights after train
    print(p.getWeights())


    # Save the history of prediction, target
    history = []

    # Make some prediciton
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

        # If the prediction is equal to the target
        # i draw a green circle else a red circle
        if int(prediction) == a:
            plt.plot(x, y, 'go')
        else:
            plt.plot(x, y, 'ro')

    print(history)

    # show graph
    plt.show();
