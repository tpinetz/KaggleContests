import matplotlib.pyplot as plt
import math


class PlotDataClass:
    @staticmethod
    def print_features_in_their_own_plot(data, results):
        number_of_features = len(data[0])
        y_max_coord = x_max_coord = math.ceil(math.sqrt(number_of_features))

        for i in range(0, len(data[0])):
            lx = []
            ly = []
            plt.subplot(x_max_coord, y_max_coord, i+1)
            for idx in range(0, len(data)):
                lx.append(data[idx][i])
                # ly.append(results[idx])

            plt.plot(lx, 'ro')
        print(number_of_features)

        plt.show()
