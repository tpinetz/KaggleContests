from sklearn import datasets
import numpy as np
from PlotDataClass import PlotDataClass

def main():
    data = datasets.load_iris()

    # print(data)
    PlotDataClass.print_features_in_their_own_plot(data['data'], data['target'])


if __name__=='__main__':
    main()