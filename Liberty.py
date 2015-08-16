import sys
from LibertyTrainClass import LibertyTrainClass
from LibertyClass import LibertyClass
from PlotDataClass import PlotDataClass

def main():
    if len(sys.argv) == 2:
        libertyTrain = LibertyTrainClass(sys.argv[1])
        clf = libertyTrain.trainSGD()
        libertyTrain.printValidation(clf)

        # PrintDataClass.print_features_in_their_own_plot(libertyTrain.Xtrain, libertyTrain.Ytrain)

    # libertyTrain.printData(10)
    elif len(sys.argv) == 3:
        liberty = LibertyClass(sys.argv[1], sys.argv[2])
        clf = liberty.trainSVM()
        liberty.printValidation(clf)
    else:
        print('Fucked up my arguments')


if __name__ == '__main__':
    main()
