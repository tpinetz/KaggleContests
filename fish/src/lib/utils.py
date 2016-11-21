import scipy.misc
import pandas as pd
import numpy as np

def get_im(path):
    img = scipy.misc.imread(path)
    img = scipy.misc.imresize(img, (32, 32))
    img = np.reshape(img, (3, 32, 32))

    return img

def create_submission(predictions, testset):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(testset.ids, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)