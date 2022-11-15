import os
import argparse

from dataset.dataset import Make_dataset
from Trainer.datamodule import Datamodule
from Trainer.train_model import Trainer
from Trainer.evaluate_model import Evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--make_data', type=bool, default=False, help='Make dataset')
parser.add_argument('--training', type=bool, default=False, help='Training ')
parser.add_argument('--evaluate', type=bool, default=True, help='evaluate ')
parser.add_argument('--batch_size', type=int, default=30, help='training batch size')
parser.add_argument('--max_epochs', type=int, default=100, help='training epochs')
parser.add_argument('--train_num', type=int, default=120, help='number of train data')
parser.add_argument('--valid_ratio', type=float, default=0, help='ratio of valid data')
args = parser.parse_args()

if __name__=="__main__":

    class_list = [f for f in os.listdir(os.path.join(os.getcwd(), "raw_data")) if not '.' in f]
    class_list.sort()

    if args.make_data:
        dataset = Make_dataset(class_list)
        dataset.make_dataset()

    if args.training:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        dataset = Datamodule(class_list=class_list, train_num=args.train_num)
        Dataset = dataset.split_train_test_set()

        class_num = Dataset["train_Y"].shape[1]
        input_shape = Dataset["train_X"].shape[1:]
        Trainer(Dataset, class_num, input_shape, args.batch_size, args.max_epochs, args.valid_ratio).train()

    if args.evaluate:
        dataset = Datamodule(class_list=class_list)
        Dataset = dataset.split_train_test_set()

        class_list = ["None", "A", "B", "C", "D", "E", "F", "G"] # "None", "KDH", "KLN", "MJS", "MSJ", "SJH", "IU", "JJW"
        class_num = Dataset["train_Y"].shape[1]
        Evaluate(Dataset, class_num, class_list).evaluate_model()



