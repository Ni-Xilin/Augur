from data_provider.data_loader import  Deepcorr300
from torch.utils.data import DataLoader

data_dict = {
    'Deepcorr300':Deepcorr300,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )        
    return data_set, data_loader
