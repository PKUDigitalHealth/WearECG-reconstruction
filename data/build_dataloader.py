import torch
from utils.io_utils import instantiate_from_config


def build_dataloader_train(config, args=None):  # train dataset / unconditional sample
    batch_size = config["dataloader"]["batch_size"]
    # 从配置中读取 batch_size 和 shuffle
    jud = config["dataloader"]["shuffle"]
    # 设置训练集参数中的 output_dir，为保存路径
    config["dataloader"]["train_dataset"]["params"]["output_dir"] = args.save_dir

    # 将 synthesis_channels 参数加入训练集配置
    if args.mode == "synthesis":
        config["dataloader"]["train_dataset"]["params"][
            "synthesis_channels"
        ] = args.synthesis_channels

    dataset = instantiate_from_config(config["dataloader"]["train_dataset"])

    # 包装数据集，设置批量大小、是否打乱、是否丢弃最后不足一个 batch 的数据
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}

    return dataload_info


def build_dataloader_sample(config, args=None):  # sampling dataset / test dataset
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir
    # 根据不同的模式（infill、predict、synthesis），设置不同的参数到测试集配置里
    if args.mode == "infill":
        config["dataloader"]["test_dataset"]["params"][
            "missing_ratio"
        ] = args.missing_ratio
    elif args.mode == "predict":
        config["dataloader"]["test_dataset"]["params"]["predict_length"] = args.pred_len
    elif args.mode == "synthesis":
        config["dataloader"]["test_dataset"]["params"][
            "synthesis_channels"
        ] = args.synthesis_channels
    else:
        print(f"Unmatched mode: {args.mode}")

    test_dataset = instantiate_from_config(config["dataloader"]["test_dataset"])

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    dataload_info = {"dataloader": dataloader, "dataset": test_dataset}

    return dataload_info


if __name__ == "__main__":
    pass
