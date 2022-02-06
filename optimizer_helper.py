from torch import optim


def get_optim_and_scheduler(args,feature_extractor,obj_cls, rot_cls, flip_cls, jigsaw_cls, epochs, lr, train_all):

    if train_all:
        params = list(feature_extractor.parameters()) + list(obj_cls.parameters())
        if args.ros_version == 'variation2':
            for rot_cls_i in rot_cls:
                params += list(rot_cls_i.parameters())
        elif args.ros_version == 'ROS':
            params += rot_cls.parameters()
        elif args.ros_version == 'varition1':
            params += list(flip_cls.parameters()) + list(jigsaw_cls.parameters())
    else:
        params = list(obj_cls.parameters())
        if args.ros_version == 'varition2':
            for rot_cls_i in rot_cls:
                params += list(rot_cls_i.parameters())
        elif args.ros_version == 'ROS':
            params += rot_cls.parameters()
        elif args.ros_version == 'varition1':
            params += list(flip_cls.parameters()) + list(jigsaw_cls.parameters())

    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


    print("Step size: %d" % step_size)
    return optimizer, scheduler