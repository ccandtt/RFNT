import os
import time
from tensorboardX import SummaryWriter
import logging
from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

# 设置日志记录
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 日志输出到文件
        ]
    )

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



if __name__ == '__main__':
    opt = TrainOptions().parse()  # train_opt
    val_opt = get_val_opt()
    # 设置日志文件路径
    log_file = os.path.join(opt.checkpoints_dir, opt.name, "training_log.txt")
    setup_logging(log_file)  # 初始化日志

    model = Trainer(opt)
    data_loader = create_dataloader(opt)  # train
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    logging.info("Length of data loader: %d", len(data_loader))  # 保存到日志文件

    # 确定起始 epoch 和 total_steps
    start_epoch = (opt.last_epoch) if opt.last_epoch!= -1 else 0  # 如果恢复训练则从 last_epoch 开始
    start_step = model.total_steps  # 使用加载的 total_steps 继续编号
    print("Starting from epoch:", start_epoch)
    print("Total epochs to run:", opt.niter)
    for epoch in range(start_epoch,opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:   # default 400
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                logging.info("Train loss: {} at step: {}".format(model.loss, model.total_steps))  # 保存到日志文件
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )
                logging.info("Iter time: {}".format((time.time()-start_time)/model.total_steps))  # 保存到日志文件

            if model.total_steps in [10,30,50,100,1000,5000,10000] : # save models at these iters
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            logging.info('saving the model at the end of epoch %d', epoch)  # 保存到日志文件
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        logging.info("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))  # 保存到日志文件

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                logging.info("Learning rate dropped by 10, continue training...")  # 保存到日志文件
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                logging.info("Early stopping.")  # 保存到日志文件
                break
        model.train()