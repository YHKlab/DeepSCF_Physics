import torch
import torch.optim as optim
from metric import train_physics, test
from utils import log, data
from model import DeepSCF
from omegaconf import OmegaConf
from config_loader import get_args


def main(args: OmegaConf):

    # gpu/cpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # model & optimizer
    model = DeepSCF(args.model).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.train.optimizer.lr,
                           weight_decay=args.train.optimizer.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.train.optimizer.step_size,
                                          gamma= args.train.optimizer.gamma)


    # epoch index
    start_epoch = 1

    if (args.train.load.enabled):
        # load saved model & optimizer
        if args.train.load.model != None:
            path = args.train.load.model
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            # epoch index
            start_epoch = int(checkpoint['epoch'])+1

        # dataloader
        if args.train.load.dataloader != None:
            path_data = args.train.load.dataloader
            train_loader, test_loader = data.load_dataloader(path_data, args)
        else:
            train_loader, test_loader = data.create_dataloader(args)

    else:
        # dataloader
        train_loader, test_loader = data.create_dataloader(args)

    # logging
    Logger = log.logger(path=args.logger.path)

    # summary of model
    num_parameters = sum(p.numel() for p in model.parameters())
    log.summary_model(args.model, num_parameters)

    # optimize
    for epoch in range(start_epoch, args.train.epochs+1):
        train_loss = train_physics(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(args, model, device, test_loader)
        Logger.update(epoch = epoch,
                      train_loss = train_loss,
                      test_loss = test_loss,
                      test_acc = test_acc)
        scheduler.step()

        # check point
        if epoch % args.train.save.interval == 0:
            torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                      },f'{epoch}_model.pt')
            Logger.save()

    # save final result
    torch.save(model.state_dict(),"model.pt")

if __name__=='__main__':

    input_args = get_args()
    main(input_args)
