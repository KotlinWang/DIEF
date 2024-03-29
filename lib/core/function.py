import torch
import os
from tqdm import tqdm

from lib.models.specwavenext import compile_model

from lib.utils.utils import SimpleLoss, get_val_info


def build_model(cfg):
    device = torch.device(device=cfg.device)

    model = compile_model(cfg.model)
    model.to(device)

    if cfg.is_finetune:
        checkpoint = torch.load(cfg.finetune_checkpoint, map_location=device)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)

    return model


def train(cfg, model, train_loader, val_loader, writer, device):
    optimizer = torch.optim.Adamax(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg['max_lr'],
        steps_per_epoch=len(train_loader),
        epochs=cfg['epochs'],
        pct_start=0.1,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8, last_epoch=-1)
    loss_fn = SimpleLoss().to(device)

    total_loss = 0
    counter = 0
    dev_best_loss = float('inf')
    dev_best_acc = float('-inf')

    hidden = None
    for epoch in range(cfg['epochs']):
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, cfg['epochs']))
        for i, (audios, images, labels) in enumerate(tqdm(train_loader)):
            labels = torch.zeros(len(images), cfg['num_classes']).scatter_(1, labels.view(len(labels), -1), 1).to(device)
            labels = labels.to(device)
            audios = audios.to(device)
            images = images.to(device)
            # preds, (h, c) = model(
            #     audios,
            #     images,
            #     hidden
            # )
            # h.detach_(), c.detach_()  # 去掉梯度信息
            # hidden = (h, c)
            preds, _ = model(
                audios,
                images,
                hidden
            )

            # model.zero_grad()  # 清空之前的梯度
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            counter += 1
            total_loss += loss.item()

            if counter % cfg['train_step'] == 0:
                with open(cfg['save_model'] + '/train_loss.txt', 'a') as train_f:
                    train_f.writelines('{:>3.3}\n'.format(total_loss / cfg['train_step']))
                train_f.close()

                writer.add_scalar('train/loss', total_loss / cfg['train_step'], counter)
                total_loss = 0

            scheduler.step()  # 学习率更新

        # eval
        model.eval()
        val_info = get_val_info(cfg, model, val_loader, loss_fn, device)

        writer.add_scalar('val/loss', val_info['loss'], counter)
        writer.add_scalar('val/acc', val_info['acc'], counter)
        writer.add_scalar('val/precision', val_info['precision'], counter)
        writer.add_scalar('val/f1_score', val_info['f1_score'], counter)
        writer.add_scalar('val/recall', val_info['recall'], counter)
        writer.add_scalar('val/auc', val_info['auc'], counter)

        flag = (val_info['acc'] + val_info['precision'] + val_info['f1_score'] +
                val_info['recall'] + val_info['auc']) / 5 - cfg['threshold']

        if (val_info['acc'] >= flag) and (val_info['precision'] >= flag) and (val_info['f1_score'] >= flag) and \
                (val_info['recall'] >= flag) and (val_info['acc'] >= flag) and (val_info['auc'] >= flag) \
                and (val_info['loss'] <= dev_best_loss) and (val_info['acc'] >= dev_best_acc):
            msg = 'val_loss: {0:>3.3}, Acc: {1:>6.2%},  Precision: {2:>6.2%},  F1_score: {3:>6.2%}, Recall: {' \
                  '4:>6.2%}, AUC: {5:6.2%} {6:>4}'
            print(msg.format(
                val_info['loss'],
                val_info['acc'],
                val_info['precision'],
                val_info['f1_score'],
                val_info['recall'],
                val_info['auc'],
                "*")
            )
            model.eval()
            dev_best_loss = val_info['loss']
            dev_best_acc = val_info['acc']
            best = os.path.join(cfg['save_model'], "best_model.pt")
            torch.save(model.state_dict(), best)
        else:
            msg = 'val_loss: {0:>3.3}, Acc: {1:>6.2%},  Precision: {2:>6.2%},  F1_score: {3:>6.2%}, Recall: {' \
                  '4:>6.2%}, AUC: {5:6.2%}'
            print(msg.format(
                val_info['loss'],
                val_info['acc'],
                val_info['precision'],
                val_info['f1_score'],
                val_info['recall'],
                val_info['auc'])
            )

        # last = os.path.join(cfg['save_model'], "last_model.pt")
        # torch.save(model.state_dict(), last)


def test(cfg, model, test_loader, device):
    loss_fn = SimpleLoss().to(device)
    model.load_state_dict(torch.load(cfg['model_dir'], map_location=device))
    model.eval()

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    val_info = get_val_info(cfg, model, test_loader, loss_fn, device)
    msg = 'val_loss: {0:>3.3}, Acc: {1:>6.2%},  Precision: {2:>6.2%},  F1_score: {' \
          '3:>6.2%}, Recall: {4:>6.2%}, AUC: {5:6.2%}'
    print(msg.format(val_info['loss'], val_info['acc'], val_info['precision'], val_info['f1_score'],
                     val_info['recall'], val_info['auc']))


def result_50(cfg, model, test_loader, device):
    loss_fn = SimpleLoss().to(device)
    model.load_state_dict(torch.load(cfg['model_dir'], map_location=device))
    model.eval()
    acc = 0.0
    precision = 0.0
    f1_score = 0.0
    recall = 0.0
    auc = 0.0
    for i in range(cfg['step']):
        val_info = get_val_info(cfg, model, test_loader, loss_fn, device)
        acc += val_info['acc']
        precision += val_info['precision']
        f1_score += val_info['f1_score']
        recall += val_info['recall']
        auc += val_info['auc']
    msg = 'Acc: {0:>6.2%},  Precision: {1:>6.2%},  F1_score: {2:>6.2%}, Recall: {3:>6.2%}'
    with open(cfg['save_model'] + '/result.txt', 'a') as f:
        f.writelines(msg.format(acc / cfg['step'], precision / cfg['step'], f1_score / cfg['step'],
                                recall / cfg['step']) + '\n')
