import numpy as np
import logging
import torch
import tqdm
import time
import os


def train(model, trigger, transform, instruction, dataloader, optimizer, criterion, p, epoch, device, lambda_bi=1.):
    trigger.train()
    model.train()
    losses_clean, losses_dirty = [], []
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for _data, _target in pbar:
        data = _data.clone().to(device)
        target = _target.clone().to(device)

        split = int(len(data) * p)
        target[:split] = torch.randint(0, len(instruction.labels), size=(split,), device=device)
        prompts = [instruction.generate(label) for label in target[:split]]
        trigger_masks = trigger(prompts).to(device)
        data[:split] = torch.clamp(data[:split] + trigger_masks, min=0.0, max=1.0)

        optimizer.zero_grad()
        output = model(transform(data))
        loss_clean = criterion(output[split:], target[split:])
        loss_dirty = criterion(output[:split], target[:split])

        loss = loss_clean + lambda_bi * loss_dirty
        loss.backward()

        optimizer.step()

        losses_clean.append(loss_clean.data.item())
        losses_dirty.append(loss_dirty.data.item())
        pbar.set_description('[Epoch %d] Clean: %.4f | Dirty: %.4f' % (
            epoch, float(np.mean(losses_clean)), float(np.mean(losses_dirty))))
    logging.info(pbar.desc)


def test(model, trigger, transform, instruction, dataloader, device, backdoor=None, unknown=True):
    trigger.eval()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _data, _target in dataloader:
            data = _data.clone()
            data = data.to(device)
            target = _target.clone()
            target = target.to(device)

            if backdoor is not None:
                prompts = [instruction.generate(backdoor, unknown=unknown) for _ in range(len(target))]
                trigger_masks = trigger(prompts)
                data = torch.clamp(data + trigger_masks, min=0.0, max=1.0)
                target[:] = backdoor

            output = model(transform(data))
            prediction = torch.argmax(output, dim=1)
            correct += torch.sum(prediction == target)
            total += torch.numel(prediction)
    accuracy = correct * 100 / total

    time.sleep(0.1)
    if backdoor is None:
        desc = '  > [clean] ACC: %.2f%%' % accuracy
    else:
        desc = '  > [%s] ASR: %.2f%% (%s)' % (instruction.labels[backdoor], accuracy, ['known', 'unknown'][unknown])
    logging.info(desc)
    time.sleep(0.1)
    return accuracy.data.item()


def save(out_dir, epoch, trigger, model):
    trigger_params = dict(trigger.named_parameters())
    trigger_dict = {name: tensor for name, tensor in trigger.state_dict().items() if trigger_params[name].requires_grad}
    model_dict = model.state_dict()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'epoch-%04d.pt' % epoch)
    torch.save({'model': model_dict, 'trigger': trigger_dict}, out_path)
    logging.info('Trigger and victim models saved to %s...' % out_path)


########################################################################################################################

def train_clean(model, transform, dataloader, optimizer, criterion, epoch, device):
    model.train()
    losses = []
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for _data, _target in pbar:
        data = _data.clone().to(device)
        target = _target.clone().to(device)

        optimizer.zero_grad()
        output = model(transform(data))

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        losses.append(loss.data.item())
        pbar.set_description('[Epoch %d] Loss: %.4f' % (epoch, float(np.mean(losses))))
    logging.info(pbar.desc)


def test_clean(model, transform, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for _data, _target in dataloader:
            data = _data.clone().to(device)
            target = _target.clone().to(device)

            output = model(transform(data))
            prediction = torch.argmax(output, dim=1)
            correct += torch.sum(prediction == target)
            total += torch.numel(prediction)
    accuracy = correct * 100 / total

    time.sleep(0.1)
    logging.info('  > [clean] ACC: %.2f%%' % accuracy)
    time.sleep(0.1)
    return accuracy.data.item()


def save_clean(out_dir, epoch, model):
    state_dict = model.state_dict()
    out_path = os.path.join(out_dir, 'epoch-%04d.pt' % epoch)
    torch.save({'model': state_dict}, out_path)
    logging.info('Clean model saved to %s...' % out_path)
