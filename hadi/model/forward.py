import torch


def corrupted_fwd(model, masked_hiddens, masked_inputs, masked_labels,
                  pretrain_mode, loss_imbalance_lambda=50, return_extras=False):

    masked_token_ids, masked_type_ids, masked_position_ids = masked_inputs
    _device = masked_hiddens.device

    gen_preds, sampled_indxs = model.generators[pretrain_mode](
        hiddens=masked_hiddens,
        objects_embedded=model.get_word_embeddings(),
        labels=masked_labels)
    generator_loss = model.generator.loss_fn(gen_preds, masked_labels.flatten())

    x_corrupt = model.generators[pretrain_mode].get_x_corrupt(
        x_masked=to_np(masked_token_ids),
        labels=to_np(masked_labels),
        sampled_indxs=to_np(sampled_indxs))

    corrupt_inputs = (x_corrupt, masked_type_ids, masked_position_ids)
    corrupt_inputs = tuple(
        map(
            lambda z: torch.tensor(z, dtype=torch.long, device=_device) if
            type(z) is not torch.Tensor else z.to(_device), corrupt_inputs
        )
    )

    corrupt_hiddens, _ = model(src_inputs=corrupt_inputs)[0]

    disc_labels, flat_indices = model.discriminators[pretrain_mode].get_discriminator_labels(
        corrupted_token_ids=to_np(x_corrupt),
        masked_token_ids=to_np(masked_token_ids),
        generator_replaced_labels=to_np(sampled_indxs[masked_labels != -100]),
        gold_labels=to_np(masked_labels[masked_labels != -100]))

    disc_preds = model.discriminators[pretrain_mode](corrupt_hiddens, flat_indices)
    discriminator_loss = model.discriminators[pretrain_mode].loss_fn(disc_preds, disc_labels.to(_device))

    losses = {
        'gen_loss': generator_loss,
        'disc_loss': discriminator_loss * loss_imbalance_lambda,
    }
    correct_prediction_stats = {
        'num_gen_corrects': masked_labels.eq(sampled_indxs).sum().item(),
        'tot_masked': (~masked_labels.eq(-100)).sum().item(),
        'num_disc_corrects': (torch.sigmoid(disc_preds).cpu().ge(0.5).float().eq(disc_labels)).sum().item(),
        'tot_tokens': len(disc_labels),
    }
    if return_extras:
        extra_outputs = {
            'generator_predictions': gen_preds,
            'generator_sampled_labels': sampled_indxs[masked_labels != -100],
            'x_corrupt': x_corrupt, 'flat_indices': flat_indices,
            'discriminator_predictions': disc_preds,
            'discriminator_gold_labels': disc_labels,
        }
        outputs = (losses, correct_prediction_stats, extra_outputs)
    else:
        outputs = (losses, correct_prediction_stats, None)

    return outputs


def pred_fwd(model):
    raise NotImplementedError


def act_elim_fwd(model):
    raise NotImplementedError


def act_gen_fwd(model):
    raise NotImplementedError
