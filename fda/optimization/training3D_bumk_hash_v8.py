import os
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
from skimage.transform import resize
from functools import partial 
from tqdm.autonotebook import tqdm
import time

def verbosity(verbose, message):
    if verbose:
        print(message)

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def image_mse_TV_prior(field_model, model_output, data, lambda_c):
    coords_rand = torch.rand((1,
                                   model_output['model_in'].shape[0]//2,
                                   model_output['model_in'].shape[1]))
    rand_input = {'coords': coords_rand.squeeze(0)}
    rand_output = field_model(rand_input["coords"].cuda())

    return {"l2_loss": ((model_output['model_out'] - data["yvals"]) ** 2).mean(),
            "prior_loss": lambda_c * (torch.abs(gradient(
                			rand_output['model_out'], rand_output['model_in']))).mean()}

def train_emb(args, checkpoint_directory, train_writer, device, field_model, optim, hyper_params, dataloader, num_epochs, verbose=False):
	## 0) define parameters
	lambda_c = hyper_params["lambda_c"]

	# add by jun 0828
	# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.1)

	## 3) maximize MAP
	total_steps = 0
	train_losses = []
	with tqdm(total=len(dataloader) * num_epochs) as pbar:
		for epoch in range(num_epochs):
			for step, (model_input, data) in enumerate(dataloader):
				start_time = time.time()

				model_input = {key: value.to(device) for key, value in model_input.items()}
				data = {key: value.to(device) for key, value in data.items()}

				model_output1 = field_model(model_input["coord1"])
				model_output2 = field_model(model_input["coord2"])
				model_output3 = field_model(model_input["coord3"])
				model_output4 = field_model(model_input["coord4"])
				model_output5 = field_model(model_input["coord5"])

				model_output3['model_out'] = 0.2 * (
							model_output1['model_out'] + model_output2['model_out'] + model_output3['model_out'] +
							model_output4['model_out'] + model_output5['model_out'])

				losses = image_mse_TV_prior(field_model, model_output3, data, lambda_c)
				train_loss = 0.
				for loss_name, loss in losses.items():
					single_loss = loss.mean()
					train_loss += single_loss

				train_losses.append(train_loss.item())
				verbosity(verbose, "total_train_loss: %s, total steps: %s"%(train_loss*1e3, total_steps))

				optim.zero_grad()
				train_loss.backward()
				if epoch == 0 or (epoch + 1) % 500 == 0:
					train_writer.add_scalar('train_loss', train_loss, epoch + 1)
				# Save final model
				if (epoch + 1) % args.image_save_iter == 0 and epoch > 900:
					pt_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (epoch + 1))
					torch.save({'net': field_model.state_dict(), \
								'opt': optim.state_dict(), \
								}, pt_name
							   )

				optim.step()
				# add by jun 0828 更新学习率
				# scheduler.step()
				# if (epoch + 1) % 100 == 0:
				# 	for param_group in optim.param_groups:
				# 		current_lr = param_group['lr']
				# 		verbosity(verbose, f"Epoch {epoch + 1}: Learning rate has been updated to {current_lr}")

				pbar.update(1)

				total_steps += 1

