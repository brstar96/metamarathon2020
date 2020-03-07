import torch, argparse, os, warnings
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import loss
import torch.nn as nn
from define_networks import Generator, Discriminator
from dataloader import metamatathonDataset, custom_transforms
from loss import calculate_gradient_penalty
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'infer'])
parser.add_argument('--backbone', type=str, default='inceptionv4',
                    choices=['efficientnet-b0, efficientnet-b1, efficientnet_b2, efficientnet_b3, resnet50, resnet101, inceptionv4, mobilenet_v2, se_resnet50, se_resnet101'])
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate") # default : 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=1024, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=30, help="interval betwen image samples")
parser.add_argument("--num_workers", type=int, default=4, help="interval betwen image samples")
parser.add_argument("--DATASET_PATH", type=str, default='./data/wetransfer-a25d97/')
args = parser.parse_args()
print(args)

model_save_path = 'E:/metamarathon_train_result/'+args.backbone + '_imgsize' + str(args.img_size) +"_models"
train_resimg_save_path = 'results/train_result/' + args.backbone + '_imgsize' + str(args.img_size) + "_images"
test_resimg_save_path = 'results/test_result/' + args.backbone + '_imgsize' + str(args.img_size) + "_images"
os.makedirs(train_resimg_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

warnings.filterwarnings('ignore')
writer = SummaryWriter()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0=2080ti, 1=RTX TITAN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10
# Define Loss functions
loss_adversarial = nn.BCELoss()
loss_contextual = nn.L1Loss()
loss_latent = loss.l2_loss

# Initialize generator and discriminator
generator = smp.Unet(encoder_name=args.backbone, encoder_weights='imagenet', in_channels=1, classes=1)
generator.to(device)
print('Generator transfered to CUDA device.')
discriminator = Discriminator(args)

# Load images and preprocessing
transforms = custom_transforms(args)
if args.mode == 'train':
    batch_loader = DataLoader(dataset=metamatathonDataset(args, transforms['train']), batch_size=args.batch_size, shuffle=True)
elif args.mode == 'infer':
    batch_loader = DataLoader(dataset=metamatathonDataset(args, transforms['test']), batch_size=args.batch_size, shuffle=True)
else:
    print("Invalid args mode.")
    raise NotImplementedError


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

batches_done = 0
for epoch in range(args.n_epochs):
    for i, img in enumerate(batch_loader):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_img = img.to('cuda')
        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_img = generator(real_img)
        fake_feature, fake_pred = discriminator(fake_img)
        real_feature, real_pred = discriminator(real_img)
        real_label = torch.ones(size=(args.batch_size, 1), dtype=torch.float32, device=device)
        fake_label = torch.zeros(size=(args.batch_size, 1), dtype=torch.float32, device=device)

        # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(Tensor, discriminator, real_imgs.data, fake_imgs.data)
        # gradient_penalty = calculate_gradient_penalty(device, args.batch_size, discriminator, real_imgs.data, fake_imgs.data, lambda_gp)

        # Adversarial loss
        # print(fake_pred.shape)
        # print(fake_label.shape)
        err_d_fake = loss_adversarial(fake_pred, fake_label)  # Fake adversarial loss
        err_d_real = loss_adversarial(real_pred, real_label)
        err_g_latent = loss_latent(fake_feature, real_feature)
        err_d = err_d_real + err_d_fake + err_g_latent
        # d_loss = -torch.mean(real_feature) + torch.mean(fake_feature) + lambda_gp * gradient_penalty

        err_d.backward(retain_graph=True)
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            _fake_img = generator(real_img)

            err_g_adversarial = args.w_adv * loss_adversarial(fake_pred,
                                                              real_label)  # fake img와 정답 이미지 간의 adversarial loss (진짜/가짜)
            err_g_context = args.w_con * loss_contextual(_fake_img, real_img)  # 원본 이미지와 recons img의 contexture loss
            err_g_latent = args.w_lat * loss_latent(fake_feature,
                                                    real_feature)  # fake img의 latent vector와 원본 이미지의 latent vector간 l2 loss
            g_loss = err_g_adversarial + err_g_context + err_g_latent

            g_loss.backward(retain_graph=True)
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(batch_loader), err_d.item(), g_loss.item())
            )

            if batches_done % args.sample_interval == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': generator.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                }
                save_image(fake_img.data[:25], train_resimg_save_path + '/%d.png' % batches_done, nrow=5,
                           normalize=True)
                # TODO: add model save code
                torch.save(state, model_save_path + '/' + args.backbone + '_%d.pth' % batches_done)

                # 로딩시엔 아래와 같이 사용
                # model.load_state_dict(state['state_dict'])
                # optimizer.load_state_dict(state['optimizer'])

            batches_done += args.n_critic

            if batches_done % args.sample_interval == 0:
                save_image(fake_img.data[:25], 'train_result/'+args.backbone+"_images/%d.png" % batches_done, nrow=5, normalize=True)
                torch.save(state, model_save_path + '/' + args.backbone + '_%d.pth' % batches_done)

            batches_done += args.n_critic