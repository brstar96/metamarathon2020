import torch, argparse, os, warnings
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from define_networks import Generator, Discriminator
from dataloader import metamatathonDataset, custom_transforms
from loss import calculate_gradient_penalty
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', choices=['train', 'infer'])
parser.add_argument('--backbone', type=str, default='se_resnet50',
                    choices=['efficientnet-b0, efficientnet-b1, efficientnet_b2, efficientnet_b3, resnet50, resnet101, inceptionv4, mobilenet_v2, se_resnet50, se_resnet101'])
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate") # default : 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=30, help="interval betwen image samples")
parser.add_argument("--num_workers", type=int, default=4, help="interval betwen image samples")
parser.add_argument("--DATASET_PATH", type=str, default='./data/wetransfer-a25d97/')
args = parser.parse_args()
print(args)

os.makedirs('train_result/'+args.backbone+"_images", exist_ok=True)
img_shape = (args.channels, args.img_size, args.img_size)
warnings.filterwarnings('ignore')
writer = SummaryWriter()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0=2080ti, 1=RTX TITAN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

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
        img = img.to('cuda')

        # Configure input
        real_imgs = Variable(img.type(Tensor))

        #  Train Discriminator
        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_imgs = generator(img)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)

        # Gradient penalty
        # TODO: Gradient penalty debugging
        # gradient_penalty = calculate_gradient_penalty(device, args.batch_size, discriminator, real_imgs.data, fake_imgs.data, lambda_gp)

        # Adversarial loss
        # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train Generator(Train the generator every n_critic steps)
        if i % args.n_critic == 0:
            # Generate a batch of images
            fake_imgs = generator(img)

            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(batch_loader), d_loss.item(), g_loss.item())
            )

            if batches_done % args.sample_interval == 0:
                save_image(fake_imgs.data[:25], 'train_result/'+args.backbone+"_images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += args.n_critic