import torch 
import torch.nn as nn 

class Generator(nn.Module):
    def __init__(self, channel_noise ,img_channels):

        # print("CHANNEL NOISE",channel_noise)
        # print("IMG_CHANNELS",img_channels)
        super(Generator,self).__init__()

        self.gen = nn.Sequential(
            self.conv_block(channel_noise, 128 ,4,1,0),
            self.conv_block(128 ,64,4,2,1),
            self.conv_block(64 ,32,4,2,1),
            self.conv_block(32 ,16,4,2,1),
            nn.ConvTranspose2d(
                16,img_channels,kernel_size=4,stride=2,padding=1
            ),
            nn.Tanh()
        )


    def conv_block(self, in_channels ,out_channels ,kernel_size ,stride ,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias= False
            ),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return(self.gen(x))


class Discriminator(nn.Module):
    def __init__ (self,img_channel):
        super(Discriminator ,self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(img_channel , out_channels=8 ,kernel_size=4 ,stride=2 ,padding=1),
            nn.LeakyReLU(),

            self.conv_block(8, 16, 4, 2, 1),
            self.conv_block(16, 32, 4, 2, 1),
            self.conv_block(32, 64, 4, 2, 1),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),

        )

    def conv_block(self, input_channel, output_channel , kernel ,stride , padding):
        return nn.Sequential(
            nn.Conv2d(
                input_channel,
                output_channel,
                kernel,
                stride,
                padding, 
                bias=False
            ),
            nn.LeakyReLU(0.2,)
        )
    
    def forward(self,x):
        return self.disc(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    batch_size, in_channels ,H,W = 8,3,64,64
    noise_dim  = 100 

    # GENERATOR intialization 
    gen = Generator(noise_dim ,in_channels)
    z = torch.randn((batch_size,noise_dim,1,1))
    assert gen(z).shape == (batch_size ,in_channels,H,W)
    output = gen(z)
    print("GENERATOR INTPUT SHAPE",z.shape)
    print("GENERATOR OUTPUT SHAPE" ,output.shape)
    print("\n")

    # DISCRIMINATOR intialization 
    x = torch.randn((batch_size, in_channels,H,W))
    print("DESCRIMINATOR INPUT SHAPE",x.shape)
    disc = Discriminator(in_channels)
    assert disc(x).shape == (batch_size, 1, 1, 1)
    discriminator_output = disc(x)
    print("DISCRIMATION OUTPUT SHAPE" ,discriminator_output.shape)
    print("SUCCESS MODEL ARCHITECTURE SETTTED UP")

test()