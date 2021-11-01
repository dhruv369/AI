dropout=.007
class Net(nn.Module):
    def __init__(self,p_normalization_type):
      ##if (p_normalization_type == 'B'):
        super(Net, self).__init__()
        self.p_normalization_type=p_normalization_type
        # Input Block
        if (self.p_normalization_type == 'B'):
          print('BN in progress')
          self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
          ) # output_size = 26 receptive field  : 3
        elif (self.p_normalization_type == 'L'):
          print('LN in progress')
          self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((8,26,26),elementwise_affine = False),
            nn.ReLU()
          ) # output_size = 26 receptive field  : 3
        elif (self.p_normalization_type == 'G'):
          print('GN in progress')
          self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4,8),
            nn.ReLU()
          ) # output_size = 26 receptive field  : 3
        
        
        # Input Block
        if (self.p_normalization_type == 'B'):
          self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
          ) # output_size = 24 receptive field  : 5
        elif (self.p_normalization_type == 'L'):
          self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((8,24,24),elementwise_affine = False),
            nn.ReLU()
          ) # output_size = 24 receptive field  : 5
        elif (self.p_normalization_type == 'G'):
          self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4,8),
            nn.ReLU()
          ) # output_size = 24 receptive field  : 5
        
        
        
        if (self.p_normalization_type == 'B'):
          self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 22 receptive field  : 7
        elif (self.p_normalization_type == 'L'):
          self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((8,22,22),elementwise_affine = False),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 22 receptive field  : 7
        elif (self.p_normalization_type == 'G'):
          self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4,8),
            nn.ReLU()
          ) # output_size = 24 receptive field  : 5
        
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11 receptive field  : 8
        
        
        if (self.p_normalization_type == 'B'):
          self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 9 receptive field  : 12
        elif (self.p_normalization_type == 'L'):
          self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16,9,9),elementwise_affine = False),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 9 receptive field  : 12
        elif (self.p_normalization_type == 'G'):
          self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4,16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 9 receptive field  : 12
        
        
        if (self.p_normalization_type == 'B'):
          self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16
        elif (self.p_normalization_type == 'L'):
          self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.LayerNorm((16,7,7),elementwise_affine = False),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16
        elif (self.p_normalization_type == 'G'):
          self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(4,16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16


        if (self.p_normalization_type == 'B'):
          self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16
        elif (self.p_normalization_type == 'L'):
          self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.LayerNorm((8,7,7),elementwise_affine = False),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16
        elif (self.p_normalization_type == 'G'):
          self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(4,8),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 7 receptive field  : 16

        
        if (self.p_normalization_type == 'B'):
          self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 5 receptive field  : 20
        elif (self.p_normalization_type == 'L'):
          self.convblock7 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
          nn.LayerNorm((32,5,5),elementwise_affine = False),
          nn.Dropout(dropout),
          nn.ReLU()
          ) # output_size = 5 receptive field  : 20
        elif (self.p_normalization_type == 'G'):
          self.convblock7 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
          nn.GroupNorm(4,32),
          nn.Dropout(dropout),
          nn.ReLU()
          ) # output_size = 5 receptive field  : 20


        if (self.p_normalization_type == 'B'):
          self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 5 receptive field  : 20
        elif (self.p_normalization_type == 'L'):
          self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.LayerNorm((16,5,5),elementwise_affine = False),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 5 receptive field  : 20
        elif (self.p_normalization_type == 'G'):
          self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
             nn.GroupNorm(4,16),
            nn.Dropout(dropout),
            nn.ReLU()
          ) # output_size = 5 receptive field  : 20
        
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1 receptive field  : 24

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),         
           
        ) #output_size = 1 receptive field  : 28


    def forward(self, x):
      ##if (p_normalization_type == 'B'):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)