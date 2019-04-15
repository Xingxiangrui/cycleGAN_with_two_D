import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

'''
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
'''

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        if opt.isTrain:
            self.dir_B_class = os.path.join(opt.dataroot, opt.phase + 'B_class')  # create a path '/path/to/data/trainB'
            self.dir_B_defect = os.path.join(opt.dataroot, opt.phase + 'B_defect')  # create a path '/path/to/data/trainB'
        else:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'

        if opt.isTrain:
            self.B_class_paths = sorted(make_dataset(self.dir_B_class, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
            self.B_defect_paths = sorted(make_dataset(self.dir_B_defect, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        else:
            self.B_paths = sorted(
                make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        #self.B_size = len(self.B_paths)  # get the size of dataset B
        #self.B_size = len(self.B_paths)
        if opt.isTrain:
            self.B_class_size = len(self.B_class_paths)
            self.B_defect_size= len(self.B_defect_paths)  # get the size of dataset B
        else:
            self.B_size = len(self.B_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.isTrain:
            if self.opt.serial_batches:   # make sure index is within then range
                #index_B       = index % self.B_size
                index_B_class = index % self.B_class_siz
                index_B_defect= index % self.B_defect_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                #index_B       = random.randint(0, self.B_size - 1)
                index_B_class = random.randint(0, self.B_class_size - 1)
                index_B_defect= random.randint(0, self.B_defect_size - 1)
        else:
            if self.opt.serial_batches:   # make sure index is within then range
                index_B       = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B       = random.randint(0, self.B_size - 1)


        #B_path        = self.B_paths[index_B]
        if self.opt.isTrain:
            B_class_path = self.B_class_paths[index_B_class]
            B_defect_path= self.B_defect_paths[index_B_defect]
            B_class_img = Image.open(B_class_path).convert('RGB')
            B_defect_img = Image.open(B_defect_path).convert('RGB')
            B_class = self.transform_B(B_class_img)
            B_defect = self.transform_B(B_defect_img)
        else:
            B_path = self.B_paths[index_B]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)

        A_img = Image.open(A_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)

        if self.opt.isTrain:
            return {'A': A, 'B_class': B_class, 'B_defect':B_defect, 'A_paths': A_path, 'B_class_paths': B_class_path, 'B_defect_paths': B_defect_path }
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """

        if self.opt.isTrain:
            return max(self.A_size, self.B_class_size, self.B_defect_size)
        else:
            return max(self.A_size, self.B_size)
