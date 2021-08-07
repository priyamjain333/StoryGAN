import os
import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re
import pdb

class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, counter = None, cache=None, min_len=4, train = True):
        self.lengths = []
        self.followings = []
        dataset = ImageFolder(folder)
        self.dir_path = folder
        self.total_frames = 0
        self.images = []
        self.labels = np.load(folder + 'labels.npy').item()
        if cache is not None and os.path.exists(cache + 'img_cache' + str(min_len) + '.npy') and os.path.exists(cache + 'following_cache' + str(min_len) +  '.npy'):
            self.images = np.load(cache + 'img_cache' + str(min_len) + '.npy')
            self.followings = np.load(cache + 'following_cache' + str(min_len) + '.npy')
        else:
            for idx, (im, _) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                v_name = img_path.replace(folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(folder, ''))
                self.followings.append(following_imgs)
            np.save(folder + 'img_cache' + str(min_len) + '.npy', self.images)
            np.save(folder + 'following_cache' + str(min_len) + '.npy', self.followings)
        train_id, test_id = np.load(self.dir_path + 'train_test_ids.npy')
        orders = train_id if train else test_id
        orders = np.array(orders).astype('int32')
        self.images = self.images[orders]
        self.followings = self.followings[orders]
        print "Total number of clips {}".format(len(self.images))

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = longer // shorter
        se = np.random.randint(0,video_len, 1)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):
        lists = [self.images[item]]
        for i in range(len(self.followings[item])):
            lists.append(str(self.followings[item][i]))
        return lists

    def __len__(self):
        return len(self.images)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.transforms = transform

    def __getitem__(self, item):
        lists = self.dataset[item]
        image = []
        subs = []
        des = []
        text = []
        for v in lists:
            id = v.replace('.png','')
            path = self.dir_path + id + '.png'
            im = PIL.Image.open(path)
            image.append( np.expand_dims(np.array( self.dataset.sample_image(im)), axis = 0) )
        image = np.concatenate(image, axis = 0)
        image = self.transforms(image)
        return {'images': image}

    def __len__(self):
        return len(self.dataset.images)

class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, transform):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.descriptions = np.load(textvec + 'descriptions_vec.npy').item()
        self.attributes = np.load(textvec + 'descriptions_attr.npy').item()
        self.subtitles = np.load(textvec + 'subtitles_vec.npy').item()
        self.descriptions_original = np.load(textvec + 'descriptions.npy').item()  
        self.transforms = transform
        self.labels = dataset.labels

    def save_story(self, output, save_path = './'):
        all_image = []
        images = output['images_numpy']
        texts = output['text']
        for i in range(images.shape[0]):
            all_image.append(np.squeeze(images[i]))
        output = PIL.Image.fromarray(np.concatenate(all_image, axis = 0))
        output.save(save_path + 'image.png')
        fid = open(save_path + 'text.txt', 'w')
        for i in range(len(texts)):
            fid.write(texts[i] +'\n' )
        fid.close()
        return 

    def __getitem__(self, item):
        lists = self.dataset[item]
        labels = []
        image = []
        subs = []
        des = []
        attri = []
        text = []
        for v in lists:
            id = v.replace('.png','')
            path = self.dir_path + id + '.png'
            im = PIL.Image.open(path)
            image.append( np.expand_dims(np.array( self.dataset.sample_image(im)), axis = 0) )
            se = 0
            if len(self.descriptions_original[id]) > 1:
                se = np.random.randint(0,len(self.descriptions_original[id]),1)
                se = se[0]
            text.append(  self.descriptions_original[id][se])
            des.append(np.expand_dims(self.descriptions[id][se], axis = 0))
            subs.append(np.expand_dims(self.subtitles[id][0], axis = 0))
            labels.append(np.expand_dims(self.labels[id], axis = 0))
            attri.append(np.expand_dims(self.attributes[id][se].astype('float32'), axis = 0))
        subs = np.concatenate(subs, axis = 0)
        attri = np.concatenate(attri, axis = 0)
        des = np.concatenate(des, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        image_numpy = np.concatenate(image, axis = 0)
        # image is T x H x W x C
        image = self.transforms(image_numpy)  
        # After transform, image is C x T x H x W
        ##
        des = np.concatenate([des, attri], 1)
        ##

        des = torch.tensor(des)
        subs = torch.tensor(subs)
        attri = torch.tensor(attri)
        labels = torch.tensor(labels.astype(np.float32))
        
        return {'images': image, 'text':text, 'description': des, 
                'subtitle': subs, 'images_numpy':image_numpy, 'labels':labels}

    def __len__(self):
        return len(self.dataset.images)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, textvec, transform):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.transforms = transform
        self.descriptions = np.load(textvec + 'descriptions_vec.npy').item()
        self.attributes =  np.load(textvec + 'descriptions_attr.npy').item()
        self.subtitles = np.load(textvec + 'subtitles_vec.npy').item()
        self.descriptions_original = np.load(textvec + 'descriptions.npy').item()  
        self.transforms = transform
        self.labels = dataset.labels

    def __getitem__(self, item):
        path = self.dir_path + self.dataset[item][0]
        id = self.dataset[item][0].replace('.png','')
        im = PIL.Image.open(path)
        image = np.array( self.dataset.sample_image(im))
        image = self.transforms(image)
        subs = self.subtitles[id][0]
        se = 0
        if len(self.descriptions_original[id]) > 1:
            se = np.random.randint(0,len(self.descriptions_original[id]),1)
            se = se[0]
        des = self.descriptions[id][se]
        attri = self.attributes[id][se].astype('float32')
        text = self.descriptions_original[id][se]
        label = self.labels[id].astype(np.float32)


        lists = self.dataset[item]
        content = []
        attri_content = []
        attri_label = []
        for v in lists:
            id = v.replace('.png','')
            se = 0
            if len(self.descriptions[id]) > 1:
                se = np.random.randint(0,len(self.descriptions[id]),1)
                se = se[0]
            content.append(np.expand_dims(self.descriptions[id][se], axis = 0))
            attri_content.append(np.expand_dims(self.attributes[id][se].astype('float32'), axis = 0))
            attri_label.append(np.expand_dims(self.labels[id].astype('float32'), axis = 0))
        content = np.concatenate(content, axis = 0)
        attri_content = np.concatenate(attri_content, axis = 0)
        attri_label = np.concatenate(attri_label, axis = 0)
        content = np.concatenate([content, attri_content, attri_label], 1)
        des = np.concatenate([des, attri])
        ##
        content = torch.tensor(content)

        return {'images': image, 'text':text, 'description': des,  
            'subtitle': subs, 'labels':label, 'content': content}

    def __len__(self):
        return len(self.dataset.images)




# def video_transform(video, image_transform):
#     vid = []
#     for im in video:
#         vid.append(image_transform(im))

#     vid = torch.stack(vid).permute(1, 0, 2, 3)

#     return vid

# n_channels = 3
# image_transforms = transforms.Compose([
#         PIL.Image.fromarray,
#         #transforms.Resize(int(args["--image_size"])),
#         transforms.ToTensor(),
#         lambda x: x[:n_channels, ::],
#         transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
#     ])
# video_transforms = functools.partial(video_transform, image_transform=image_transforms)


# counter = np.load('/media/bigguy/yl353/pororo_png/frames_counter.npy').item()
# base= VideoFolderDataset('/media/bigguy/yl353/pororo_png/', counter = counter, cache = '/media/bigguy/yl353/pororo_png/')
# imageloader = ImageDataset(base, '/media/bigguy/yl353/pororo_png/', image_transforms)
# storyloader = StoryDataset(base, '/media/bigguy/yl353/pororo_png/', video_transforms)
# temp = storyloader[10]
# #storyloader.save_story(storyloader[10])
# pdb.set_trace()
# aaa = 0
