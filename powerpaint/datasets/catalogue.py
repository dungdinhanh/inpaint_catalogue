import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import cv2
from powerpaint.datasets.mask_generator.masks_seg import RandomRectangleMaskWithSegmGenerator, RandomRectangleMaskWithSegmOverlapGenerator

def augment_images(image, mask, resolution):
    mask = Image.fromarray((mask.squeeze(0) * 255).astype("uint8")).convert("L")

    resize = transforms.Resize((resolution, resolution))
    image, mask = resize(image), resize(mask)

    crop = transforms.RandomCrop(resolution)
    image, mask = crop(image), crop(mask)

    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    mask = to_tensor(mask)
    mask[mask != 0] = 1

    normalize = transforms.Normalize([0.5], [0.5])
    image = normalize(image)

    return image, mask


class CatalogueDataset(IterableDataset):
    def __init__(
        self,
        transforms,
        pipeline,
        task_prompt,
        image_root,
        segmentation_root,
        list_file,
        # tokenizer=None,
        resolution=512,
        scheme_ratio=0.5,
        name="HL50M",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.image_root = image_root
        self.segmentation_root = segmentation_root
        self.resolution = resolution
        self.scheme_ratio = scheme_ratio
        self.tokenizer = pipeline.tokenizer 
        self.task_prompt = task_prompt
        self.transforms = transforms # this one is used in laion but not in open image/ let's switch later to see if it can help improvement or not
        self.pipeline = pipeline
        self.image_paths = self._read_list_file(list_file)
        self.mask_gen_a = RandomRectangleMaskWithSegmGenerator() # may be switch to random mask root if necessary
        self.mask_gen_b = RandomRectangleMaskWithSegmOverlapGenerator() # may switch to random mask root if necessary

    def _read_list_file(self, list_file):
        with open(list_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def __iter__(self):
        for rel_path in self.image_paths:
            img_path = os.path.join(self.image_root, rel_path)
            seg_path = os.path.join(self.segmentation_root, rel_path)

            try:
                image = Image.open(img_path).convert("RGB")
                seg_image = Image.open(seg_path).convert("L")
            except Exception:
                continue

            seg = (np.array(seg_image) > 0).astype(np.float32) # becare full here
            # seg = torch.from_numpy(seg).to(self.pipeline.device)
            seg = torch.from_numpy(seg).type(torch.float16).to(self.pipeline.device)
            if random.random() < self.scheme_ratio:
                gen_mask, _ = self.mask_gen_a(seg)
            else:
                gen_mask, _ = self.mask_gen_b(seg)

            img_tensor, mask_tensor = augment_images(image, gen_mask, self.resolution)

            if len(np.unique(mask_tensor)) <= 1:
                continue

            # Use the same context prompt for all
            prompt = self.task_prompt.context_inpainting.placeholder_tokens
            promptA = ""
            promptB = ""

            if self.tokenizer:
                tokens = self.tokenizer(
                    [promptA, promptB, prompt],
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_idsA, input_idsB, input_ids = [x.squeeze(0) for x in tokens.input_ids]
            else:
                input_idsA = promptA
                input_idsB = promptB
                input_ids = prompt

            yield {
                "pixel_values": img_tensor,
                "mask": mask_tensor,
                "prompt": prompt,
                "promptA": promptA,
                "promptB": promptB,
                "input_ids": input_ids,
                "input_idsA": input_idsA,
                "input_idsB": input_idsB,
            }
