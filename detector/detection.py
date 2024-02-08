import torch
import requests
from PIL import ImageDraw, Image, ImageFont
from typing import List
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import os

font = ImageFont.truetype(
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic"
)


class Detector:
    """Class that implements Owl v2 for object detection."""

    def __init__(self, scaled_width=1920, scaled_height=1080):
        # TODO: lots of hard coded values here, need to make them configurable
        self.device = torch.device("cuda")
        self.model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.ORIGINAL_WIDTH = 960
        self.ORIGINAL_HEIGHT = 540
        self.SCALED_WIDTH = scaled_width
        self.SCALED_HEIGHT = scaled_height

    def detect(self, image, texts: List, threshold=0.25, nms=False):
        """Calls the detector and post-processes the outputs.

        Args:
            image (_type_): Image to run the detector on.
            texts (List): Prompts to run the detector on and find.
            threshold (float, optional): Confidence - defaults to 0.25.

        Returns:
            List[Tuple]: List of detector outputs including bboxes, scores, labels, and unnormalized image.
        """
        outputs, inputs = self._detect(image, texts)
        unnormalized_image = self._get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]]).to(self.device)
        # print(target_sizes)
        # print(image.size[::-1])
        # target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )
        
        all_results = {}
        all_boxes = []
        all_scores = []
        all_labels = []
        for result in results:
            boxes, scores, labels = (
                result["boxes"],
                result["scores"],
                result["labels"],
            )
            all_boxes.extend(boxes.cpu().detach().numpy().tolist())
            all_scores.extend(scores.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
        if nms:
            print("Going to perform nms")
            all_boxes, all_scores, all_labels = self.nms(all_boxes, all_scores, all_labels)
        
        all_results["boxes"] = all_boxes
        all_results["scores"] = all_scores
        all_results["labels"] = all_labels
        return all_results

    def _detect(self, image, texts: List):
        """Runs the detector.

        Args:
            image (_type_): Image to run the detector on.
            texts (List): Prompt list to run the detector on and find.

        Returns:
            _type_: Detector outputs and inputs.
        """
        print(image.size[::-1])
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(
            self.device
        )
        # inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs, inputs

    def _get_preprocessed_image(self, pixel_values):
        """Preprocesses the image for the model.

        Args:
            pixel_values (_type_): Pixel values of the image.

        Returns:
            _type_: Unnormalized image.
        """
        pixel_values = pixel_values.squeeze().cpu().numpy()
        unnormalized_image = (
            pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]
        ) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image
    
    def iou(self, boxA, boxB):
        """Compute the Intersection Over Union of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def nms(self, all_boxes, all_scores, all_labels, iou_threshold=0.5):
        """Apply Non-Maximum Suppression (NMS) to filter detections.

        Args:
            all_scores (list): Scores of the individual predictions for each result.
            all_boxes (list of bboxes): Boxes of the individual predictions for each result.
            all_labels (list): Labels of the individual predictions for each result.
            iou_threshold (float): Threshold value for deciding whether boxes overlap too much.

        Returns:
            list: Scores after applying NMS.
            list: Boxes after applying NMS.
            list: Labels after applying NMS.
        """
        # Indices sorted by score in descending order
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)

        keep_indices = []
        while sorted_indices:
            current_index = sorted_indices.pop(0)
            keep_indices.append(current_index)

            sorted_indices = [
                i for i in sorted_indices
                if self.iou(all_boxes[current_index], all_boxes[i]) < iou_threshold
            ]

        # Filter the results based on the indices to keep
        print(keep_indices)
        filtered_scores = [all_scores[i] for i in keep_indices]
        filtered_boxes = [all_boxes[i] for i in keep_indices]
        filtered_labels = [all_labels[i] for i in keep_indices]

        return filtered_boxes, filtered_scores, filtered_labels


    def scale_bboxes(self, bboxes):
        """
        Scale a list of bounding box coordinates from original image size to new image size.

        Parameters:
        - bboxes: List of bounding box tuples [(x_min, y_min, x_max, y_max), ...]
        - original_size: Tuple (original_width, original_height)
        - new_size: Tuple (new_width, new_height)

        Returns:
        - List of scaled bounding box coordinates.
        """
        # scaled_bboxes = []

        # original_width, original_height = original_size
        # new_width, new_height = new_size

        # Calculate scaling factors for width and height
        scale_x = self.SCALED_WIDTH / self.ORIGINAL_WIDTH
        scale_y = self.SCALED_HEIGHT / self.ORIGINAL_HEIGHT

        # for bbox in bboxes:
        x_min, y_min, x_max, y_max = bboxes
        # Scale bounding box coordinates
        scaled_x_min = int(x_min * scale_x)
        scaled_y_min = int(y_min * scale_y)
        scaled_x_max = int(x_max * scale_x)
        scaled_y_max = int(y_max * scale_y)

        # scaled_bboxes.append(
        #     (scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max)
        # )

        return scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max

    def plot_predictions(
        self, input_img, text_queries, all_scores, all_boxes, all_labels, show_image: bool = True, save_Image: bool = False, file_dir: str = ''
    ):
        """Plot the predictions on the image.

        Args:
            input_img (_type_): Image to plot the predictions on.
            text_queries (_type_): Prompts that the detector was run on.
            all_scores (_type_): Scores of the individual predictions for each result.
            all_boxes (_type_): Boxes of the individual predictions for each result.
            all_labels (_type_): Labels of the individual predictions for each result.
            show_image (bool, optional): Whether to show the image. Defaults to True.

        Returns:
            _type_: Copy of the image being plotted.
        """
        input_img_copy = input_img.copy()
        draw = ImageDraw.Draw(input_img_copy)
        
        for score, box, label in zip(all_scores, all_boxes, all_labels):
            box = self.scale_bboxes(tuple(box))
            
            # for box in boxes:
            # Convert the coordinates to integers
            draw.rectangle(xy=((box[0], box[1]), (box[2], box[3])), outline="red")
            draw.text(
                xy=(box[0], box[1]),
                text = f"{text_queries[label]} {str(score)}",
                fill = "red",
                font=font,
            )
                
        if show_image:
            input_img_copy.show()
            
        if save_Image:
            input_img_copy.save(file_dir)
                
        return input_img_copy


if __name__ == "__main__":
    detector = Detector(1600, 900)
    texts = ["Image of a hand"]
    
    imageDir = '/home/chenam14/ws/handover-sim/results/2024-02-08_10-31-26_yang-icra2021_s0_test/000'
    saveDir = '/home/chenam14/ws/handover-sim/results/2024-02-08_10-31-26_yang-icra2021_s0_test/000_detections'
    os.makedirs(saveDir, exist_ok=True)
    
    imageFiles = [f for f in os.listdir(imageDir) if os.path.isfile(os.path.join(imageDir, f))]
    imageFiles.sort()

    # Loop through all image files, loading them as PIL images
    for imageFile in imageFiles:
        # Construct the full image path
        imagePath = os.path.join(imageDir, imageFile)
        saveFile = os.path.join(saveDir, imageFile)
        # Load the image file
        try:
            img = Image.open(imagePath)
            print("img size:", img.size)
            res = detector.detect(img, texts, 0.02, nms=True)
            print("=== Bounding box results ===")
            print(res)
            detector.plot_predictions(
                img, texts, res["scores"], res["boxes"], res["labels"], False, True, saveFile
            )
        except IOError:
            print(f"Error opening {imageFile}")