from typing import Dict, List, Union, Tuple
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests
import numpy as np
import os
import cv2

from pixelagent.openai import Agent, tool
from ultralytics import YOLO

class ObjectDetectionAgent:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Initialize the object detection agent with a specific YOLO model.
        
        Args:
            model_name: Name or path of the YOLO model to use
        """
        self.model = YOLO(model_name)
        
        # Create the agent with appropriate tools
        self.agent = Agent(
            name="object_detector",
            system_prompt="""You are an object detection expert. 
            You can analyze images to detect objects and provide detailed information about them.
            For each detection, provide the object class, confidence score, and bounding box coordinates.
            When appropriate, organize detections by category and highlight the most confident detections.
            When showing results, refer to the annotated image with bounding boxes.
            """,
            tools=[self.detect_objects, self.get_model_info],
            reset=True,
        )
    
    @tool
    def detect_objects(self, image_url: str, conf_threshold: float = 0.25, save_path: str = "detection_result.jpg") -> Dict:
        """
        Detect objects in an image with confidence scores and bounding boxes.
        Also generates an annotated image with bounding boxes.
        
        Args:
            image_url: URL or local path to the image
            conf_threshold: Minimum confidence threshold for detections (0.0-1.0)
            save_path: Path to save the annotated image
            
        Returns:
            Dictionary with detection information and path to annotated image
        """
        # Handle URL or local path
        if image_url.startswith(('http://', 'https://')):
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            original_img = img.copy()
        else:
            img_array = cv2.imread(image_url)
            original_img = Image.open(image_url).convert("RGB")
        
        # Run inference
        results = self.model(img_array, conf=conf_threshold)
        
        # Extract and format results
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                detection = {
                    "class": result.names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": {
                        "x1": float(box.xyxy[0][0].item()),
                        "y1": float(box.xyxy[0][1].item()),
                        "x2": float(box.xyxy[0][2].item()),
                        "y2": float(box.xyxy[0][3].item()),
                    }
                }
                detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create annotated image
        annotated_img_path = self._create_annotated_image(original_img, detections, save_path)
        
        return {
            "detections": detections,
            "annotated_image_path": annotated_img_path,
            "total_objects": len(detections),
            "class_counts": self._count_classes(detections)
        }
    
    def _create_annotated_image(self, image: Image.Image, detections: List[Dict], save_path: str) -> str:
        """
        Create an annotated image with bounding boxes and labels.
        
        Args:
            image: PIL Image object
            detections: List of detection dictionaries
            save_path: Path to save the image
            
        Returns:
            Path to the saved annotated image
        """
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Generate colors for different classes
        class_names = set(d["class"] for d in detections)
        colors = self._generate_colors(len(class_names) or 1)  # Ensure at least 1 color
        class_to_color = dict(zip(class_names, colors))
        
        # Draw bounding boxes and labels
        for det in detections:
            bbox = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]
            
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = class_to_color[cls]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text = f"{cls}: {conf:.2f}"
            text_size = draw.textbbox((0, 0), text, font=font)[2:]
            draw.rectangle([x1, y1 - text_size[1] - 2, x1 + text_size[0], y1], fill=color)
            
            # Draw text
            draw.text((x1, y1 - text_size[1] - 2), text, fill="white", font=font)
        
        # Save the image
        image.save(save_path)
        return save_path
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n):
            hue = i / n
            # Convert HSV to RGB (simplified)
            h = hue * 6
            x = 255 * (1 - abs(h % 2 - 1))
            if h < 1:
                rgb = (255, int(x), 0)
            elif h < 2:
                rgb = (int(x), 255, 0)
            elif h < 3:
                rgb = (0, 255, int(x))
            elif h < 4:
                rgb = (0, int(x), 255)
            elif h < 5:
                rgb = (int(x), 0, 255)
            else:
                rgb = (255, 0, int(x))
            colors.append(rgb)
        return colors
    
    def _count_classes(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count occurrences of each class in detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of class counts
        """
        counts = {}
        for det in detections:
            cls = det["class"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts
    
    @tool
    def get_model_info(self) -> Dict:
        """
        Get information about the currently loaded YOLO model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model.info["model_name"],
            "description": self.model.info["description"],
            "task": self.model.task,
            "classes": self.model.names
        }
    
    def run(self, query: str, image_url: str = None, conf_threshold: float = 0.25, save_path: str = "detection_result.jpg") -> str:
        """
        Run the agent with a query and image.
        
        Args:
            query: User query for the agent
            image_url: URL or path to an image
            conf_threshold: Confidence threshold for detections
            save_path: Path to save the annotated image
            
        Returns:
            Agent's response
        """
        if not image_url:
            return self.agent.run(query)
        
        try:
            # Perform detection directly (not through the tool)
            detect_result = self.detect_objects(image_url, conf_threshold, save_path)
            
            # Create a context message about the detections
            context = f"I've analyzed the image and detected {detect_result['total_objects']} objects. "
            context += f"The annotated image has been saved to '{detect_result['annotated_image_path']}'. "
            
            # Include the image as an attachment for the agent
            return self.agent.run(f"{context}\n\n{query}", attachments=image_url)
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error analyzing image: {str(e)}"
            print(error_message)
            return self.agent.run(f"{error_message}\n\n{query}", attachments=image_url)


# Example usage
def example():
    # Initialize agent with default YOLOv8n model
    detector = ObjectDetectionAgent()
    
    # Sample image URL
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    # Run detection with visualization
    response = detector.run(
        "Analyze this image and tell me what objects you detect", 
        image_url=url,
        conf_threshold=0.25,  # Lower threshold to detect more objects
        save_path="nature_boardwalk_detected.jpg"
    )
    print(response)


if __name__ == "__main__":
    example()