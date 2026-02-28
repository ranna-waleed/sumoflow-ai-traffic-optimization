import os
import cv2
import xml.etree.ElementTree as ET

# The 7 classes exactly as defined in dataset.yaml
CLASSES = {
    0: 'car',
    1: 'bus',
    2: 'truck',
    3: 'motorcycle',
    4: 'taxi',
    5: 'microbus',
    6: 'bicycle'
}

def create_voc_xml(image_path, yolo_txt_path, output_xml_path):
    # Read image to get actual dimensions (crucial for VOC pixel coordinates)
    img = cv2.imread(image_path)
    if img is None:
        return
    height, width, depth = img.shape

    # Setup XML structure
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = os.path.basename(image_path)
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    if not os.path.exists(yolo_txt_path):
        return

    # Read YOLO txt file and convert coordinates
    with open(yolo_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # Convert YOLO (normalized) to Pascal VOC (absolute pixels)
            xmin = int((x_center - w / 2) * width)
            xmax = int((x_center + w / 2) * width)
            ymin = int((y_center - h / 2) * height)
            ymax = int((y_center + h / 2) * height)

            # Add object to XML
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = CLASSES.get(class_id, 'unknown')
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(max(1, xmin))
            ET.SubElement(bndbox, 'ymin').text = str(max(1, ymin))
            ET.SubElement(bndbox, 'xmax').text = str(min(width - 1, xmax))
            ET.SubElement(bndbox, 'ymax').text = str(min(height - 1, ymax))

    # Save the XML file
    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

def convert_all_splits():
    base_dir = "detection/dataset"
    splits = ["train", "val", "test"]
    
    for split in splits:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        
        # Fast R-CNN convention: put XMLs in an "annotations" folder
        xml_dir = os.path.join(base_dir, "annotations", split)
        os.makedirs(xml_dir, exist_ok=True)
        
        if not os.path.exists(img_dir): 
            print(f"Skipping {split} - folder not found.")
            continue
            
        images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        print(f"Converting {split} split ({len(images)} images)...")
        
        for img_name in images:
            txt_name = img_name.replace('.png', '.txt')
            xml_name = img_name.replace('.png', '.xml')
            
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(lbl_dir, txt_name)
            xml_path = os.path.join(xml_dir, xml_name)
            
            if os.path.exists(txt_path):
                create_voc_xml(img_path, txt_path, xml_path)

    print("\nSuccess! All YOLO labels have been converted to Pascal VOC XML format.")
    print("Check the new 'detection/dataset/annotations/' folder!")

if __name__ == "__main__":
    convert_all_splits()