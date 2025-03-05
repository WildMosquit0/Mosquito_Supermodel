import os
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def get_bbox_dimensions(image_path: str, annotation_path: str) -> pd.DataFrame:
    """
    Processes a single image and its YOLO annotation file.
    
    For each bounding box, computes:
      - Original bounding box width, height, and area in pixels (using original image dimensions).
      - Normalized values (width, height, area) as provided (fraction of the image dimensions).
      - Resized bounding box dimensions in pixels, assuming the image is resized to 640x640 
        if the original image is larger than 640x640 (otherwise, original dimensions are used).
      
    Returns:
      A pandas DataFrame with columns:
        "BBox_Class", "Original_Image_Width", "Original_Image_Height",
        "Original_BBox_Width_pixels", "Original_BBox_Height_pixels", "Original_BBox_Area_pixels",
        "Original_BBox_Width_norm", "Original_BBox_Height_norm", "Original_BBox_Area_norm",
        "Resized_Image_Width", "Resized_Image_Height",
        "Resized_BBox_Width_pixels", "Resized_BBox_Height_pixels", "Resized_BBox_Area_pixels",
        "Resized_BBox_Width_norm", "Resized_BBox_Height_norm", "Resized_BBox_Area_norm".
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return pd.DataFrame()
    
    # Get original image dimensions
    orig_w, orig_h = image.size

    bboxes = []
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found: {annotation_path}")
        return pd.DataFrame()

    with open(annotation_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid annotation format in {annotation_path}: {line}")
                continue

            cls, x_center, y_center, w_norm, h_norm = parts
            # Convert normalized coordinates to float
            x_center, y_center, w_norm, h_norm = float(x_center), float(y_center), float(w_norm), float(h_norm)
            
            # Original bounding box dimensions in pixels
            bbox_w_orig = w_norm * orig_w
            bbox_h_orig = h_norm * orig_h
            bbox_area_orig = bbox_w_orig * bbox_h_orig
            
            # Normalized values (as provided)
            bbox_w_norm_val = w_norm
            bbox_h_norm_val = h_norm
            bbox_area_norm = w_norm * h_norm

            # Determine resized image dimensions (resize only if larger than 640x640)
            if orig_w > 640 or orig_h > 640:
                resized_w, resized_h = 640, 640
            else:
                resized_w, resized_h = orig_w, orig_h
            
            # Resized bounding box dimensions in pixels
            bbox_w_resized = w_norm * resized_w
            bbox_h_resized = h_norm * resized_h
            bbox_area_resized = bbox_w_resized * bbox_h_resized

            bboxes.append({
                "BBox_Class": cls,
                "Original_Image_Width": orig_w,
                "Original_Image_Height": orig_h,
                "Original_BBox_Width_pixels": bbox_w_orig,
                "Original_BBox_Height_pixels": bbox_h_orig,
                "Original_BBox_Area_pixels": bbox_area_orig,
                "Original_BBox_Width_norm": bbox_w_norm_val,
                "Original_BBox_Height_norm": bbox_h_norm_val,
                "Original_BBox_Area_norm": bbox_area_norm,
                "Resized_Image_Width": resized_w,
                "Resized_Image_Height": resized_h,
                "Resized_BBox_Width_pixels": bbox_w_resized,
                "Resized_BBox_Height_pixels": bbox_h_resized,
                "Resized_BBox_Area_pixels": bbox_area_resized,
                "Resized_BBox_Width_norm": bbox_w_norm_val,
                "Resized_BBox_Height_norm": bbox_h_norm_val,
                "Resized_BBox_Area_norm": bbox_area_norm
            })
            
    df = pd.DataFrame(bboxes)
    return df

def process_image_file(filename: str, images_dir: str, annotations_dir: str) -> pd.DataFrame:
    """
    Worker function to process one image file.
    """
    image_path = os.path.join(images_dir, filename)
    annotation_path = os.path.join(annotations_dir, os.path.splitext(filename)[0] + ".txt")
    if os.path.exists(annotation_path):
        df = get_bbox_dimensions(image_path, annotation_path)
        if not df.empty:
            df["Image"] = filename
        return df
    else:
        print(f"No annotation found for image: {filename}")
        return pd.DataFrame()

# Top-level worker function to avoid lambda pickling issues.
def worker_process_file(filename: str, images_dir: str, annotations_dir: str) -> pd.DataFrame:
    return process_image_file(filename, images_dir, annotations_dir)

def get_all_bbox_dimensions(images_dir: str, annotations_dir: str, batch_size: int = 50) -> pd.DataFrame:
    """
    Processes images and their annotations in batches using multiprocessing.
    
    Returns:
      A pandas DataFrame concatenating bounding box details for all images.
    """
    all_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_dfs = []
    with ProcessPoolExecutor() as executor:
        # Use partial to fix images_dir and annotations_dir
        worker = partial(worker_process_file, images_dir=images_dir, annotations_dir=annotations_dir)
        # Process in batches
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            results = list(executor.map(worker, batch))
            for df in results:
                if not df.empty:
                    all_dfs.append(df)
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame()
    return final_df

def aggregate_bbox_area(bbox_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the bounding box areas for each image.
    
    For each image, computes:
      - Total Original_BBox_Area_pixels: sum of Original_BBox_Area_pixels over all bbox in the image.
      - Total Original_BBox_Area_norm: sum of Original_BBox_Area_norm over all bbox in the image.
      - Total Resized_BBox_Area_pixels: sum of Resized_BBox_Area_pixels over all bbox in the image.
      - Total Resized_BBox_Area_norm: sum of Resized_BBox_Area_norm over all bbox in the image.
      
    Returns a DataFrame with one row per image.
    """
    aggregated = bbox_df.groupby("Image").agg({
        "Original_BBox_Area_pixels": "sum",
        "Original_BBox_Area_norm": "sum",
        "Resized_BBox_Area_pixels": "sum",
        "Resized_BBox_Area_norm": "sum"
    }).reset_index()
    
    aggregated.rename(columns={
        "Original_BBox_Area_pixels": "Total_Original_BBox_Area_pixels",
        "Original_BBox_Area_norm": "Total_Original_BBox_Area_norm",
        "Resized_BBox_Area_pixels": "Total_Resized_BBox_Area_pixels",
        "Resized_BBox_Area_norm": "Total_Resized_BBox_Area_norm"
    }, inplace=True)
    return aggregated

if __name__ == "__main__":
    # Set paths for your separate images and annotations directories
    images_directory =      "/home/bohbot/workspace/datasets/mos/all_mos_new/images/train"
    annotations_directory = "/home/bohbot/workspace/datasets/mos/all_mos_new/labels/train"
    
    # Process images in batches with multiprocessing
    bbox_df = get_all_bbox_dimensions(images_directory, annotations_directory, batch_size=50)
    
    if bbox_df.empty:
        print("No bounding box data found.")
    else:
        print("Detailed bounding box data:")
        #print(bbox_df)
        
        # Compute and print mean values for Original and Resized bounding box dimensions (pixels)
        mean_orig_width = bbox_df["Original_BBox_Width_pixels"].mean()
        mean_orig_height = bbox_df["Original_BBox_Height_pixels"].mean()
        mean_orig_area = bbox_df["Original_BBox_Area_pixels"].mean()
        
        mean_resized_width = bbox_df["Resized_BBox_Width_pixels"].mean()
        mean_resized_height = bbox_df["Resized_BBox_Height_pixels"].mean()
        mean_resized_area = bbox_df["Resized_BBox_Area_pixels"].mean()
        
        # Print normalized areas (which are computed as w_norm * h_norm)
        mean_orig_area_norm = bbox_df["Original_BBox_Area_norm"].mean()
        mean_resized_area_norm = bbox_df["Resized_BBox_Area_norm"].mean()
        
        print("\nMean values for Original Bounding Box Dimensions (pixels):")
        print(f"Width: {mean_orig_width:.2f}, Height: {mean_orig_height:.2f}, Area: {mean_orig_area:.2f}")
        print(f"Mean Normalized Area: {mean_orig_area_norm:.4f}")
        
        print("\nMean values for Resized Bounding Box Dimensions (pixels):")
        print(f"Width: {mean_resized_width:.2f}, Height: {mean_resized_height:.2f}, Area: {mean_resized_area:.2f}")
        print(f"Mean Normalized Area: {mean_resized_area_norm:.4f}")
        
        # Aggregate total bounding box area per image and print the result.
        aggregated_df = aggregate_bbox_area(bbox_df)
        print("\nAggregated Bounding Box Area per Image:")
        #print(aggregated_df)
        
        # Optionally, compute the overall mean of the total area across images.
        overall_mean_orig_total = aggregated_df["Total_Original_BBox_Area_pixels"].mean()
        overall_mean_orig_total_norm = aggregated_df["Total_Original_BBox_Area_norm"].mean()
        overall_mean_resized_total = aggregated_df["Total_Resized_BBox_Area_pixels"].mean()
        overall_mean_resized_total_norm = aggregated_df["Total_Resized_BBox_Area_norm"].mean()
        
        print("\nOverall Mean Total Bounding Box Area per Image:")
        print(f"Original (pixels): {overall_mean_orig_total:.2f}, Normalized: {overall_mean_orig_total_norm:.4f}")
        print(f"Resized (pixels): {overall_mean_resized_total:.2f}, Normalized: {overall_mean_resized_total_norm:.4f}")
