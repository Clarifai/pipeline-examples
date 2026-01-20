import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Tuple
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dataset(
    user_id: str,
    app_id: str,
    dataset_id: str,
    pat: str,
    output_dir: str,
    eid: int = 0,
    concepts: list = None,
) -> str:
    logger.info("=" * 80)
    logger.info("STEP 1: Downloading Dataset from Clarifai API")
    logger.info("=" * 80)
    logger.info(f"User: {user_id}, App: {app_id}, Dataset: {dataset_id}")
    logger.info(f"Concepts: {concepts}")

    if not concepts:
        raise ValueError("Concepts are required for dataset download")

    try:
        from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
        from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
        from clarifai_grpc.grpc.api.resources_pb2 import UserAppIDSet

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)
        metadata = (("authorization", f"Key {pat}"),)
        user_app_id = UserAppIDSet(user_id=user_id, app_id=app_id)

        logger.info(f"Fetching inputs from dataset: {dataset_id}")
        all_inputs = []
        page = 1
        per_page = 128

        while True:
            list_request = service_pb2.ListDatasetInputsRequest(
                user_app_id=user_app_id,
                dataset_id=dataset_id,
                page=page,
                per_page=per_page,
            )
            response = stub.ListDatasetInputs(list_request, metadata=metadata)

            if response.status.code != 10000:
                raise Exception(f"Failed to list dataset inputs: {response.status.description}")

            if not response.dataset_inputs:
                break

            # Extract inputs from dataset_inputs
            for dataset_input in response.dataset_inputs:
                all_inputs.append(dataset_input.input)

            logger.info(f"Fetched page {page}, total inputs so far: {len(all_inputs)}")

            if len(response.dataset_inputs) < per_page:
                break
            page += 1

        logger.info(f"Total inputs in dataset '{dataset_id}': {len(all_inputs)}")

        logger.info("Filtering inputs with target concepts...")
        filtered_inputs = []

        for input_obj in all_inputs:
            input_id = input_obj.id

            annot_request = service_pb2.ListAnnotationsRequest(
                user_app_id=user_app_id,
                input_ids=[input_id],
                per_page=100,
            )
            annot_response = stub.ListAnnotations(annot_request, metadata=metadata)

            if annot_response.status.code != 10000:
                logger.warning(f"Failed to get annotations for {input_id}: {annot_response.status.description}")
                continue

            has_target_concept = False
            for annotation in annot_response.annotations:
                if annotation.data.regions:
                    for region in annotation.data.regions:
                        for concept in region.data.concepts:
                            if concept.id in concepts and concept.value > 0:
                                has_target_concept = True
                                break
                        if has_target_concept:
                            break
                if has_target_concept:
                    break

            if has_target_concept:
                filtered_inputs.append((input_obj, annot_response.annotations))

        logger.info(f"Found {len(filtered_inputs)} inputs with target concepts")

        if not filtered_inputs:
            raise Exception(f"No inputs found with concepts: {concepts}")

        dataset_name = f"dataset_{dataset_id}"
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        num_workers = int(os.environ.get('NUM_DOWNLOAD_WORKERS', '20'))
        logger.info(f"Downloading images with {num_workers}-worker thread pool...")
        downloaded_data = []

        def download_single_image(input_annot_pair: Tuple) -> dict:
            input_obj, annotations = input_annot_pair

            try:
                if input_obj.data.image.url:
                    import requests
                    headers = {'Authorization': f'Key {pat}'}
                    response = requests.get(input_obj.data.image.url, headers=headers, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                else:
                    import base64
                    image_bytes = base64.b64decode(input_obj.data.image.base64)

                regions = []
                for annotation in annotations:
                    if annotation.data.regions:
                        for region in annotation.data.regions:
                            for concept in region.data.concepts:
                                if concept.id in concepts and concept.value > 0:
                                    bbox = region.region_info.bounding_box
                                    regions.append({
                                        'concept_id': concept.id,
                                        'bbox': {
                                            'left_col': bbox.left_col,
                                            'top_row': bbox.top_row,
                                            'right_col': bbox.right_col,
                                            'bottom_row': bbox.bottom_row,
                                        }
                                    })

                return {
                    'input_id': input_obj.id,
                    'image_bytes': image_bytes,
                    'regions': regions,
                    'success': True
                }

            except Exception as e:
                logger.warning(f"Failed to download {input_obj.id}: {e}")
                return {'input_id': input_obj.id, 'success': False, 'error': str(e)}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(download_single_image, pair): pair for pair in filtered_inputs}
            logger.info(f"Submitted {len(futures)} download tasks to thread pool")

            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    logger.info(f"Task {idx}/{len(futures)} completed: {result.get('input_id', 'unknown')}")
                    if result['success']:
                        downloaded_data.append(result)
                        if len(downloaded_data) % 10 == 0:
                            logger.info(f"Downloaded {len(downloaded_data)}/{len(filtered_inputs)} images")
                except Exception as e:
                    logger.error(f"Task {idx} failed with exception: {e}")
                    import traceback
                    traceback.print_exc()

        logger.info(f"Successfully downloaded {len(downloaded_data)} images")

        if not downloaded_data:
            raise Exception("No images were successfully downloaded. Check authentication and network connectivity.")

        import pickle
        cache_file = os.path.join(dataset_dir, "downloaded_data.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'data': downloaded_data,
                'concepts': concepts,
            }, f)

        logger.info(f"Cached dataset to: {cache_file}")
        logger.info(f"Dataset name: {dataset_name}")

        return dataset_name

    except ImportError as e:
        raise Exception(f"Clarifai gRPC not installed: {e}. Please install: pip install clarifai-grpc")


def convert_dataset_to_coco_format(
    dataset_name: str,
    dataset_split: str,
    output_root: str,
    data_dir: str = None,
) -> NamedTuple(
    "ConvertOutput", [("images_output_root", str), ("annotations_path", str)]
):
    logger.info("=" * 80)
    logger.info("STEP 2: Converting Dataset to COCO Format")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}, Split: {dataset_split}")

    if data_dir and os.path.exists(data_dir):
        logger.info(f"Using existing test data from: {data_dir}")

        annotations_file = os.path.join(data_dir, "annotations.json")
        images_dir = os.path.join(data_dir, "images")

        if os.path.exists(annotations_file) and os.path.exists(images_dir):
            logger.info("Data already in COCO format")
            ConvertOutputTuple = NamedTuple(
                "ConvertOutput",
                [("images_output_root", str), ("annotations_path", str)],
            )
            return ConvertOutputTuple(data_dir, annotations_file)

    from PIL import Image
    import pickle

    dataset_dir = os.path.join(output_root, dataset_name)
    cache_file = os.path.join(dataset_dir, "downloaded_data.pkl")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cached data not found: {cache_file}")

    logger.info(f"Loading cached data from: {cache_file}")
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)

    downloaded_data = cached['data']
    concepts = cached['concepts']

    logger.info(f"Loaded {len(downloaded_data)} cached images")

    if not downloaded_data:
        raise Exception("No images in cached dataset")

    concept_to_idx = {concept_id: idx for idx, concept_id in enumerate(sorted(concepts))}
    logger.info(f"Concept mapping: {concept_to_idx}")

    data_prefix = os.path.join(output_root, dataset_name, dataset_split)
    images_dir = os.path.join(data_prefix, "images")
    annotation_file = os.path.join(data_prefix, "annotations.json")
    os.makedirs(images_dir, exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for idx, concept_id in enumerate(sorted(concepts)):
        coco_data["categories"].append({
            "id": idx,
            "name": concept_id,
            "supercategory": "object"
        })

    annotation_id = 1
    images_saved = 0

    for item in downloaded_data:
        input_id = item['input_id']
        image_bytes = item['image_bytes']
        regions = item['regions']

        if not regions:
            logger.warning(f"Skipping {input_id}: no regions found")
            continue

        try:
            img = Image.open(BytesIO(image_bytes))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            filename = f"{input_id}.jpg"
            img_path = os.path.join(images_dir, filename)
            img.save(img_path, 'JPEG', quality=95)

            image_id = images_saved + 1

            coco_data["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            })

            for region in regions:
                concept_id = region['concept_id']
                category_id = concept_to_idx[concept_id]

                bbox_norm = region['bbox']
                x1 = bbox_norm['left_col'] * width
                y1 = bbox_norm['top_row'] * height
                x2 = bbox_norm['right_col'] * width
                y2 = bbox_norm['bottom_row'] * height

                bbox_width = x2 - x1
                bbox_height = y2 - y1

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

            images_saved += 1

            if images_saved % 100 == 0:
                logger.info(f"Processed {images_saved}/{len(downloaded_data)} images")

        except Exception as e:
            logger.warning(f"Failed to process {input_id}: {e}")
            continue

    logger.info(f"Writing COCO annotations to: {annotation_file}")
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    logger.info("Conversion complete!")
    logger.info(f"  Images saved: {images_saved}")
    logger.info(f"  Images directory: {images_dir}")
    logger.info(f"  Annotations file: {annotation_file}")
    logger.info(f"  Total annotations: {len(coco_data['annotations'])}")

    if images_saved == 0:
        raise Exception("No images were successfully processed during conversion")

    ConvertOutputTuple = NamedTuple(
        "ConvertOutput", [("images_output_root", str), ("annotations_path", str)]
    )
    return ConvertOutputTuple(data_prefix, annotation_file)


def create_classes_file(
    dataset_name: str, output_dir: str, concepts: list = None
) -> str:
    logger.info("Creating classes.txt file...")

    classes_path = os.path.join(output_dir, "classes.txt")

    if not concepts:
        import pickle
        from pathlib import Path

        cache_paths = [
            Path(output_dir).parent / "downloaded_data.pkl",
            Path(output_dir).parent.parent / dataset_name / "downloaded_data.pkl",
        ]

        for cache_path in cache_paths:
            if cache_path.exists():
                logger.info(f"Loading concepts from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                concepts = cached.get('concepts', [])
                break

    if not concepts:
        raise Exception("No concepts found. Concepts must be provided or available in cached data.")

    sorted_concepts = sorted(concepts)
    with open(classes_path, 'w') as f:
        f.write('\n'.join(sorted_concepts))
    logger.info(f"Created classes.txt with {len(sorted_concepts)} classes: {sorted_concepts}")
    return classes_path
