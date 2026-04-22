import logging
import os
import re
import tempfile
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Tuple
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_extract(zf, extract_dir):
    """Safely extract a ZIP archive, preventing Zip Slip attacks."""
    extract_dir = os.path.realpath(extract_dir)
    for member in zf.infolist():
        member_path = os.path.realpath(os.path.join(extract_dir, member.filename))
        if not member_path.startswith(extract_dir + os.sep) and member_path != extract_dir:
            raise ValueError(f"Unsafe path in zip archive: {member.filename}")
        zf.extract(member, extract_dir)


def _transform_image_url(url: str, size: str = "large") -> str:
    """Transform Clarifai image URL to use specified size."""
    return re.sub(r'/(orig|large|small|thumb)/', f'/{size}/', url)


def _ensure_export_exists(stub, metadata, user_app_id, dataset_id, dataset_version_id):
    """Ensure export exists for the dataset version, create if not."""
    import time
    from clarifai_grpc.grpc.api import service_pb2, resources_pb2
    from clarifai_grpc.grpc.api.status import status_code_pb2

    response = stub.GetDatasetVersion(
        service_pb2.GetDatasetVersionRequest(
            user_app_id=user_app_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
        ),
        metadata=metadata
    )

    if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"GetDatasetVersion failed: {response.status.code} - {response.status.description}")

    export_info = response.dataset_version.export_info
    if export_info.HasField('clarifai_data_protobuf'):
        exp = export_info.clarifai_data_protobuf
        if exp.status.code == status_code_pb2.DATASET_VERSION_EXPORT_SUCCESS:
            logger.info(f"Export already exists, size: {exp.size:,} bytes")
            return exp

    logger.info("Creating dataset version export...")
    request = service_pb2.PutDatasetVersionExportsRequest(
        user_app_id=user_app_id,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        exports=[resources_pb2.DatasetVersionExport(format=resources_pb2.CLARIFAI_DATA_PROTOBUF)],
    )
    export_response = stub.PutDatasetVersionExports(request, metadata=metadata)
    if export_response.status.code != status_code_pb2.SUCCESS:
        raise Exception(
            f"PutDatasetVersionExports failed: {export_response.status.code} - "
            f"{export_response.status.description}"
        )

    max_wait = 300
    start = time.time()
    while time.time() - start < max_wait:
        response = stub.GetDatasetVersion(
            service_pb2.GetDatasetVersionRequest(
                user_app_id=user_app_id,
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
            ),
            metadata=metadata
        )
        if response.dataset_version.export_info.HasField('clarifai_data_protobuf'):
            exp = response.dataset_version.export_info.clarifai_data_protobuf
            if exp.status.code == status_code_pb2.DATASET_VERSION_EXPORT_SUCCESS:
                logger.info(f"Export ready! Size: {exp.size:,} bytes")
                return exp
            elif exp.status.code in (status_code_pb2.DATASET_VERSION_EXPORT_PENDING,
                                      status_code_pb2.DATASET_VERSION_EXPORT_IN_PROGRESS):
                logger.info(f"Export in progress... ({int(time.time() - start)}s)")
                time.sleep(5)
            else:
                raise Exception(f"Export failed: {exp.status.code}")
        else:
            time.sleep(5)

    raise Exception("Timeout waiting for export to complete")


def download_dataset(
    user_id: str,
    app_id: str,
    dataset_id: str,
    dataset_version_id: str,
    pat: str,
    output_dir: str,
    concepts: list,
    **kwargs,
) -> str:
    """Download dataset using dataset version export."""
    import requests
    import pickle
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc, resources_pb2
    from clarifai_grpc.grpc.api.status import status_code_pb2

    logger.info("=" * 80)
    logger.info("STEP 1: Downloading Dataset from Clarifai API (Export Method)")
    logger.info("=" * 80)
    logger.info(f"User: {user_id}, App: {app_id}, Dataset: {dataset_id}")
    logger.info(f"Dataset version ID: {dataset_version_id}")
    logger.info(f"Concepts: {concepts}")

    if not concepts:
        raise ValueError("Concepts are required for dataset download")

    stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
    metadata = (("authorization", f"Key {pat}"),)
    user_app_id = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

    logger.info("Checking/creating dataset version export...")
    export = _ensure_export_exists(stub, metadata, user_app_id, dataset_id, dataset_version_id)

    export_url = export.url
    logger.info(f"Export size: {export.size:,} bytes")

    logger.info("Downloading export archive...")
    headers = {'Authorization': f'Key {pat}'}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "export.zip")
        downloaded_bytes = 0
        with requests.get(export_url, headers=headers, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
        logger.info(f"Downloaded {downloaded_bytes:,} bytes")

        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            _safe_extract(zf, extract_dir)

        all_dir = os.path.join(extract_dir, "all")
        if not os.path.exists(all_dir):
            raise Exception("'all' directory not found in export")

        all_inputs = []
        for filename in sorted(os.listdir(all_dir)):
            if filename.startswith("inputbatch"):
                filepath = os.path.join(all_dir, filename)
                logger.info(f"Reading: {filename}")
                with open(filepath, "rb") as f:
                    input_batch = resources_pb2.InputBatch()
                    input_batch.ParseFromString(f.read())
                    logger.info(f"  Inputs in batch: {len(input_batch.inputs)}")
                    all_inputs.extend(input_batch.inputs)

        logger.info(f"Total inputs loaded: {len(all_inputs)}")

        filtered_inputs = []
        for inp in all_inputs:
            regions = []
            if inp.data.regions:
                for region in inp.data.regions:
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
            if regions:
                filtered_inputs.append((inp, regions))

        logger.info(f"Found {len(filtered_inputs)} inputs with target concepts")
        if not filtered_inputs:
            raise Exception(f"No inputs found with concepts: {concepts}")

        dataset_name = f"dataset_{dataset_id}"
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        downloaded_data = []
        num_workers = int(os.environ.get('NUM_DOWNLOAD_WORKERS', '20'))

        def download_single_image(inp_data: Tuple) -> dict:
            inp, regions = inp_data
            try:
                if inp.data.image.url:
                    image_url = _transform_image_url(inp.data.image.url, "large")
                    resp = requests.get(image_url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    image_bytes = resp.content
                elif inp.data.image.base64:
                    import base64
                    image_bytes = base64.b64decode(inp.data.image.base64)
                else:
                    return {'input_id': inp.id, 'success': False, 'error': 'No image'}
                return {'input_id': inp.id, 'image_bytes': image_bytes, 'regions': regions, 'success': True}
            except Exception as e:
                logger.warning(f"Failed to download {inp.id}: {e}")
                return {'input_id': inp.id, 'success': False, 'error': str(e)}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(download_single_image, pair): pair for pair in filtered_inputs}
            logger.info(f"Submitted {len(futures)} download tasks")
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    downloaded_data.append(result)
                    if len(downloaded_data) % 10 == 0:
                        logger.info(f"Downloaded {len(downloaded_data)}/{len(filtered_inputs)} images")

        logger.info(f"Successfully downloaded {len(downloaded_data)} images")
        if not downloaded_data:
            raise Exception("No images were successfully downloaded")

        cache_file = os.path.join(dataset_dir, "downloaded_data.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({'data': downloaded_data, 'concepts': concepts}, f)
        logger.info(f"Cached dataset to: {cache_file}")

        return dataset_name


def convert_dataset_to_coco_format(
    dataset_name: str,
    dataset_split: str,
    output_root: str,
    data_dir: str = None,
) -> NamedTuple("ConvertOutput", [("images_output_root", str), ("annotations_path", str)]):
    """Convert downloaded dataset to COCO format for training."""
    from PIL import Image
    import pickle

    logger.info("=" * 80)
    logger.info("STEP 2: Converting Dataset to COCO Format")
    logger.info("=" * 80)

    dataset_dir = os.path.join(output_root, dataset_name)
    cache_file = os.path.join(dataset_dir, "downloaded_data.pkl")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cached data not found: {cache_file}")

    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)

    downloaded_data = cached['data']
    concepts = cached['concepts']
    logger.info(f"Loaded {len(downloaded_data)} cached images")

    concept_to_idx = {c: idx for idx, c in enumerate(sorted(concepts))}
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
            continue

        try:
            img = Image.open(BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            filename = f"{input_id}.jpg"
            img.save(os.path.join(images_dir, filename), 'JPEG', quality=95)

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
        except Exception as e:
            logger.warning(f"Failed to process {input_id}: {e}")

    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    logger.info(f"Conversion complete! Images saved: {images_saved}")
    logger.info(f"Total annotations: {len(coco_data['annotations'])}")

    ConvertOutputTuple = NamedTuple("ConvertOutput", [("images_output_root", str), ("annotations_path", str)])
    return ConvertOutputTuple(data_prefix, annotation_file)


def create_classes_file(dataset_name: str, output_dir: str, concepts: list = None) -> str:
    """Create classes.txt file for training."""
    import pickle
    from pathlib import Path

    classes_path = os.path.join(output_dir, "classes.txt")

    if not concepts:
        cache_paths = [
            Path(output_dir).parent / "downloaded_data.pkl",
            Path(output_dir).parent.parent / dataset_name / "downloaded_data.pkl",
        ]
        for cache_path in cache_paths:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    concepts = pickle.load(f).get('concepts', [])
                break

    if not concepts:
        raise Exception("No concepts found")

    sorted_concepts = sorted(concepts)
    with open(classes_path, 'w') as f:
        f.write('\n'.join(sorted_concepts))
    logger.info(f"Created classes.txt with {len(sorted_concepts)} classes")
    return classes_path
