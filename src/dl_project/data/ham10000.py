"""HAM10000 dataset, metadata encoding and image transforms (PLAN_BTL.md sections 5.3-5.5).

Leakage rules implemented here:
* Images are assigned to splits only through the committed manifest (``data/splits/manifest.json``).
* Every data-dependent statistic (age mean/std, class counts/prior) is fitted on the TRAIN split only
  (``MetadataEncoder.fit`` and ``class_counts`` receive train records; callers must not pass val/test).
* Image normalisation uses fixed ImageNet statistics because the backbones are ImageNet-pretrained,
  so no image statistic is estimated from the data at all.

The metadata encoder is pure Python so it can be tested and serialised without PyTorch; the
Dataset and transforms import torch/torchvision lazily.
"""
from collections import Counter
import json
from pathlib import Path

from .splits import load_manifest, read_metadata

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
# Defined before looking at any test result (PLAN_BTL.md 5.9): clinically important minority classes.
RARE_CLASSES = ("akiec", "bcc", "df", "mel", "vasc")

SEXES = ("male", "female", "unknown")
# Fixed HAM10000 vocabulary (index 0 = unknown, also used for unseen values, e.g. from PAD-UFES-20).
LOCALIZATIONS = (
    "unknown", "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand",
    "lower extremity", "neck", "scalp", "trunk", "upper extremity",
)
LOCALIZATION_TO_INDEX = {name: index for index, name in enumerate(LOCALIZATIONS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_DIR_GLOB = "HAM10000_images_part_*"


# ----------------------------------------------------------------- metadata
def parse_age(value):
    """Return age as float, or None when missing ('' / 'nan'). Age 0 is kept (infants exist in HAM10000)."""
    value = (value or "").strip().lower()
    if value in ("", "nan", "none"):
        return None
    return float(value)


class MetadataEncoder:
    """Encode (age, sex, localization) into a numeric vector + a localization index.

    numeric = [age_z, age_missing, sex_male, sex_female, sex_unknown]  (NUM_NUMERIC = 5)
    age_z uses mean/std fitted on training records; a missing age is encoded as age_z = 0 (the train
    mean) with age_missing = 1.
    """

    NUM_NUMERIC = 2 + len(SEXES)
    NUM_LOCALIZATIONS = len(LOCALIZATIONS)

    def __init__(self, age_mean=None, age_std=None):
        self.age_mean = age_mean
        self.age_std = age_std

    @property
    def fitted(self):
        return self.age_mean is not None and self.age_std is not None

    def fit(self, train_records):
        ages = [age for age in (parse_age(record.get("age")) for record in train_records) if age is not None]
        if not ages:
            raise ValueError("Cannot fit MetadataEncoder: no training record has an age.")
        mean = sum(ages) / len(ages)
        variance = sum((age - mean) ** 2 for age in ages) / len(ages)
        self.age_mean = mean
        self.age_std = variance ** 0.5 or 1.0
        return self

    def encode(self, record):
        if not self.fitted:
            raise RuntimeError("MetadataEncoder must be fitted on training records first.")
        age = parse_age(record.get("age"))
        age_z = 0.0 if age is None else (age - self.age_mean) / self.age_std
        sex = (record.get("sex") or "unknown").strip().lower()
        sex = sex if sex in SEXES else "unknown"
        numeric = [age_z, 1.0 if age is None else 0.0] + [1.0 if sex == name else 0.0 for name in SEXES]
        location = (record.get("localization") or "unknown").strip().lower()
        return numeric, LOCALIZATION_TO_INDEX.get(location, 0)

    def state_dict(self):
        return {"age_mean": self.age_mean, "age_std": self.age_std,
                "sexes": list(SEXES), "localizations": list(LOCALIZATIONS)}

    @classmethod
    def from_state_dict(cls, state):
        if list(state.get("sexes", SEXES)) != list(SEXES) or \
                list(state.get("localizations", LOCALIZATIONS)) != list(LOCALIZATIONS):
            raise ValueError("Encoder vocabulary differs from this code version.")
        return cls(state["age_mean"], state["age_std"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def class_counts(train_records):
    """Per-class image counts in CLASSES order. Pass TRAIN records only."""
    counts = Counter(record["dx"] for record in train_records)
    unknown = sorted(set(counts) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown dx label(s): {unknown}")
    return [counts.get(name, 0) for name in CLASSES]


def class_prior(train_records):
    counts = class_counts(train_records)
    total = sum(counts)
    return [count / total for count in counts]


# ------------------------------------------------------------- files/splits
def index_images(image_root):
    """Map image_id -> path for all .jpg files in HAM10000_images_part_* under ``image_root``."""
    image_root = Path(image_root)
    index = {}
    for folder in sorted(image_root.glob(IMAGE_DIR_GLOB)):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.jpg"):
            # On case-insensitive file systems a duplicated lower-case copy would map to the same id.
            index.setdefault(path.stem, path)
    if not index:
        raise FileNotFoundError(f"No images found in {image_root}/{IMAGE_DIR_GLOB}; see scripts/download_data.md.")
    return index


def load_split_records(dataset_root, manifest_path, metadata_name="HAM10000_metadata.csv"):
    """Return {"train": [...], "validation": [...], "test": [...]} records following the manifest."""
    records = read_metadata(Path(dataset_root) / metadata_name)
    manifest = load_manifest(manifest_path)
    assignments = manifest["splits"]
    missing = sorted({record["image_id"] for record in records} ^ set(assignments))
    if missing:
        raise ValueError(f"Metadata and manifest disagree on {len(missing)} image id(s), e.g. {missing[:5]}. "
                         "Run scripts/make_splits.py --verify.")
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        splits[assignments[record["image_id"]]].append(record)
    return splits


def subsample_records(records, fraction, seed):
    """Lesion-grouped, class-stratified subsample of training records (experiment E12)."""
    import random

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1:
        return list(records)
    lesions_by_class = {}
    for record in records:
        lesions_by_class.setdefault(record["dx"], set()).add(record["lesion_id"])
    rng = random.Random(seed)
    keep = set()
    for dx in sorted(lesions_by_class):
        lesions = sorted(lesions_by_class[dx])
        keep.update(rng.sample(lesions, max(1, round(len(lesions) * fraction))))
    return [record for record in records if record["lesion_id"] in keep]


# --------------------------------------------------------------- transforms
def build_transforms(train, image_size=224, resize_size=256, domain_aug=False):
    """torchvision transforms. Train: geometric + mild colour aug; eval: resize + centre crop.

    ``domain_aug`` adds smartphone-like corruptions (blur, strong colour shift, JPEG, down/up-scaling),
    an ablation component for the PAD-UFES-20 domain-shift experiment (E8).
    """
    from torchvision import transforms as T

    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return T.Compose([T.Resize(resize_size), T.CenterCrop(image_size), T.ToTensor(), normalize])
    steps = [
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ]
    if domain_aug:
        steps += [
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.5, 2.5))], p=0.3),
            T.RandomApply([_DownUpScale(image_size)], p=0.3),
            T.RandomApply([_JpegCompression()], p=0.3),
        ]
    return T.Compose(steps + [T.ToTensor(), normalize])


class _DownUpScale:
    """Downscale to 25-60% of the size and back (low-resolution smartphone look)."""

    def __init__(self, image_size, min_scale=0.25, max_scale=0.6):
        self.image_size, self.min_scale, self.max_scale = image_size, min_scale, max_scale

    def __call__(self, image):
        import random
        from PIL import Image

        scale = random.uniform(self.min_scale, self.max_scale)
        small = max(16, int(self.image_size * scale))
        return image.resize((small, small), Image.BILINEAR).resize(image.size, Image.BILINEAR)


class _JpegCompression:
    def __init__(self, quality=(20, 60)):
        self.quality = quality

    def __call__(self, image):
        import io
        import random
        from PIL import Image

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# ------------------------------------------------------------------ dataset
try:  # The encoder/split helpers above stay usable without PyTorch (e.g. in CI's data-only env).
    from torch.utils.data import Dataset as _Dataset
except ImportError:  # pragma: no cover
    _Dataset = object


class HAM10000Dataset(_Dataset):
    """Returns dict(image, meta_num, meta_loc, label, image_id) for each record."""

    def __init__(self, records, image_index, encoder, transform=None):
        missing = [record["image_id"] for record in records if record["image_id"] not in image_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image file(s) missing, e.g. {missing[:5]}.")
        self.records = list(records)
        self.image_index = image_index
        self.encoder = encoder
        self.transform = transform
        # Pre-encode metadata once (cheap, avoids repeated parsing in workers).
        self._meta = [encoder.encode(record) for record in self.records]
        self.labels = [CLASS_TO_INDEX[record["dx"]] for record in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from PIL import Image

        record = self.records[index]
        with Image.open(self.image_index[record["image_id"]]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        numeric, location = self._meta[index]
        return {
            "image": image,
            "meta_num": torch.tensor(numeric, dtype=torch.float32),
            "meta_loc": torch.tensor(location, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "image_id": record["image_id"],
        }


def build_datasets(dataset_root, manifest_path, image_size=224, domain_aug=False, train_fraction=1.0,
                   subsample_seed=0):
    """Create train/validation/test datasets with an encoder fitted on train only.

    Returns (datasets: dict, encoder, train_records_used).
    """
    splits = load_split_records(dataset_root, manifest_path)
    train_records = subsample_records(splits["train"], train_fraction, subsample_seed)
    encoder = MetadataEncoder().fit(train_records)
    image_index = index_images(dataset_root)
    eval_transform = build_transforms(False, image_size)
    datasets = {
        "train": HAM10000Dataset(train_records, image_index, encoder,
                                 build_transforms(True, image_size, domain_aug=domain_aug)),
        "validation": HAM10000Dataset(splits["validation"], image_index, encoder, eval_transform),
        "test": HAM10000Dataset(splits["test"], image_index, encoder, eval_transform),
    }
    return datasets, encoder, train_records
