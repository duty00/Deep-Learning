# Data

Images and metadata are not stored in Git (HAM10000 is CC BY-NC 4.0). Download instructions:
`scripts/download_data.md`. Expected layout:

```text
data/raw/ham10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

`splits/manifest.json` is committed: it maps every `image_id` to train/validation/test (70/15/15, grouped by
`lesion_id`) and stores the SHA-256 of the metadata file it was built from. Check it with
`python scripts/make_splits.py --verify`.
