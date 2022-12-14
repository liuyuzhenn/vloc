# Instruction

## Reconstruction

### Usage

```ps
python src/reconstruct.py --colmap_path /path/to/colmap --work_space /path/to/work_space --img_dir /path/to/image_folder
```
*e.g.*

```ps
python src/reconstruct.py --colmap_path "E:/software/COLMAP-3.7-windows-cuda/COLMAP.bat" --work_space "data/Fort_Channing_gate"  --img_dir  "data/Fort_Channing_gate/img"
```

During reconstruction, the following files will be generated `database.db`, `image-pairs.txt`, `log.txt`, `result.txt`

Once the reconstruction is done, the structure looks like this (* denotes the generated folders/files): 
```
workspace
  ├─sparse  
  │   └─0  
  ├─image-pairs.txt  
  ├─database.db  
  ├─log.txt  
  └─result.txt  
```
### Result

Fort_Channing_gate:

![pic alt](../img/demo/Fort_Channing_gate.png "opt title")

## Extract Features

### Usage

```ps
python src/gen_desc.py --database /path/to/output_file --img_list /path/to/image_list
```

*e.g.*
```ps
python src/gen_desc.py --database "./database.db" --img_list "./example/image_list.txt"
```

After features are extracted, use the `DatabaseOperator` class to load descriptor/keypoints to memory, *e.g.*

```py
db = DatabaseOperator('database.db')
descriptors = DatabaseOperator.fetch_all_descriptors() # load descriptors
keypoints = DatabaseOperator.fetch_all_keypoints() # load keypoints
```

Refer to `DatabaseOperator` in `database.py` for further details.

## Localization

Localize image given a 3D model.

To localize an image, you shold:
- Preprocess (this procedure only needs to be excecuted once for each 3D model).
	- Load 3D model using `Model3D` class.
	- Call function `cluster_model3d` to classify model descriptors.
- Extract keypoints/descriptors from the image.
- Call function `localize`, which returns:
	- `status`: success or not
	- `rvec`: rotation vector
	- `tvec`: translation vector
	- `inliers`: number of inliers used for PnP


An example could be found in file `localize.py`.


### Usage

```ps
python src/localize.py --database /path/to/database --model_dir /path/to/model3d --img_path /path/to/image --num_kps 1000
```
Please refer to `--help` for further details.
